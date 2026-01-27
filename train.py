"""
CLI training interface for 2048 AI agent.
Run with: python train.py [command]
"""

from copy import deepcopy

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Optional, TypedDict
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, Dataset

from transformers import get_scheduler
import multiprocessing
import torch
import typer
from torch.nn import functional as F
from torch.optim import AdamW, Muon
from tqdm import tqdm


from game import Game2048, Direction, GameMLP, MLPConfig, GameURM, GameURMConfig, Grid
from logger import MetricLogger


def export_model_to_onnx(model: torch.nn.Module, output_path: str, config: MLPConfig) -> None:
    """
    Export a GameMLP model to ONNX format for browser inference.

    The exported model takes a (batch, 48) input tensor and outputs:
    - action_logits: (batch, 4) - log probabilities for UP/DOWN/LEFT/RIGHT
    - value: (batch, 1) - estimated value of the state
    """
    import onnx

    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input matching to_model_format() output
    dummy_input = torch.randn(1, 48)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["board_state"],
        output_names=["action_logits", "value"],
        dynamo=False,  # Use legacy exporter for simpler models
    )

    # Load and re-save with weights embedded (not external) for browser compatibility
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Convert external data to embedded tensors
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

    load_external_data_for_model(onnx_model, str(output_path.parent))

    # Remove the external data file and save with embedded weights
    external_data_path = output_path.parent / (output_path.name + ".data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save(onnx_model, str(output_path))

    # Save config alongside for JS to know architecture
    config_path = output_path.parent / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def export_best_game_for_demo(episode: "EpisodeData", output_path: str) -> None:
    """
    Export the best game episode to JSON format for the demo website replay.

    Converts the internal EpisodeData format to a clean JSON structure
    suitable for browser visualization.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not episode or not episode.get("moves"):
        print(f"Warning: No valid episode to export")
        return

    direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    def grid_to_values(grid):
        """Convert exponent grid to actual tile values."""
        return [[2**cell if cell > 0 else 0 for cell in row] for row in grid]

    demo_data = {"score": episode["total_points"], "total_steps": episode["total_steps"], "moves": []}

    for i, move in enumerate(episode["moves"]):
        state_before = move.get("state_before", [])
        state_after = move.get("result_state", [])

        move_data = {
            "step": i + 1,
            "state_before": grid_to_values(state_before) if state_before else [],
            "action": direction_names[move["selected_direction"]],
            "state_after": grid_to_values(state_after) if state_after else [],
            "points_earned": move.get("points_earned", 0),
            "entropy": move.get("entropy", 0.0),
        }
        demo_data["moves"].append(move_data)

    with open(output_path, "w") as f:
        json.dump(demo_data, f, indent=2)

    print(f"Exported best game ({episode['total_points']} points, {episode['total_steps']} moves) to {output_path}")


class StepData(TypedDict, total=False):
    """
    Schema for per-step data collected during gameplay and enriched during advantage calculation.

    Fields are grouped by when they're populated:
    - Gameplay fields: set in play_game_for_episode()
    - Advantage fields: set in calculate_advantage()

    Using total=False allows incremental construction while still providing type hints.
    """

    # gameplay: model predictions
    predicted_future_value: float
    selected_direction: int  # index into Direction enum (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
    game_state: torch.Tensor  # flattened model input (48,)
    action_mask: list[bool]  # True = invalid action
    entropy: float  # action distribution entropy
    policy_logprobs: list[float]

    # gameplay: game state
    state_before: Grid  # raw grid before move
    result_state: Grid  # raw grid after move
    points_earned: int  # points from this move
    max_points_possible: int  # best possible points this turn
    points_possible: dict[Direction, int]  # points per direction

    # gameplay: heuristic deltas
    smoothness_delta: float
    max_tile_created: int  # exponent of highest tile created this move
    max_exponent_before: int
    max_exponent_after: int
    corner_delta: float
    adjacency_delta: float
    chain_delta: float
    monotonicity_before: float
    monotonicity_after: float
    emptiness_before: float
    emptiness_after: float
    topological_delta: float
    topological_anchor: tuple[int, int] | None  # not used

    # advantage calculation: set in calculate_advantage()
    reward: float  # immediate reward (absolute + shaped)
    future_reward_raw: float  # discounted return-to-go (unnormalized)
    future_reward: float  # normalized return-to-go
    advantage: float  # future_reward - predicted_future_value


class EpisodeData(TypedDict):
    """Schema for a complete episode/rollout."""

    moves: list[StepData]
    total_points: int
    total_steps: int
    final_state: Grid


app = typer.Typer(help="Train and evaluate 2048 AI agents")


def format_grid(grid: list[list[int]], indent: str = "  ") -> str:
    """
    Format a 2048 grid for pretty printing.
    Grid contains exponents (0 = empty, 1 = 2, 2 = 4, etc.)
    """
    lines = []
    # Find max width needed for any cell
    max_val = max(2**cell if cell > 0 else 0 for row in grid for cell in row)
    cell_width = max(4, len(str(max_val)) + 1)

    # Top border
    lines.append(indent + "┌" + "─" * (cell_width * 4 + 3) + "┐")

    for i, row in enumerate(grid):
        cells = []
        for cell in row:
            if cell == 0:
                cells.append(".".center(cell_width))
            else:
                cells.append(str(2**cell).center(cell_width))
        lines.append(indent + "│" + "│".join(cells) + "│")
        if i < 3:
            lines.append(indent + "├" + "─" * (cell_width * 4 + 3) + "┤")

    # Bottom border
    lines.append(indent + "└" + "─" * (cell_width * 4 + 3) + "┘")

    return "\n".join(lines)


@torch.no_grad
def play_game_for_episode(
    # actor_model: torch.nn.Module,
    # critic_model: torch.nn.Module,
    model: torch.nn.Module,
    max_steps: int | None = None,
    device: torch.device = None,
    seed: int | None = None,
) -> EpisodeData:
    """
    Given a model, play an episode of a game.

    If seed is provided, the game will be deterministic (useful for evaluation).
    """
    if seed is not None:
        random.seed(seed)

    game = Game2048()
    game.reset()

    game_data: list[StepData] = []
    move_count = 0
    total_points = 0

    # this will be a running average
    step = 0

    while game.has_next_step() and (not max_steps or (max_steps > 0 and step < max_steps)):
        step_data: StepData = {}

        # find valid directions
        valid_directions = [d for d in Direction if game.direction_has_step(d)]
        if not valid_directions:
            break

        # we look at the possible gains to determine the overall reward
        possible_gains = game.preview_move_rewards()
        best_direction, highest_points = max(possible_gains.items(), key=lambda item: item[1])

        # capture grid state before the move (for visualization)
        state_before = [row[:] for row in game.grid]

        # now agent makes a decision
        model_input = game.to_model_format()
        if device is not None:
            model_input = model_input.to(device)
        # action_logits, _ = actor_model(model_input.unsqueeze(0))
        # _, predicted_future_value = critic_model(model_input.unsqueeze(0))
        action_logits, predicted_future_value = model(model_input.unsqueeze(0))
        action_logits = action_logits.squeeze(0)
        predicted_future_value = predicted_future_value.squeeze(0)

        # we interret the directions as being UP/DOWN/LEFT/RIGHT:
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        valid_dirs = game.current_valid_directions()
        invalid_mask = [_dir not in valid_dirs for _dir in dirs]

        # extract the action sequence & mask out invalid states
        action_logits[invalid_mask] = -torch.inf

        # now we let the model decide which direction it should move into
        adjusted_action_dist = torch.softmax(action_logits, dim=-1)  # this is the one we want to save
        try:
            selected_action = torch.multinomial(adjusted_action_dist, num_samples=1).item()
        except RuntimeError:
            from IPython import embed

            embed()
        # action: Direction = actor_model.directions[selected_action]
        action: Direction = model.directions[selected_action]

        # compute entropy of the action distribution (model's uncertainty)
        valid_probs = adjusted_action_dist[adjusted_action_dist > 0]
        step_entropy = -(valid_probs * valid_probs.log()).sum().item()

        # take step
        new_state, points_earned, done, info = game.step(action)
        move_count += 1
        total_points += points_earned

        # and then this determines which action was actually selected
        step_data["predicted_future_value"] = (
            predicted_future_value.detach().item()
        )  # this one already comes normalized
        step_data["selected_direction"] = selected_action
        step_data["game_state"] = model_input.detach()
        step_data["state_before"] = state_before  # raw grid before move
        step_data["result_state"] = new_state  # state after the move
        step_data["max_points_possible"] = highest_points
        step_data["points_earned"] = points_earned
        step_data["points_possible"] = possible_gains
        step_data["action_mask"] = invalid_mask
        step_data["smoothness_delta"] = info.get("smoothness_delta", 0.0)
        step_data["max_tile_created"] = info.get("max_tile_created", 0)
        step_data["max_exponent_before"] = info.get("max_exponent_before", 0)
        step_data["max_exponent_after"] = info.get("max_exponent_after", 0)
        step_data["corner_delta"] = info.get("corner_delta", 0.0)
        step_data["adjacency_delta"] = info.get("adjacency_delta", 0.0)
        step_data["chain_delta"] = info.get("chain_delta", 0.0)
        # step_data["monotonicity_delta"] = info.get("monotonicity_delta", 0.0)
        monotonicity_next = info["monotonicity_after"] if not done else 0.0
        step_data["monotonicity_after"] = monotonicity_next
        step_data["monotonicity_before"] = info["monotonicity_before"]
        step_data["emptiness_before"] = info["emptiness_before"]
        step_data["emptiness_after"] = info["emptiness_after"] if not done else 0.0
        step_data["topological_delta"] = info.get("topological_delta", 0.0)
        step_data["entropy"] = step_entropy

        step_data["policy_logprobs"] = (action_logits.log_softmax(dim=-1)).tolist()

        # save the data generated at the current step
        game_data.append(step_data)

        # we dont want to save the terminal state as part of the rollout, since all possible actions would be masked
        # and unless our initial training run results in a perfect model (highly unlikely) we dont need to worry about
        # the edge case of it winning
        if done:
            break  # the `step` variable is `1-indexed` so the final step isn't incremented

        step += 1

    episode_data: EpisodeData = {
        "moves": game_data,
        "total_points": total_points,
        "total_steps": step,
        "final_state": new_state,
    }
    return episode_data


def _worker_play_game(args: tuple) -> EpisodeData:
    """
    Worker function for multiprocessing.
    Reconstructs model from state dict and plays a game.
    """
    state_dict, config_dict, max_steps = args
    config = MLPConfig(**config_dict)
    model = GameMLP(config)
    model.load_state_dict(state_dict)
    model.eval()
    return play_game_for_episode(model, max_steps=max_steps, device=None)


class MyDataset(Dataset):
    def __init__(self, episodes: list[EpisodeData]):
        self.moves = []

        for ep in episodes:
            self.moves.extend(ep["moves"])

        # now we create a single batched input tensor

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx: int):
        move = self.moves[idx]

        return {
            "game_state": move["game_state"],
            "direction_idx": move["selected_direction"],
            "action_mask": move["action_mask"],
            "advantage": move["advantage"],
            "future_reward": move["future_reward"],
            "policy_logprobs": move["policy_logprobs"],
        }


def collate_fn(minibatch: list[dict]):
    # input list of these objects
    #     "game_state": move["game_state"],
    #     "direction_idx": move["selected_direction"],
    #     "action_mask": move["action_mask"],
    #     "advantage": move["advantage"],
    #     "future_reward": move["future_reward"],
    #     "policy_logprobs": move["policy_logprobs"],
    # }

    input_batch = torch.stack([item["game_state"] for item in minibatch])
    target_batch = torch.tensor([item["direction_idx"] for item in minibatch])  # (B,)
    action_mask = torch.tensor([item["action_mask"] for item in minibatch])
    advantage = torch.tensor([item["advantage"] for item in minibatch], dtype=torch.float32)
    future_reward = torch.tensor([item["future_reward"] for item in minibatch])
    old_logprobs = torch.tensor([item["policy_logprobs"] for item in minibatch])
    # old_logprobs = old_policy_logprobs.squeeze(1)

    return {
        "input_batch": input_batch,
        "target_batch": target_batch,
        "action_mask": action_mask,
        "advantage": advantage,
        "future_reward": future_reward,
        "old_logprobs": old_logprobs,
    }


def model_optimize_step(
    # actor_model: torch.nn.Module,
    # critic_model: torch.nn.Module,
    model: torch.nn.Module,
    episodes: list[EpisodeData],
    # actor_optimizer: torch.optim.Optimizer,
    # critic_optimizer: torch.optim.Optimizer,
    optimizer: torch.optim.Optimizer,
    # actor_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    # critic_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    beta: float = 0.1,
    critic_strength: float = 1.0,
    device: torch.device = None,
    batch_size: int = 32,
    kl_strength: float = 0.01,
):
    """
    Performs an optimization step as part of the RL loop.
    """
    # so what we'll do is create a separate input batch for each rollout so it can be multiplied
    # by the advantage
    # so our input data should become [game state] -> [direction label]
    # batches = []
    dataset = MyDataset(episodes)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    # Accumulate stats across all batches
    total_loss = 0.0
    total_policy_loss = 0.0
    total_entropy_loss = 0.0
    total_value_loss = 0.0
    total_grad_norm = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    max_kl = 0.0
    num_batches = 0

    for minibatch in train_loader:
        # now we create a single batched input tensor
        input_batch = minibatch["input_batch"]
        target_batch = minibatch["target_batch"]
        action_mask = minibatch["action_mask"]
        advantage = minibatch["advantage"]
        rtg_batch = minibatch["future_reward"]
        old_policy_logprobs = minibatch["old_logprobs"]

        if device is not None:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            action_mask = action_mask.to(device)
            advantage = advantage.to(device)
            old_policy_logprobs = old_policy_logprobs.to(device)

        # with torch.no_grad():
        #     # normalize advantage for more stability during training
        #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # now that the data is prepared, we compute the loss and take an optimizer step
        # actor_model.train()
        # critic_model.train()
        model.train()

        # action_logits, _ = actor_model(
        #     inputs=input_batch,
        # )
        # _, value_logits = critic_model(
        #     inputs=input_batch,
        # )
        action_logits, value_logits = model(input_batch)

        targets = target_batch.to(device=action_logits.device)

        # convert (B, 1) into (B, 1)
        # targets = targets.squeeze(0)
        masked_action_logits = torch.masked_fill(action_logits, action_mask, float("-inf"))

        # compute the advantaged cross-entropy loss
        new_policy_logprobs = masked_action_logits.log_softmax(dim=-1)  # (B, T)
        assert new_policy_logprobs.shape == old_policy_logprobs.shape, (
            f"{new_policy_logprobs.shape=} == {old_policy_logprobs.shape=}"
        )

        # next, we need to pluck out the actions taken by the agent in order
        new_logprobs = torch.gather(
            new_policy_logprobs,
            dim=-1,
            index=targets.unsqueeze(1),  # we want to extend into this direction
        )
        old_logprobs = torch.gather(
            old_policy_logprobs,
            dim=-1,
            index=targets.unsqueeze(1),
        )

        # PPO-clip importance ratio:
        eps = 0.1
        importance_ratio = (new_logprobs - old_logprobs).squeeze(1).clamp(-20, 20)
        importance_ratio = importance_ratio.exp()
        clipped = importance_ratio.clamp(1 - eps, 1 + eps)
        assert advantage.shape == importance_ratio.shape, f"{advantage.shape} != {importance_ratio.shape}"
        ppo_clip = torch.minimum(advantage * importance_ratio, advantage * clipped)
        # rho = e^( log(p(old)) - log(p(new)) )
        # L_clip = min(A*rho, clip(rho, 1-eps, 1+eps)*A)

        # policy_loss_per_sample = F.cross_entropy(input=masked_action_logits, target=targets, reduction="none")
        policy_loss_per_sample = -ppo_clip
        policy_loss = policy_loss_per_sample.mean()
        # here we compute KL(old||new)
        with torch.no_grad():
            # new policy
            # new_action_logits, _ = actor_model(
            #     inputs=input_batch
            # )
            new_action_logits, _ = model(input_batch)
            new_action_logits: torch.Tensor = new_action_logits
            new_action_logits = torch.masked_fill(new_action_logits, action_mask, float("-inf"))

            # we must compute softmax across the new logits but only
            # where the moves are valid

            # calculate the probs
            # new_action_probs = torch.softmax(new_action_logits, dim=1)

            # next we calculate P(old)/P(new)
            old_probs = torch.masked.softmax(action_logits, mask=~action_mask, dim=1)
            new_probs = torch.masked.softmax(new_action_logits, dim=1, mask=~action_mask)

            # with logprobs

            per_sample_kl = torch.masked.sum(old_probs * (old_probs / new_probs).log(), dim=1, mask=~action_mask)

        # from IPython import embed
        # embed()

        # entropy regularization
        masked_probs = torch.masked.softmax(action_logits, dim=1, mask=~action_mask)
        entropy_per_sample = -torch.masked.sum(masked_probs * (masked_probs + 1e-8).log(), dim=1, mask=~action_mask)
        entropy_loss = -beta * entropy_per_sample.mean()

        # value loss for the learned baseline
        value_logits = value_logits.view(-1)
        value_loss_per_sample = F.smooth_l1_loss(value_logits, rtg_batch, reduction="none")
        value_loss = critic_strength * value_loss_per_sample.mean()

        # combine all losses
        loss = policy_loss + entropy_loss + value_loss

        # now we backprop
        loss.backward()
        # clip gradnorm and take an optimizer step
        # actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_model.parameters(), 1.0)
        # critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # grad_norm = actor_grad_norm + critic_grad_norm
        # actor_optimizer.step()
        # critic_optimizer.step()
        # actor_optimizer.zero_grad()
        # critic_optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        # current_lr = actor_lr_scheduler.get_last_lr()[0] if actor_lr_scheduler is not None else 0.0
        # critic_current_lr = critic_lr_scheduler.get_last_lr()[0] if critic_lr_scheduler is not None else 0.0
        # if actor_lr_scheduler is not None:
        #     actor_lr_scheduler.step()
        # if critic_lr_scheduler is not None:
        #     critic_lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else 0.0

        # here we compute KL(old||new)
        with torch.no_grad():
            # new policy
            # new_action_logits, _ = actor_model(
            #     inputs=input_batch
            # )
            new_action_logits, _ = model(input_batch)
            new_action_logits: torch.Tensor = new_action_logits
            new_action_logits = torch.masked_fill(new_action_logits, action_mask, float("-inf"))

            # we must compute softmax across the new logits but only
            # where the moves are valid

            # calculate the probs
            # new_action_probs = torch.softmax(new_action_logits, dim=1)

            # next we calculate P(old)/P(new)
            old_probs = torch.masked.softmax(action_logits, mask=~action_mask, dim=1)
            new_probs = torch.masked.softmax(new_action_logits, dim=1, mask=~action_mask)

            # with logprobs

            per_sample_kl = torch.masked.sum(old_probs * (old_probs / new_probs).log(), dim=1, mask=~action_mask)

        # Print batch statistics
        batch_kl = per_sample_kl.mean().item()
        batch_entropy = entropy_per_sample.mean().item()
        # print(
        #     f"Batch {num_batches + 1}: loss={loss.item():.4f}, policy={policy_loss.item():.4f}, "
        #     f"value={value_loss.item():.4f}, entropy={batch_entropy:.4f}, kl={batch_kl:.4f}, "
        #     f"grad_norm={grad_norm.item():.4f}"
        # )

        # Accumulate stats
        total_loss += loss.detach().cpu().item()
        total_policy_loss += policy_loss.detach().cpu().item()
        total_entropy_loss += entropy_loss.detach().cpu().item()
        total_value_loss += value_loss.detach().cpu().item()
        total_grad_norm += grad_norm.detach().cpu().item()
        total_entropy += entropy_per_sample.mean().detach().cpu().item()
        total_kl += per_sample_kl.sum().detach().cpu().item()
        max_kl = max(max_kl, per_sample_kl.max().detach().cpu().item())
        num_batches += 1

    # collect averaged stats for logging
    optimizer.scheduler_step()
    stats = {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "entropy_loss": total_entropy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        # "actor_grad_norm": actor_grad_norm.detach().cpu().item(),
        # "critic_grad_norm": critic_grad_norm.detach().cpu().item(),
        "grad_norm": total_grad_norm / num_batches,
        "entropy": total_entropy / num_batches,
        "kl_total": total_kl,
        "kl_average": total_kl / num_batches,
        "kl_max": max_kl,
        # "actor_lr": current_lr,
        # "critic_lr": critic_current_lr,
        "lr": current_lr,
    }
    return stats

    # # print optimizer step stats
    # logger.print("\nOptimizer step completed:")
    # logger.print(f"  Total loss: {total_loss.item():.4f}")
    # logger.print(f"  Gradient norm: {grad_norm.item():.4f}")
    # logger.print(f"  Number of batches: {len(batches)}")


def calculate_advantage(
    episodes: list[EpisodeData],
    discount_rate: float,
    rtg_first_moment: float,
    points_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    max_tile_weight: float = 1.0,
    corner_weight: float = 1.0,
    adjacency_weight: float = 1.0,
    chain_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    emptiness_weight: float = 1.0,
    topological_weight: float = 1.0,
    win_bonus: float = 1000.0,
    rtg_beta: float = 0.9,
    rtg_m2: float = 1.0,
    rtg_mu: float = 0.0,
    rtg_step: int = 1,
    upsample_ratio: float = 0.0,
) -> tuple[list[EpisodeData], list[StepData], float, float, float]:
    """
    Calculate per-step advantage for policy gradient.

    Reward = points_weight * points_earned
             + smoothness_weight * smoothness_delta
             + max_tile_weight * max_tile_created
             + corner_weight * corner_delta
             + adjacency_weight * adjacency_delta
             + chain_weight * chain_delta
             + monotonicity_weight * monotonicity_delta
             + topological_weight * topological_delta
             + win_bonus (one-time when 2048 tile is created)

    - points_earned is the raw game score from merges
    - smoothness_delta rewards moves that improve board structure
    - max_tile_created (exponent) rewards creating high-value tiles from merges
    - corner_delta rewards keeping the max tile in a corner
    - adjacency_delta rewards high-value tiles being adjacent to each other
    - chain_delta rewards building monotonically decreasing chains from max tile
    - monotonicity_delta rewards consistent increase/decrease patterns
    - topological_delta rewards proper tile organization (neighbors, gaps, density)
    - win_bonus is a one-time reward for creating the 2048 tile (exponent 11)
    """

    # 2048 = 2^11, so exponent is 11
    WIN_TILE_EXPONENT = 11

    # first we calculate the reward of each step
    for ep in episodes:
        for i, move in enumerate(ep["moves"]):
            # We have two types of rewards: absolute and shaped (PBRS)
            points_reward = move["points_earned"] * points_weight
            absolute_reward = points_reward  # this is what each step gets no matter what

            # Shaped reward, Potential-Based Reward Shaping (PBRS) - only use monotonicity for right now
            # allows us to provide relative rewawrds within each steps which get cancelled out once
            # the discounted reward is calculated.
            # mono_after = move["monotonicity_after"] if not last_step else 0.0
            shaped_reward = sum(
                [
                    monotonicity_weight * (discount_rate * move["monotonicity_after"] - move["monotonicity_before"]),
                    emptiness_weight * (discount_rate * move["emptiness_after"] - move["emptiness_before"]),
                ]
            )

            # # zero this out on the final step
            # if last_step:
            #     shaped_reward = 0.0
            move["reward"] = absolute_reward + shaped_reward

    # next we calculate the discount rate and per-timestep baseline
    for ep in episodes:
        moves = ep["moves"]
        G = 0.0
        for t in reversed(range(len(moves))):
            move = moves[t]
            G = move["reward"] + discount_rate * G
            move["future_reward_raw"] = G

    # next, calculate statistics of the current batch so we can have them after we normalize
    eps = 1e-8
    future_rewards_raw = [m["future_reward_raw"] for ep in episodes for m in ep["moves"]]
    N = len(future_rewards_raw)
    if N == 0:
        return episodes, rtg_first_moment, rtg_m2, rtg_mu

    # first, calculate the batch statistics for the return-to-go
    rtg_batch_mean = sum(future_rewards_raw) / N
    rtg_batch_var = 0.0 if N <= 1 else sum((fr - rtg_batch_mean) ** 2 for fr in future_rewards_raw) / N

    # rtg_updated_first_moment = rtg_beta * rtg_first_moment + (1 - rtg_beta) * rtg_batch_mean
    # rtg_updated_m2 = rtg_beta * rtg_m2 + (1 - rtg_beta) * (rtg_batch_var + rtg_batch_mean**2)  # this estimates E[G^2]
    # rtg_updated_mu = rtg_beta * rtg_mu + (1 - rtg_beta) * rtg_batch_mean  # this estimates E[G]

    # calculate bias-corrected moments
    bias_correction = max(1 - rtg_beta ** max(rtg_step, 1), eps)

    # first moment for advantage normalization - centers the advantage distribution
    rtg_mu_corrected = rtg_mu / bias_correction

    # now get the variance - normalizes the advantage spread
    rtg_m2_corrected = rtg_m2 / bias_correction
    rtg_var = max(rtg_m2_corrected - rtg_mu_corrected**2, eps)
    rtg_stddev = rtg_var**0.5

    # finally, we compute the normalized rtg
    for ep in episodes:
        for move in ep["moves"]:
            # triangle swap
            move["future_reward"] = (move["future_reward_raw"] - rtg_mu_corrected) / (rtg_stddev + eps)

    # calculate advantage using timestep-specific baselines in the normalized space
    for ep in episodes:
        for m in ep["moves"]:
            baseline = m["predicted_future_value"]
            # we want it to be disconnected at this point
            assert all(not isinstance(v, torch.Tensor) for v in [m["future_reward"], baseline]), (
                "one of the variables in advantage computation is a tensor"
            )

            # computed in the normalized space
            m["advantage"] = m["future_reward"] - baseline

    # create new samples via augmentation (mirroring and rotation)
    augmented_steps: list[StepData] = []
    steps_to_upsample = int(N * upsample_ratio)
    if steps_to_upsample > 0:
        flattened_steps = [m for ep in episodes for m in ep["moves"]]
        sampled_steps = random.sample(flattened_steps, min(steps_to_upsample, len(flattened_steps)))

        # direction indices: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

        def remap_direction_mirror(dir_idx: int, mirror_axis: str) -> int:
            """Remap direction index after mirroring."""
            d = dirs[dir_idx]
            if mirror_axis == "horizontal" and d in [Direction.LEFT, Direction.RIGHT]:
                new_d = Direction.LEFT if d == Direction.RIGHT else Direction.RIGHT
            elif mirror_axis == "vertical" and d in [Direction.UP, Direction.DOWN]:
                new_d = Direction.UP if d == Direction.DOWN else Direction.DOWN
            else:
                return dir_idx
            return dirs.index(new_d)

        def remap_direction_rotate(dir_idx: int, degrees: int) -> int:
            """Remap direction index after clockwise rotation."""
            # rotation maps: 90° CW: UP->RIGHT, RIGHT->DOWN, DOWN->LEFT, LEFT->UP
            rotate_90 = {
                Direction.UP: Direction.RIGHT,
                Direction.RIGHT: Direction.DOWN,
                Direction.DOWN: Direction.LEFT,
                Direction.LEFT: Direction.UP,
            }
            d = dirs[dir_idx]
            rotations = degrees // 90
            for _ in range(rotations):
                d = rotate_90[d]
            return dirs.index(d)

        def remap_action_mask(mask: list[bool], remap_fn, *args) -> list[bool]:
            """Remap action mask using the same direction remapping."""
            new_mask = [False] * 4
            for old_idx in range(4):
                new_idx = remap_fn(old_idx, *args)
                new_mask[new_idx] = mask[old_idx]
            return new_mask

        for step in sampled_steps:
            # chance of mirroring
            if random.random() < 0.5:
                augmented_step: StepData = deepcopy(step)
                old_state_before = augmented_step.pop("state_before")
                old_state_after = augmented_step.pop("result_state")
                _ = augmented_step.pop("game_state", None)

                mirror_axis = random.choice(["horizontal", "vertical"])
                mirrored_before = Game2048.mirror_grid(old_state_before, mirror_axis)
                mirrored_after = Game2048.mirror_grid(old_state_after, mirror_axis)

                augmented_step["state_before"] = mirrored_before
                augmented_step["result_state"] = mirrored_after
                augmented_step["game_state"] = Game2048(mirrored_before).to_model_format()
                augmented_step["points_possible"] = Game2048(mirrored_before).preview_move_rewards()

                old_dir_idx = augmented_step.pop("selected_direction")
                augmented_step["selected_direction"] = remap_direction_mirror(old_dir_idx, mirror_axis)

                old_mask = augmented_step.pop("action_mask", [False] * 4)
                augmented_step["action_mask"] = remap_action_mask(old_mask, remap_direction_mirror, mirror_axis)

                augmented_steps.append(augmented_step)

            # chance of rotation (independent of mirroring, uses original step)
            if random.random() < 0.5:
                augmented_step: StepData = deepcopy(step)
                old_state_before = augmented_step.pop("state_before")
                old_state_after = augmented_step.pop("result_state")
                _ = augmented_step.pop("game_state", None)

                degrees = random.choice([90, 180, 270])
                rotated_before = Game2048.rotate_grid(old_state_before, degrees)
                rotated_after = Game2048.rotate_grid(old_state_after, degrees)

                augmented_step["state_before"] = rotated_before
                augmented_step["result_state"] = rotated_after
                augmented_step["game_state"] = Game2048(rotated_before).to_model_format()
                augmented_step["points_possible"] = Game2048(rotated_before).preview_move_rewards()

                old_dir_idx = augmented_step.pop("selected_direction")
                augmented_step["selected_direction"] = remap_direction_rotate(old_dir_idx, degrees)

                old_mask = augmented_step.pop("action_mask", [False] * 4)
                augmented_step["action_mask"] = remap_action_mask(old_mask, remap_direction_rotate, degrees)

                augmented_steps.append(augmented_step)

    # # stabilize policy updates: center and scale advantages within the batch
    # advantages = [m["advantage"] for ep in episodes for m in ep["moves"]]
    # adv_mean = sum(advantages) / len(advantages)
    # adv_var = sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)
    # adv_std = adv_var**0.5
    # for ep in episodes:
    #     for m in ep["moves"]:
    #         m["advantage"] = (m["advantage"] - adv_mean) / (adv_std + eps)

    # finally, we compute updated rtg moments using the batch statistics
    # We wait until the end to do this so that the loss between the current value head and
    # the advantage is based on the same distribution

    # next, calculate new moments using current batch statistics
    # (we do this AFTER normalizing the batch to prevent noise from entering value head)
    rtg_updated_first_moment = rtg_beta * rtg_first_moment + (1 - rtg_beta) * rtg_batch_mean
    rtg_updated_m2 = rtg_beta * rtg_m2 + (1 - rtg_beta) * (rtg_batch_var + rtg_batch_mean**2)  # this estimates E[G^2]
    rtg_updated_mu = rtg_beta * rtg_mu + (1 - rtg_beta) * rtg_batch_mean  # this estimates E[G]
    rtg_updated_first_moment = rtg_updated_mu  # keep both mean trackers aligned

    # we return the new moments
    return episodes, augmented_steps, rtg_updated_first_moment, rtg_updated_m2, rtg_updated_mu


@dataclass
class RewardWeights:
    """Container for reward component weights."""

    points: float = 0.0
    smoothness: float = 0.0
    max_tile: float = 0.0
    corner: float = 0.0
    adjacency: float = 0.0
    chain: float = 0.0
    monotonicity: float = 0.0
    emptiness: float = 0.0
    topological: float = 0.0


def compute_batch_stats(
    rollout_episodes: list[EpisodeData],
    optimizer_stats: dict,
    highest_score: int,
    ema_avg_score: float,
    ema_pct_512: float,
    ema_pct_1024: float,
    ema_pct_2048: float,
    batch_pct_512: float,
    batch_pct_1024: float,
    batch_pct_2048: float,
    ema_decay: float,
    ema_explained_var: float,
) -> tuple[dict, float]:
    """
    Compute all batch statistics for logging.
    Returns (metrics_dict, updated_ema_explained_var).
    """
    all_moves = [m for ep in rollout_episodes for m in ep["moves"] if not ep.get("augmented", False)]
    rewards = [m["reward"] for m in all_moves]
    advantages = [m["advantage"] for m in all_moves]
    future_rewards_norm = [m["future_reward"] for m in all_moves]
    future_rewards = [m["future_reward_raw"] for m in all_moves]
    value_preds = [m["predicted_future_value"] for m in all_moves]

    n = len(rewards)
    reward_mean = sum(rewards) / n
    reward_var = sum((r - reward_mean) ** 2 for r in rewards) / n
    adv_mean = sum(advantages) / n
    adv_var = sum((a - adv_mean) ** 2 for a in advantages) / n

    future_mean = sum(future_rewards) / n
    future_var = sum((f - future_mean) ** 2 for f in future_rewards) / n

    future_norm_mean = sum(future_rewards_norm) / n
    future_norm_var = sum((f - future_norm_mean) ** 2 for f in future_rewards_norm) / n

    v_mean = sum(value_preds) / n
    v_var = sum((v - v_mean) ** 2 for v in value_preds) / n

    zero_reward_pct = sum(1 for r in rewards if r == 0) / n * 100

    scores = sorted([ep["total_points"] for ep in rollout_episodes])
    avg_score = sum(scores) / len(scores)
    median_score = (
        scores[len(scores) // 2]
        if len(scores) % 2 == 1
        else (scores[len(scores) // 2 - 1] + scores[len(scores) // 2]) / 2
    )

    adv_l2_norm = sum(a**2 for a in advantages) ** 0.5
    future_norm_std = future_norm_var**0.5
    adv_std = adv_var**0.5
    v_std = v_var**0.5
    future_raw_std = future_var**0.5
    variance_reduction = (future_norm_std - adv_std) / future_norm_std * 100 if future_norm_std > 0 else 0.0
    explained_var = 1.0 - adv_var / future_norm_var if future_norm_var > 0 else 0.0
    updated_ema_explained_var = (1 - ema_decay) * ema_explained_var + ema_decay * explained_var

    augmented_episode = [ep for ep in rollout_episodes if ep.get("augmented", False)]
    if len(augmented_episode) > 0:
        augmented_samples = len(augmented_episode[0]["moves"])
    else:
        augmented_samples = 0

    metrics = {
        "samples": n,
        "augmented_samples": augmented_samples,
        "actor_loss": optimizer_stats.get("actor_loss", 0),
        "critic_loss": optimizer_stats.get("critic_loss", 0),
        "total_loss": optimizer_stats.get("total_loss", 0),
        "policy_loss": optimizer_stats.get("policy_loss", 0),
        "entropy_loss": optimizer_stats.get("entropy_loss", 0),
        "value_loss": optimizer_stats.get("value_loss", 0),
        "actor_grad_norm": optimizer_stats.get("actor_grad_norm", 0),
        "critic_grad_norm": optimizer_stats.get("critic_grad_norm", 0),
        "grad_norm": optimizer_stats.get("grad_norm", 0),
        "entropy": optimizer_stats.get("entropy", 0),
        "peak_score": highest_score,
        "avg_score": avg_score,
        "ema_avg_score": ema_avg_score,
        "median_score": median_score,
        "pct_512": batch_pct_512,
        "ema_pct_512": ema_pct_512,
        "pct_1024": batch_pct_1024,
        "ema_pct_1024": ema_pct_1024,
        "pct_2048": batch_pct_2048,
        "ema_pct_2048": ema_pct_2048,
        "reward_var": reward_var,
        "reward_mean": reward_mean,
        "zero_reward_pct": zero_reward_pct,
        "advantage_var": adv_var,
        "advantage_l2": adv_l2_norm,
        "adv_min": min(advantages),
        "adv_max": max(advantages),
        "G_norm_mean": future_norm_mean,
        "G_norm_std": future_norm_std,
        "G_norm_min": min(future_rewards_norm),
        "G_norm_max": max(future_rewards_norm),
        "G_raw_std": future_raw_std,
        "V_std": v_std,
        "A_std": adv_std,
        "var_reduction": variance_reduction,
        "explained_var": explained_var,
        "ema_explained_var": updated_ema_explained_var,
        "kl_total": optimizer_stats.get("kl_total", 0),
        "kl_average": optimizer_stats.get("kl_average", 0),
        "kl_max": optimizer_stats.get("kl_max", 0),
        "actor_lr": optimizer_stats.get("actor_lr", 0),
        "critic_lr": optimizer_stats.get("critic_lr", 0),
    }
    return metrics, updated_ema_explained_var


def print_episode_breakdown(
    logger,
    episode: EpisodeData,
    weights: RewardWeights,
    gamma: float,
) -> None:
    """Print reward breakdown tables for an episode."""
    if not episode.get("moves"):
        return

    logger.print(f"\n  Best game this batch (score: {episode['total_points']}, steps: {episode['total_steps']}):")

    moves = episode["moves"]
    total_points = sum(m.get("points_earned", 0) for m in moves)
    total_smoothness = sum(m.get("smoothness_delta", 0) for m in moves)
    total_tile_bonus = sum(m.get("max_tile_created", 0) for m in moves)
    total_corner = sum(m.get("corner_delta", 0) for m in moves)
    total_adjacency = sum(m.get("adjacency_delta", 0) for m in moves)
    total_chain = sum(m.get("chain_delta", 0) for m in moves)
    total_topo = sum(m.get("topological_delta", 0) for m in moves)

    num_steps = len(moves)
    gamma_T = gamma**num_steps

    # PBRS potentials: monotonicity
    mono_initial = moves[0]["monotonicity_before"]
    mono_final = moves[-1]["monotonicity_after"]
    mono_pbrs_contribution = gamma_T * mono_final - mono_initial

    # PBRS potentials: emptiness
    empty_initial = moves[0].get("emptiness_before", 0.0)
    empty_final = moves[-1].get("emptiness_after", 0.0)
    empty_pbrs_contribution = gamma_T * empty_final - empty_initial

    reward_components = [
        ("points_earned", total_points, weights.points),
        ("smoothness", total_smoothness, weights.smoothness),
        ("tile_bonus", total_tile_bonus, weights.max_tile),
        ("corner", total_corner, weights.corner),
        ("adjacency", total_adjacency, weights.adjacency),
        ("chain", total_chain, weights.chain),
        ("topological", total_topo, weights.topological),
    ]

    logger.print("  Reward breakdown:")
    logger.print("    ┌─────────────────┬──────────┬────────┬──────────┐")
    logger.print("    │ Component       │      Raw │ Weight │ Weighted │")
    logger.print("    ├─────────────────┼──────────┼────────┼──────────┤")
    total_weighted = 0.0
    for name, raw, weight in reward_components:
        weighted = raw * weight
        total_weighted += weighted
        logger.print(f"    │ {name:<15} │ {raw:>8.1f} │ {weight:>6.2f} │ {weighted:>8.1f} │")
    logger.print("    ├─────────────────┼──────────┼────────┼──────────┤")
    logger.print(f"    │ {'TOTAL':<15} │          │        │ {total_weighted:>8.1f} │")
    logger.print("    └─────────────────┴──────────┴────────┴──────────┘")

    # print PBRS table if any shaped rewards are enabled
    if weights.monotonicity != 0.0 or weights.emptiness != 0.0:
        logger.print("")
        logger.print(f"  PBRS Reward Shaping (γ={gamma:.4f}, T={num_steps}, γ^T={gamma_T:.4f}):")
        logger.print("    ┌─────────────┬──────────┬──────────┬────────┬──────────┐")
        logger.print("    │ Potential   │    Φ(s₀) │   Φ(s_T) │ Weight │ γ^T·Φ_T-Φ₀│")
        logger.print("    ├─────────────┼──────────┼──────────┼────────┼──────────┤")

        total_pbrs = 0.0
        if weights.monotonicity != 0.0:
            weighted_mono = mono_pbrs_contribution * weights.monotonicity
            total_pbrs += weighted_mono
            logger.print(
                f"    │ monotonicity│ {mono_initial:>8.1f} │ {mono_final:>8.1f} │ {weights.monotonicity:>6.2f} │ {weighted_mono:>9.2f} │"
            )
        if weights.emptiness != 0.0:
            weighted_empty = empty_pbrs_contribution * weights.emptiness
            total_pbrs += weighted_empty
            logger.print(
                f"    │ emptiness   │ {empty_initial:>8.1f} │ {empty_final:>8.1f} │ {weights.emptiness:>6.2f} │ {weighted_empty:>9.2f} │"
            )

        logger.print("    ├─────────────┼──────────┼──────────┼────────┼──────────┤")
        logger.print(f"    │ TOTAL       │          │          │        │ {total_pbrs:>9.2f} │")
        logger.print("    └─────────────┴──────────┴──────────┴────────┴──────────┘")


def print_last_steps(logger, episode: EpisodeData, num_steps: int) -> None:
    """Print the last N steps of an episode with grid visualization."""
    if not episode.get("moves"):
        return

    direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    moves_to_show = episode["moves"][-num_steps:]
    start_idx = len(episode["moves"]) - len(moves_to_show)

    pts_summary = [str(m.get("points_earned", 0)) for m in moves_to_show]
    logger.print(f"\n  Last {len(moves_to_show)} steps (pts: {' → '.join(pts_summary)}):")

    for i, move in enumerate(moves_to_show):
        step_num = start_idx + i + 1
        action = direction_names[move["selected_direction"]]
        pts = move.get("points_earned", 0)
        logger.print(f"\n  Step {step_num}: {action} (+{pts} pts)")
        if "result_state" in move:
            logger.print(format_grid(move["result_state"], indent="  "))


def print_final_state(logger, episode: EpisodeData) -> None:
    """Print the final game state."""
    if "final_state" in episode:
        logger.print("\n  Final state:")
        logger.print(format_grid(episode["final_state"], indent="  "))


def export_episode_visualization(
    viz_dir: str,
    train_step: int,
    episode: EpisodeData,
    weights: RewardWeights,
    discount_rate: float,
) -> None:
    """Export episode data to JSON for visualization."""
    if not episode.get("moves"):
        return

    viz_path = Path(viz_dir)
    viz_path.mkdir(parents=True, exist_ok=True)

    def grid_to_values(grid):
        return [[2**cell if cell > 0 else 0 for cell in row] for row in grid]

    direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    viz_data = {
        "step": train_step,
        "score": episode["total_points"],
        "total_steps": episode["total_steps"],
        "moves": [],
    }

    for i, move in enumerate(episode["moves"]):
        state_before = move.get("state_before", [])
        state_after = move.get("result_state", [])
        move_data = {
            "step": i + 1,
            "state_before": grid_to_values(state_before) if state_before else [],
            "action": direction_names[move["selected_direction"]],
            "state_after": grid_to_values(state_after) if state_after else [],
            "points_earned": move.get("points_earned", 0),
            "rewards": {
                "points": move.get("points_earned", 0) * weights.points,
                "smoothness": move.get("smoothness_delta", 0) * weights.smoothness,
                "tile_bonus": move.get("max_tile_created", 0) * weights.max_tile,
                "corner": move.get("corner_delta", 0) * weights.corner,
                "adjacency": move.get("adjacency_delta", 0) * weights.adjacency,
                "chain": move.get("chain_delta", 0) * weights.chain,
                "monotonicity": (discount_rate * move.get("monotonicity_after", 0) - move.get("monotonicity_before", 0))
                * weights.monotonicity,
                "topological": move.get("topological_delta", 0) * weights.topological,
                "emptiness": (discount_rate * move.get("emptiness_after", 0) - move.get("emptiness_before", 0))
                * weights.emptiness,
            },
            "entropy": move.get("entropy", 0.0),
            "advantage": move.get("advantage", 0.0),
        }
        viz_data["moves"].append(move_data)

    export_file = viz_path / f"step_{train_step:06d}.json"
    with open(export_file, "w") as f:
        json.dump(viz_data, f, indent=2)


# class MultiOptimizer:
#     def __init__(self, *optimizers):
#         self.optimizers = optimizers

#     def step(self):
#         for opt in self.optimizers:
#             opt.step()

#     def zero_grad(self):
#         for opt in self.optimizers:
#             opt.zero_grad()

#     def state_dict(self):
#         return [opt.state_dict() for opt in self.optimizers]

#     def load_state_dict(self, state_dicts):
#         for opt, sd in zip(self.optimizers, state_dicts):
#             opt.load_state_dict(sd)


class MultiOptimizer:
    """Wraps multiple optimizers and their schedulers into a single interface."""

    def __init__(self, *optimizer_scheduler_pairs):
        """
        Args:
            *optimizer_scheduler_pairs: Tuples of (optimizer, scheduler) or just optimizer.
                If scheduler is None or not provided, no scheduler step is taken for that optimizer.
        """
        self.optimizers = []
        self.schedulers = []

        for item in optimizer_scheduler_pairs:
            if isinstance(item, tuple):
                opt, sched = item
            else:
                opt, sched = item, None
            self.optimizers.append(opt)
            self.schedulers.append(sched)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def scheduler_step(self, *args, **kwargs):
        """Step all schedulers. Pass metrics here if using ReduceLROnPlateau."""
        for sched in self.schedulers:
            if sched is not None:
                sched.step(*args, **kwargs)

    def get_lr(self):
        """Returns list of current learning rates for each optimizer."""
        return [opt.param_groups[0]["lr"] for opt in self.optimizers]

    def state_dict(self):
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "schedulers": [s.state_dict() if s else None for s in self.schedulers],
        }

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
        for sched, sd in zip(self.schedulers, state_dict["schedulers"]):
            if sched and sd:
                sched.load_state_dict(sd)


@app.command()
def train(
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of training steps"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to save/load model"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    gamma: float = typer.Option(0.99, "--gamma", help="Discount factor"),
    entropy_strength: float = typer.Option(0.1, "--entropy", help="Parmaeter for controlling entropy regularization."),
    critic_strength: float = typer.Option(
        1.0,
        "--critic",
        help="Strength of the critic (estimated reward-to-go) during loss computation",
    ),
    epsilon: float = typer.Option(1.0, "--epsilon", help="Initial exploration rate"),
    momentum: float = typer.Option(0.99, "--momentum", help="Momentum used for the EMA of average reward"),
    num_episodes: int = typer.Option(1, "--episodes", help="Number of games to play in a given batch"),
    batch_size: int = typer.Option(1, "--batch-size", help="Minibatch size when updating policy"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers for game execution"),
    max_steps: int = typer.Option(None, "--max-steps", help="Maximum number of steps a game can reach."),
    hidden_size: int = typer.Option(64, "-h", "--hidden", help="hidden capacity of the model"),
    num_layers: int = typer.Option(2, "--num-layers", "-l", help="Number of residual hidden layers"),
    model_type: str = typer.Option("mlp", "--model-type", "-t", help="Model architecture: 'mlp' or 'urm'"),
    # URM-specific options
    num_heads: int = typer.Option(4, "--num-heads", help="Number of attention heads (URM only)"),
    num_loops: int = typer.Option(4, "--num-loops", help="Number of recurrent reasoning loops (URM only)"),
    num_truncated_loops: int = typer.Option(
        1,
        "--truncated-loops",
        help="Loops without gradients for memory savings (URM only)",
    ),
    print_frequency: int = typer.Option(10, "--print-freq", "-p", help="Printing frequency"),
    show_last_steps: int = typer.Option(
        0,
        "--show-last-steps",
        help="Show the last N steps of the best game (0 = disabled)",
    ),
    points_weight: float = typer.Option(0.0, "--points", help="Weight for raw game points (0 = disabled)"),
    smoothness_weight: float = typer.Option(
        0.0, "--smoothness", help="Weight for smoothness reward shaping (0 = disabled)"
    ),
    max_tile_weight: float = typer.Option(0.0, "--tile-bonus", help="Weight for max tile created bonus (0 = disabled)"),
    corner_weight: float = typer.Option(0.0, "--corner", help="Weight for corner bonus (max tile in corner)"),
    adjacency_weight: float = typer.Option(
        0.0,
        "--adjacency",
        help="Weight for adjacency bonus (high-value tiles next to each other)",
    ),
    chain_weight: float = typer.Option(
        0.0,
        "--chain",
        help="Weight for monotonic chain bonus (descending sequence from max tile)",
    ),
    monotonicity_weight: float = typer.Option(
        0.0,
        "--mono",
        help="Weight for monotonicity score (consistent increase/decrease patterns)",
    ),
    warmup_steps: int = typer.Option(
        200,
        "--warmup-steps",
        help="Number of warmup steps for the learning rate scheduler",
    ),
    emptiness_weight: float = typer.Option(
        0.0,
        "--emptiness",
        help="Weight for emptiness score (prevents model from filling board with points early on)",
    ),
    topological_weight: float = typer.Option(
        0.0,
        "--topo",
        help="Weight for topological score (proper neighbors, gap penalty, corner density)",
    ),
    win_bonus: float = typer.Option(
        0.0,
        "--win-bonus",
        help="One-time bonus for creating the 2048 tile",
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Use CUDA:0 for training"),
    viz_dir: Optional[str] = typer.Option(
        None,
        "--viz-dir",
        help="Directory to export visualization data (disabled if not set)",
    ),
    rtg_beta: Optional[float] = typer.Option(
        0.9,
        "--rtg-beta",
        help="This is the beta1 used when estimating the moments of the return-to-go",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        "--log-dir",
        help="Directory for JSONL logs (default: current directory)",
    ),
    use_wandb: bool = typer.Option(
        False,
        "--wandb",
        help="Enable Weights & Biases logging",
    ),
    wandb_project: Optional[str] = typer.Option(
        "2048-rl",
        "--wandb-project",
        help="W&B project name",
    ),
    wandb_run_name: Optional[str] = typer.Option(
        None,
        "--wandb-run",
        help="W&B run name (auto-generated if not set)",
    ),
    eval_freq: Optional[int] = typer.Option(
        None,
        "--eval-freq",
        help="Evaluate model every N training steps (disabled if not set)",
    ),
    eval_games: int = typer.Option(
        100,
        "--eval-games",
        help="Number of games to play during evaluation",
    ),
    critic_lr: float = typer.Option(
        0.001,
        "--critic-lr",
        help="Learning rate for the critic (estimated reward-to-go) during loss computation",
    ),
    decouple_critic: bool = typer.Option(
        False,
        "--decouple-critic",
        help="Decouple the critic from the compute graph",
    ),
    upsample_ratio: float = typer.Option(
        0.0,
        "--upsample-ratio",
        help="Ratio of samples to upsample for data augmentation",
    ),
    export_demo: bool = typer.Option(
        False,
        "--export-demo",
        help="Export best game and model to docs/ folder for GitHub Pages demo",
    ),
    checkpoint_dir: Optional[str] = typer.Option(
        "checkpoints",
        "--checkpoint-dir",
        help="Directory to save model checkpoints",
    ),
):
    """Watch the AI play 2048."""
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    # build config dict for logging
    train_config = {
        "steps": steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "beta": entropy_strength,
        "critic_strength": critic_strength,
        "batch_size": num_episodes,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "model_type": model_type,
        "num_heads": num_heads,
        "num_loops": num_loops,
        "num_truncated_loops": num_truncated_loops,
        "points_weight": points_weight,
        "smoothness_weight": smoothness_weight,
        "max_tile_weight": max_tile_weight,
        "corner_weight": corner_weight,
        "adjacency_weight": adjacency_weight,
        "chain_weight": chain_weight,
        "monotonicity_weight": monotonicity_weight,
        "topological_weight": topological_weight,
        "win_bonus": win_bonus,
        "rtg_beta": rtg_beta,
    }

    # initialize metric logger
    logger = MetricLogger(
        log_dir=log_dir,
        experiment_name=f"train_{model_type}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=train_config,
    )

    if gpu and not torch.cuda.is_available():
        logger.print("Warning: --gpu flag set but CUDA is not available, using CPU")
    else:
        logger.print(f"Using device: {device}")

    model = None
    if model_path:
        logger.print(f"Loading model from: {model_path}")
        logger.print("this path is not available")
        logger.close()
        import sys

        sys.exit(0)
    else:
        # Create model based on model_type
        model_type_lower = model_type.lower()
        if model_type_lower == "mlp":
            logger.print(f"Creating GameMLP model (hidden={hidden_size}, layers={num_layers})")
            # actor_model = GameMLP(MLPConfig(hidden_dim=hidden_size, num_layers=num_layers))
            # critic_model = GameMLP(MLPConfig(hidden_dim=hidden_size, num_layers=num_layers, decouple_critic=False))
            model = GameMLP(MLPConfig(hidden_dim=hidden_size, num_layers=num_layers, decouple_critic=decouple_critic))
        elif model_type_lower == "urm":
            logger.print(
                f"Creating GameURM model (hidden={hidden_size}, layers={num_layers}, "
                f"heads={num_heads}, loops={num_loops}, truncated={num_truncated_loops})"
            )
            logger.print("this model type is not available")
            logger.close()
            import sys

            sys.exit(1)
            # model = GameURM(
            #     GameURMConfig(
            #         hidden_dim=hidden_size,
            #         num_layers=num_layers,
            #         num_heads=num_heads,
            #         num_loops=num_loops,
            #         num_truncated_loops=num_truncated_loops,
            #     )
            # )
        else:
            logger.print(f"Unknown model type: {model_type}. Use 'mlp' or 'urm'.")
            logger.close()
            import sys

            sys.exit(1)

    # initialize variables for tracking ema on RTG
    rtg_mu = 0.0
    rtg_m2 = 1.0
    rtg_moment = 0.0

    # actor_model = actor_model.to(device)
    # critic_model = critic_model.to(device)
    model = model.to(device)

    # After creating the model, before training:
    with torch.no_grad():
        # actor_model.action_head.weight.zero_()
        # actor_model.action_head.bias.zero_()
        # critic_model.value_head.weight.zero_()
        # critic_model.value_head.bias.zero_()
        model.action_head.weight.zero_()
        model.action_head.bias.zero_()
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()

    # actor_optimizer = AdamW(
    #     actor_model.parameters(),
    #     lr=learning_rate,
    #     betas=(0.9, 0.999), weight_decay=0.01
    # )
    # critic_optimizer = AdamW(
    #     critic_model.parameters(), lr=critic_lr, betas=(0.9, 0.999), weight_decay=0.01
    # )
    #         return [
    #         {"params": other_params_2d, "lr": other_lr},
    #         {"params": other_params_1d, "lr": other_lr},
    #         {"params": value_params_2d, "lr": value_lr},
    #         {"params": value_params_1d, "lr": value_lr},
    #     ]
    # value_params_1d = []
    # value_params_2d = []
    # other_params_1d = []
    # other_params_2d = []
    other_params_2d, other_params_1d, value_params_2d, value_params_1d = model.get_param_groups(
        critic_lr, learning_rate
    )

    adamw = AdamW(
        [other_params_1d, value_params_1d],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    muon = Muon([other_params_2d, value_params_2d], adjust_lr_fn="match_rms_adamw", weight_decay=0.01)
    adamw_scheduler = get_scheduler(
        "cosine",
        adamw,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps,
    )

    muon_scheduler = get_scheduler(
        "cosine",
        muon,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps,
    )

    optimizer = MultiOptimizer((muon, muon_scheduler), (adamw, adamw_scheduler))
    # actor_lr_scheduler = get_scheduler(
    #     "cosine",
    #     actor_optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=steps,
    # )
    # critic_lr_scheduler = get_scheduler(
    #     "cosine",
    #     critic_optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=steps,
    # )

    # Add this before the training loop starts
    with torch.no_grad():
        test_game = Game2048()
        test_game.reset()
        test_input = test_game.to_model_format().unsqueeze(0).to(device)
        # act_logits, val_logits = actor_model(test_input)
        # val_logit = critic_model(test_input)
        # print(f"Initial action logits: {act_logits}")
        # print(f"Initial value logit: {val_logit}")
        action_logits, value_logit = model(test_input)
        print(f"Initial action logits: {action_logits}")
        print(f"Initial value logit: {value_logit}")

    # train the model
    highest_score = 0

    # Best game/model tracking for demo export
    best_game_episode: Optional[EpisodeData] = None
    best_eval_avg_score = 0.0
    model_config = MLPConfig(hidden_dim=hidden_size, num_layers=num_layers, decouple_critic=decouple_critic)

    # EMA trackers for running statistics (decay=0.001 ~ last 1000 steps)
    ema_decay = 0.001
    ema_avg_score = 0.0
    ema_pct_512 = 0.0
    ema_pct_1024 = 0.0
    ema_pct_2048 = 0.0
    ema_explained_var = 0.0

    # helper to get max tile from episode
    def get_max_tile(ep):
        if "final_state" in ep:
            return max(2**cell if cell > 0 else 0 for row in ep["final_state"] for cell in row)
        return 0

    for train_step in tqdm(range(steps), desc=f"Running RL training"):
        # actor_model.eval()
        # critic_model.eval()
        model.eval()

        # here we compute a single pass
        rollout_episodes = []

        # game loop - parallel if workers > 1
        if workers > 1:
            raise NotImplementedError("multithreading is not implemented")
            # # Prepare args for worker processes (state dict, config, max_steps)
            # state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            # config_dict = {
            #     "hidden_dim": hidden_size,
            #     "num_layers": num_layers,
            #     "dropout": 0.0,  # dropout not used at inference
            # }
            # worker_args = [
            #     (state_dict, config_dict, max_steps) for _ in range(batch_size)
            # ]

            # with ProcessPoolExecutor(max_workers=workers) as executor:
            #     rollout_episodes = list(executor.map(_worker_play_game, worker_args))
        else:
            for _ in range(num_episodes):
                rollout_episodes.append(
                    # play_game_for_episode(actor_model, critic_model, max_steps=max_steps, device=device)
                    play_game_for_episode(model, max_steps=max_steps, device=device)
                )

        # rollout_episodes.extend(augment_data_samples(rollout_episodes, upsample_ratio))

        # get the updated moments after each batch
        rollout_episodes, augmented_steps, rtg_moment, rtg_m2, rtg_mu = calculate_advantage(
            rollout_episodes,
            gamma,
            rtg_moment,
            points_weight,
            smoothness_weight,
            max_tile_weight,
            corner_weight,
            adjacency_weight,
            chain_weight,
            monotonicity_weight,
            emptiness_weight,
            topological_weight,
            win_bonus,
            rtg_beta=rtg_beta,
            rtg_m2=rtg_m2,
            rtg_mu=rtg_mu,
            rtg_step=train_step + 1,  # 1-indexed for the correct normalization
            upsample_ratio=upsample_ratio,
        )

        # add augmented steps to episodes
        rollout_episodes.append(
            {
                "moves": augmented_steps,
                "total_points": sum(step["points_earned"] for step in augmented_steps),
                "total_steps": len(augmented_steps),
                "augmented": True,
                "final_state": augmented_steps[-1]["result_state"],
            }
        )

        # Now we update the model's policy using this rollout
        stats = model_optimize_step(
            # actor_model,
            # critic_model,
            model,
            rollout_episodes,
            # actor_optimizer,
            # critic_optimizer,
            optimizer,
            # actor_lr_scheduler,
            # critic_lr_scheduler,
            None,
            entropy_strength,
            critic_strength,
            device,
            batch_size=batch_size,
        )

        # Get highest score from batch and track if we hit a new record
        non_augmented_episodes = [ep for ep in rollout_episodes if not ep.get("augmented", False)]
        batch_max_score = max(episode["total_points"] for episode in non_augmented_episodes)
        new_high_score = batch_max_score > highest_score
        highest_score = max(batch_max_score, highest_score)

        # Track best game episode for demo export
        if new_high_score:
            best_game_episode = max(non_augmented_episodes, key=lambda ep: ep["total_points"])
            best_game_episode = deepcopy(best_game_episode)  # Make a copy to preserve state

        # Calculate average number of steps in batch
        avg_steps = sum([len(episode["moves"]) for episode in non_augmented_episodes]) / len(non_augmented_episodes)

        # compute batch tile stats and update EMAs every step
        max_tiles = [get_max_tile(ep) for ep in non_augmented_episodes]
        num_eps = len(non_augmented_episodes)
        batch_avg_score = sum(ep["total_points"] for ep in non_augmented_episodes) / num_eps
        batch_pct_512 = sum(1 for t in max_tiles if t >= 512) / num_eps * 100
        batch_pct_1024 = sum(1 for t in max_tiles if t >= 1024) / num_eps * 100
        batch_pct_2048 = sum(1 for t in max_tiles if t >= 2048) / num_eps * 100

        # update EMAs
        ema_avg_score = (1 - ema_decay) * ema_avg_score + ema_decay * batch_avg_score
        ema_pct_512 = (1 - ema_decay) * ema_pct_512 + ema_decay * batch_pct_512
        ema_pct_1024 = (1 - ema_decay) * ema_pct_1024 + ema_decay * batch_pct_1024
        ema_pct_2048 = (1 - ema_decay) * ema_pct_2048 + ema_decay * batch_pct_2048

        # always compute and log metrics to file/wandb
        batch_metrics, ema_explained_var = compute_batch_stats(
            rollout_episodes=rollout_episodes,
            optimizer_stats=stats,
            highest_score=highest_score,
            ema_avg_score=ema_avg_score,
            ema_pct_512=ema_pct_512,
            ema_pct_1024=ema_pct_1024,
            ema_pct_2048=ema_pct_2048,
            batch_pct_512=batch_pct_512,
            batch_pct_1024=batch_pct_1024,
            batch_pct_2048=batch_pct_2048,
            ema_decay=ema_decay,
            ema_explained_var=ema_explained_var,
        )

        # print_frequency controls stdout; file/wandb always logged
        should_print = train_step % print_frequency == 0
        logger.log(batch_metrics, step=train_step, verbose=should_print)

        # detailed stdout output only at print_frequency
        if should_print:
            best_episode = max(non_augmented_episodes, key=lambda ep: ep["total_points"])
            weights = RewardWeights(
                points=points_weight,
                smoothness=smoothness_weight,
                max_tile=max_tile_weight,
                corner=corner_weight,
                adjacency=adjacency_weight,
                chain=chain_weight,
                monotonicity=monotonicity_weight,
                emptiness=emptiness_weight,
                topological=topological_weight,
            )

            print_episode_breakdown(logger, best_episode, weights, gamma)

            if show_last_steps > 0:
                print_last_steps(logger, best_episode, show_last_steps)

            print_final_state(logger, best_episode)

            if viz_dir:
                export_episode_visualization(viz_dir, train_step, best_episode, weights, gamma)

        # always save viz data when we hit a new high score (even if not printing)
        if new_high_score and viz_dir and not should_print:
            record_episode = max(non_augmented_episodes, key=lambda ep: ep["total_points"])
            record_weights = RewardWeights(
                points=points_weight,
                smoothness=smoothness_weight,
                max_tile=max_tile_weight,
                corner=corner_weight,
                adjacency=adjacency_weight,
                chain=chain_weight,
                monotonicity=monotonicity_weight,
                emptiness=emptiness_weight,
                topological=topological_weight,
            )
            export_episode_visualization(viz_dir, train_step, record_episode, record_weights, gamma)

        if train_step > 0 and eval_freq and train_step % eval_freq == 0:
            # actor_model.eval()
            # critic_model.eval()
            model.eval()
            typer.echo(f"[Step {train_step}] Evaluating model on {eval_games} games", color="green")
            eval_episodes = []
            for i in tqdm(range(eval_games), desc="evaluating model", total=eval_games):
                eval_episodes.append(
                    play_game_for_episode(
                        # actor_model,
                        # critic_model,
                        model,
                        max_steps=max_steps,
                        # device=next(p.device for p in actor_model.parameters()),
                        device=next(p.device for p in model.parameters()),
                        seed=i,
                    )
                )

            # Compute evaluation statistics
            eval_scores = [ep["total_points"] for ep in eval_episodes]
            eval_max_score = max(eval_scores)
            eval_avg_score = sum(eval_scores) / len(eval_scores)
            eval_median_score = sorted(eval_scores)[len(eval_scores) // 2]
            eval_max_tiles = [get_max_tile(ep) for ep in eval_episodes]
            eval_pct_512 = sum(1 for t in eval_max_tiles if t >= 512) / len(eval_episodes) * 100
            eval_pct_1024 = sum(1 for t in eval_max_tiles if t >= 1024) / len(eval_episodes) * 100
            eval_pct_2048 = sum(1 for t in eval_max_tiles if t >= 2048) / len(eval_episodes) * 100

            eval_metrics = {
                "eval/max_score": eval_max_score,
                "eval/avg_score": eval_avg_score,
                "eval/median_score": eval_median_score,
                "eval/pct_512": eval_pct_512,
                "eval/pct_1024": eval_pct_1024,
                "eval/pct_2048": eval_pct_2048,
            }
            logger.log(eval_metrics, step=train_step)

            typer.echo(
                f"Eval Results - Max: {eval_max_score:.0f}, Avg: {eval_avg_score:.1f}, Median: {eval_median_score:.0f}",
                color="green",
            )
            typer.echo(
                f"Tiles Reached - 512: {eval_pct_512:.1f}%, 1024: {eval_pct_1024:.1f}%, 2048: {eval_pct_2048:.1f}%",
                color="green",
            )

            # Save best model checkpoint when eval average improves
            if eval_avg_score > best_eval_avg_score:
                best_eval_avg_score = eval_avg_score
                checkpoint_path = Path(checkpoint_dir) / "best_model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model_config.model_dump(),
                        "eval_avg_score": eval_avg_score,
                        "train_step": train_step,
                    },
                    checkpoint_path,
                )
                typer.echo(
                    f"New best model saved (avg score: {eval_avg_score:.1f}) to {checkpoint_path}", color="green"
                )

            # actor_model.train()
            # critic_model.train()
            model.train()

    # End of training: export demo if requested
    if export_demo:
        typer.echo("\nExporting demo assets to docs/ folder...")
        docs_data_dir = Path("docs/data")
        docs_data_dir.mkdir(parents=True, exist_ok=True)

        # Export best game
        if best_game_episode:
            export_best_game_for_demo(best_game_episode, str(docs_data_dir / "best_game.json"))
        else:
            typer.echo("Warning: No best game to export (no games were played)")

        # Export model to ONNX
        try:
            export_model_to_onnx(model, str(docs_data_dir / "model.onnx"), model_config)
            typer.echo(f"Model exported to {docs_data_dir / 'model.onnx'}")
        except Exception as e:
            typer.echo(f"Error exporting model to ONNX: {e}")
            typer.echo("Make sure 'onnx' package is installed: pip install onnx")

    logger.close()


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    games: int = typer.Option(100, "--games", "-g", help="Number of games to evaluate"),
):
    """Evaluate a trained 2048 AI agent."""
    typer.echo(f"Evaluating model from: {model_path}")
    typer.echo(f"Running {games} evaluation games...")

    # TODO: Implement evaluation
    typer.echo("Evaluation not yet implemented")


@app.command("export-demo")
def export_demo_cmd(
    model_path: Path = typer.Option(
        "checkpoints/best_model.pt",
        "--model",
        "-m",
        help="Path to model checkpoint (.pt file)",
    ),
    game_path: Optional[Path] = typer.Option(
        None,
        "--game",
        "-g",
        help="Path to game JSON file (optional, will play new games if not provided)",
    ),
    output_dir: Path = typer.Option(
        "docs/data",
        "--output",
        "-o",
        help="Output directory for demo assets",
    ),
    num_games: int = typer.Option(
        10,
        "--num-games",
        "-n",
        help="Number of games to play to find best game (if --game not provided)",
    ),
):
    """
    Export trained model and best game to docs/ for GitHub Pages demo.

    Usage:
        python train.py export-demo --model checkpoints/best_model.pt
        python train.py export-demo --model checkpoints/best_model.pt --game viz_data/step_001000.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model from checkpoint
    if not model_path.exists():
        typer.echo(f"Error: Model checkpoint not found at {model_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = MLPConfig(**checkpoint["config"])
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume it's just a state dict, use defaults
        config = MLPConfig()
        state_dict = checkpoint

    model = GameMLP(config)
    model.load_state_dict(state_dict)
    model.eval()
    typer.echo(f"Model loaded (hidden_dim={config.hidden_dim}, num_layers={config.num_layers})")

    # Get best game
    if game_path and game_path.exists():
        # Load existing game from viz_data format
        typer.echo(f"Loading game from {game_path}...")
        with open(game_path) as f:
            game_data = json.load(f)

        # Convert viz_data format to demo format if needed
        if "moves" in game_data and game_data["moves"]:
            demo_data = {
                "score": game_data.get("score", 0),
                "total_steps": game_data.get("total_steps", len(game_data["moves"])),
                "moves": game_data["moves"],
            }
            output_game_path = output_dir / "best_game.json"
            with open(output_game_path, "w") as f:
                json.dump(demo_data, f, indent=2)
            typer.echo(f"Game exported to {output_game_path}")
    else:
        # Play games to find best one
        typer.echo(f"Playing {num_games} games to find best game...")
        best_episode = None
        best_score = 0

        for i in tqdm(range(num_games), desc="Playing games"):
            episode = play_game_for_episode(model, max_steps=None, device=None)
            if episode["total_points"] > best_score:
                best_score = episode["total_points"]
                best_episode = episode

        if best_episode:
            export_best_game_for_demo(best_episode, str(output_dir / "best_game.json"))
        else:
            typer.echo("Warning: No games were played successfully")

    # Export model to ONNX
    try:
        onnx_path = output_dir / "model.onnx"
        export_model_to_onnx(model, str(onnx_path), config)
        typer.echo(f"Model exported to {onnx_path}")
    except ImportError as e:
        typer.echo(f"Error: Missing dependency - {e}")
        typer.echo("Install with: pip install onnx onnxscript")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error exporting model: {e}")
        raise typer.Exit(1)

    # Also save PyTorch checkpoint to docs for version control
    import shutil

    pt_dest = output_dir / "best_model.pt"
    shutil.copy(model_path, pt_dest)
    typer.echo(f"PyTorch checkpoint copied to {pt_dest}")

    typer.echo(f"\nDemo assets exported to {output_dir}/")
    typer.echo("To test locally: cd docs && python -m http.server 8000")


def get_keypress():
    """Read a single keypress from the terminal."""
    import sys
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow keys (escape sequences)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


@app.command()
def human():
    """Play 2048 yourself! Controls: WASD or Arrow keys, Q to quit."""
    import os

    # Clear screen
    os.system("clear" if os.name == "posix" else "cls")

    typer.echo("🎮 2048 - Human Player Mode")
    typer.echo("Controls: W/↑=Up, S/↓=Down, A/←=Left, D/→=Right, Q=Quit")
    typer.echo("-" * 40)

    game = Game2048()
    game.reset()

    move_count = 0

    # Key mappings
    key_to_direction = {
        "w": Direction.UP,
        "a": Direction.LEFT,
        "s": Direction.DOWN,
        "d": Direction.RIGHT,
        "\x1b[A": Direction.UP,  # Up arrow
        "\x1b[B": Direction.DOWN,  # Down arrow
        "\x1b[C": Direction.RIGHT,  # Right arrow
        "\x1b[D": Direction.LEFT,  # Left arrow
    }

    display_board(game)

    while game.has_next_step():
        typer.echo("\nYour move: ", nl=False)

        key = get_keypress()

        # Handle quit
        if key.lower() == "q":
            typer.echo("\n\n👋 Thanks for playing!")
            break

        # Map key to direction
        direction = key_to_direction.get(key.lower() if len(key) == 1 else key)

        if direction is None:
            typer.echo("Invalid key. Use WASD or arrow keys.")
            continue

        if not game.direction_has_step(direction):
            typer.echo(f"Can't move {direction.value}! Try another direction.")
            continue

        # Make the move
        _, points_earned, done, _ = game.step(direction)
        move_count += 1

        # Clear screen and redraw
        os.system("clear" if os.name == "posix" else "cls")
        typer.echo("🎮 2048 - Human Player Mode")
        typer.echo("Controls: W/↑=Up, S/↓=Down, A/←=Left, D/→=Right, Q=Quit")
        typer.echo("-" * 40)
        typer.echo(f"Move {move_count}: {direction.value.upper()} (+{points_earned} points)")

        display_board(game)

        if done:
            break

    # Game over
    typer.echo("\n" + "=" * 40)
    typer.echo("🎮 GAME OVER!")
    typer.echo(f"Final Score: {game.get_score()}")
    typer.echo(f"Total Moves: {move_count}")

    # Find highest tile
    max_tile = max(2**cell if cell > 0 else 0 for row in game.grid for cell in row)
    typer.echo(f"Highest Tile: {max_tile}")

    if max_tile >= 2048:
        typer.echo("🎉 Congratulations! You reached 2048!")
    typer.echo("=" * 40)


def display_board(game: Game2048) -> None:
    """Display the game board in a nice format."""
    typer.echo("")
    typer.echo(format_grid(game.grid, indent=""))
    typer.echo(f"Score: {game.get_score()}")


@app.command()
def play(
    model_path: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained model (optional)"),
    delay: float = typer.Option(0.5, "--delay", "-d", help="Delay between moves in seconds"),
):
    """Watch the AI play 2048."""
    model = None
    if model_path:
        typer.echo(f"Loading model from: {model_path}")
        typer.echo("this path is not availaable")
        import sys

        sys.exit(0)
    else:
        typer.echo("Playing with random agent (no model specified)")
        model = GameMLP(MLPConfig(hidden_dim=64))

    # initialize game
    game = Game2048()
    game.reset()

    typer.echo("\nStarting game with random agent...")
    display_board(game)

    move_count = 0
    total_points = 0

    # this will be a running average
    total_reward = 0.0
    momentum = 0.90
    step = 1

    # stores the data generated from each episode per rollout
    episode_data = []

    # game loop
    while game.has_next_step():
        step_data = {}

        # find valid directions
        valid_directions = [d for d in Direction if game.direction_has_step(d)]

        if not valid_directions:
            break

        # we look at the possible gains to determine the overall reward
        possible_gains = game.preview_move_rewards()
        best_direction, highest_points = max(possible_gains.items(), key=lambda item: item[1])

        # now agent makes a decision
        model_input = game.to_model_format()
        logits, _ = model(model_input.unsqueeze(0))

        # we interret the directions as being UP/DOWN/LEFT/RIGHT:
        dirs = model.directions
        valid_dirs = game.current_valid_directions()
        invalid_mask = [_dir not in valid_dirs for _dir in dirs]

        # extract the action sequence & mask out invalid states
        action_probs = logits[-1]
        action_probs[invalid_mask] = -torch.inf

        # now we let the model decide which direction it should move into
        adjusted_action_dist = torch.softmax(action_probs, dim=0)  # this is the one we want to save
        selected_action = torch.multinomial(adjusted_action_dist, num_samples=1).item()
        action: Direction = model.directions[selected_action]

        # take step
        new_state, points_earned, done, info = game.step(action)
        move_count += 1
        total_points += points_earned

        # calculate the per-step reward
        if highest_points != 0:
            step_reward = 1.0 if action == best_direction else points_earned / highest_points
        else:
            step_reward = 0

        # update the running avg and collect the bias-corrected format
        total_reward = total_reward * momentum + step_reward * (1 - momentum)
        total_reward_corrected = total_reward / (1 - momentum**step)

        # and then this determines which action was actually selected
        step_data["adjusted_logits"] = adjusted_action_dist
        step_data["selected_direction"] = selected_action
        step_data["game_state"] = model_input
        step_data["step_reward"] = step_reward
        step_data["current_total_reward"] = total_reward_corrected

        # save the data generated at the current step
        episode_data.append(step_data)

        # display move
        typer.echo(f"\nMove {move_count}: {action.value.upper()} (points earned: {points_earned})")
        typer.echo(f"Best available: {best_direction.value.upper()} ({highest_points} points)")
        typer.echo(
            f"Step reward: {step_reward:.3f} | Total reward (EMA): {total_reward:.3f} | Bias Corrected: {total_reward_corrected:.3f}"
        )
        display_board(game)
        step += 1

        if done:
            typer.echo("\n🎮 Game Over!")
            break

        # delay for visualization
        time.sleep(delay)

    # final stats
    typer.echo(f"\n{'=' * 25}")
    typer.echo(f"Final Score: {game.get_score()}")
    typer.echo(f"Total Moves: {move_count}")
    typer.echo(f"Total Reward: {total_points}")
    typer.echo(f"{'=' * 25}\n")


if __name__ == "__main__":
    app()
