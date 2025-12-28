"""
CLI training interface for 2048 AI agent.
Run with: python train.py [command]
"""

import typer
from typing import Optional
from pathlib import Path
import random
import time
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm


from game import Game2048, Direction, GameMLP, MLPConfig, GameURM, GameURMConfig

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
    model: torch.nn.Module, max_steps: int | None = None, device: torch.device = None
) -> dict:
    """
    Given a model, play an episode of a game.
    """
    # ----------------------------------------
    # initialize game
    game = Game2048()
    game.reset()

    # stores the data generated from each episode per rollout
    game_data = []
    move_count = 0
    total_points = 0

    # this will be a running average
    step = 0

    while game.has_next_step() and (
        not max_steps or (max_steps > 0 and step < max_steps)
    ):
        step_data = {}

        # find valid directions
        valid_directions = [d for d in Direction if game.direction_has_step(d)]
        if not valid_directions:
            break

        # we look at the possible gains to determine the overall reward
        possible_gains = game.preview_move_rewards()
        best_direction, highest_points = max(
            possible_gains.items(), key=lambda item: item[1]
        )

        # capture grid state before the move (for visualization)
        state_before = [row[:] for row in game.grid]

        # now agent makes a decision
        model_input = game.to_model_format()
        if device is not None:
            model_input = model_input.to(device)
        action_logits, predicted_future_value = model(model_input.unsqueeze(0))

        # we interret the directions as being UP/DOWN/LEFT/RIGHT:
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        valid_dirs = game.current_valid_directions()
        invalid_mask = [_dir not in valid_dirs for _dir in dirs]

        # extract the action sequence & mask out invalid states
        action_probs = action_logits[-1]
        action_probs[invalid_mask] = -torch.inf

        # now we let the model decide which direction it should move into
        adjusted_action_dist = torch.softmax(
            action_probs, dim=0
        )  # this is the one we want to save
        try:
            selected_action = torch.multinomial(
                adjusted_action_dist, num_samples=1
            ).item()
        except RuntimeError:
            from IPython import embed

            embed()
        action: Direction = model.directions[selected_action]

        # compute entropy of the action distribution (model's uncertainty)
        valid_probs = adjusted_action_dist[adjusted_action_dist > 0]
        step_entropy = -(valid_probs * valid_probs.log()).sum().item()

        # take step
        new_state, points_earned, done, info = game.step(action)
        move_count += 1
        total_points += points_earned

        # # calculate the per-step reward
        # if highest_points != 0:
        #     step_reward = (
        #         1.0 if points_earned == highest_points else points_earned / highest_points
        #     )
        # else:
        #     # since no good move was possible, we just give this one
        #     # as a freebie to the model so overall reward doesn't penalize
        #     # exploration as a means of rebalancing the board
        #     step_reward = 1.0

        # update the running avg and collect the bias-corrected format
        # total_reward = total_reward * momentum + step_reward * (1 - momentum)
        # total_reward_corrected = total_reward / (1 - momentum ** (step + 1))

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
        step_data["monotonicity_after"] = info.get("monotonicity_after", 0.0)
        step_data["monotonicity_before"] = info.get("monotonicity_before", 0.0)
        step_data["topological_delta"] = info.get("topological_delta", 0.0)
        step_data["topological_anchor"] = info.get("topological_anchor")
        step_data["entropy"] = step_entropy

        # save the data generated at the current step
        game_data.append(step_data)

        # we dont want to save the terminal state as part of the rollout, since all possible actions would be masked
        # and unless our initial training run results in a perfect model (highly unlikely) we dont need to worry about
        # the edge case of it winning
        if done:
            break  # the `step` variable is `1-indexed` so the final step isn't incremented

        step += 1

    # add the rollout data
    episode_data = {
        "moves": game_data,
        "total_points": total_points,
        "total_steps": step,
        "final_state": new_state,  # Store the final grid for logging
    }
    return episode_data


def _worker_play_game(args: tuple) -> dict:
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


def model_optimize_step(
    model: torch.nn.Module,
    episodes: list[dict],
    optimizer: torch.optim.Optimizer,
    beta: float = 0.1,
    critic_strength: float = 1.0,
    device: torch.device = None,
):
    """
    Performs an optimization step as part of the RL loop.
    """
    # so what we'll do is create a separate input batch for each rollout so it can be multiplied
    # by the advantage
    # so our input data should become [game state] -> [direction label]
    # batches = []
    minibatch = []
    for ep in episodes:
        for move in ep["moves"]:
            input_state = move["game_state"]
            direction_idx = torch.tensor([move["selected_direction"]])
            action_mask = move["action_mask"]
            minibatch.append(
                (
                    input_state,
                    direction_idx,
                    action_mask,
                    move["advantage"],
                    move["future_reward"],
                )
            )

    # now we create a single batched input tensor
    input_batch = torch.stack([item[0] for item in minibatch])
    target_batch = torch.stack([item[1] for item in minibatch])
    action_mask = torch.tensor([item[2] for item in minibatch])
    advantage = torch.tensor([item[3] for item in minibatch], dtype=torch.float32)
    vtg_batch = torch.tensor([item[4] for item in minibatch])

    if device is not None:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        action_mask = action_mask.to(device)
        advantage = advantage.to(device)

    # now that the data is prepared, we compute the loss and take an optimizer step
    model.train()

    action_logits, value_logits = model(
        inputs=input_batch,
    )

    targets = target_batch.to(device=action_logits.device)

    # convert (B, T) into (B x T)
    targets = targets.view(-1)
    action_logits[action_mask] = -torch.inf  # mask to kill the training signal

    # compute the advantaged cross-entropy loss
    policy_loss_per_sample = F.cross_entropy(
        input=action_logits, target=targets, reduction="none"
    )
    policy_loss_per_sample *= advantage
    policy_loss = policy_loss_per_sample.mean()

    # entropy regularization
    masked_probs = torch.softmax(action_logits, dim=1)
    entropy_per_sample = -(masked_probs * (masked_probs + 1e-8).log()).sum(dim=1)
    entropy_loss = -beta * entropy_per_sample.mean()

    # value loss for the learned baseline
    value_logits = value_logits.view(-1)
    value_loss_per_sample = F.smooth_l1_loss(value_logits, vtg_batch, reduction="none")
    value_loss = critic_strength * value_loss_per_sample.mean()

    # combine all losses
    loss = policy_loss + entropy_loss + value_loss

    # now we backprop
    loss.backward()

    # clip gradnorm and take an optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # collect stats for logging
    stats = {
        "loss": loss.detach().cpu().item(),
        "policy_loss": policy_loss.detach().cpu().item(),
        "entropy_loss": entropy_loss.detach().cpu().item(),
        "value_loss": value_loss.detach().cpu().item(),
        "grad_norm": grad_norm,
        "entropy": entropy_per_sample.mean().detach().cpu().item(),
    }
    return stats

    # # print optimizer step stats
    # typer.echo("\nOptimizer step completed:")
    # typer.echo(f"  Total loss: {total_loss.item():.4f}")
    # typer.echo(f"  Gradient norm: {grad_norm.item():.4f}")
    # typer.echo(f"  Number of batches: {len(batches)}")


def calculate_advantage(
    episodes: list[dict],
    discount_rate: float,
    rtg_first_moment: float,
    points_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    max_tile_weight: float = 1.0,
    corner_weight: float = 1.0,
    adjacency_weight: float = 1.0,
    chain_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    topological_weight: float = 1.0,
    win_bonus: float = 1000.0,
    rtg_beta=0.9,
    rtg_m2=1.0,
    rtg_mu=0.0,
) -> tuple[list[dict], float, float, float]:
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
            # check if we're done or not
            last_step = i + 1 == len(ep["moves"])

            # # combine merge points with board quality improvement
            points_reward = move["points_earned"] * points_weight
            # smoothness_reward = move.get("smoothness_delta", 0.0) * smoothness_weight
            # tile_bonus = move.get("max_tile_created", 0) * max_tile_weight
            # corner_bonus = move.get("corner_delta", 0.0) * corner_weight
            # adjacency_bonus = move.get("adjacency_delta", 0.0) * adjacency_weight
            # chain_bonus = move.get("chain_delta", 0.0) * chain_weight
            # topological_bonus = move.get("topological_delta", 0.0) * topological_weight

            # One-time bonus for creating the 2048 tile (detect first crossing)
            # max_before = move.get("max_exponent_before", 0)
            # max_after = move.get("max_exponent_after", max_before)
            # created_2048 = (
            #     max_before < WIN_TILE_EXPONENT and max_after >= WIN_TILE_EXPONENT
            # )
            # # win_reward = win_bonus if created_2048 else 0.0

            # Legacy method of pointing, do not use
            # move["reward"] = points_reward
            #  All of the below points should strictly be **shaped reward**, and not absolute
            # + smoothness_reward
            # + tile_bonus
            # + corner_bonus
            # + adjacency_bonus
            # + chain_bonus
            # + monotonicity_bonus
            # + topological_bonus
            # + win_reward

            # We have two types of rewards: absolute and shaped (PBRS)
            points_reward = move["points_earned"] * points_weight
            absolute_reward = (
                points_reward  # this is what each step gets no matter what
            )

            # Shaped reward, Potential-Based Reward Shaping (PBRS) - only use monotonicity for right now
            # allows us to provide relative rewawrds within each steps which get cancelled out once
            # the discounted reward is calculated.
            mono_after = move["monotonicity_after"] if not last_step else 0.0
            shaped_reward = monotonicity_weight * (
                discount_rate * mono_after - move["monotonicity_before"]
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
            move["future_reward"] = G

    # calculate the batch statistics for the return-to-go
    N = sum(len(ep["moves"]) for ep in episodes)
    rtg_batch_mean = sum(m["future_reward"] for ep in episodes for m in ep["moves"]) / N
    rtg_batch_var = ((N - 1) ** -1) * sum(
        (
            (m["future_reward"] - rtg_batch_mean) ** 2
            for ep in episodes
            for m in ep["moves"]
        )
    )

    # next, update the rtg moments
    rtg_first_moment = rtg_beta * rtg_first_moment + (1 - rtg_beta) * rtg_batch_mean
    eps = 1e-8
    rtg_m2 = rtg_beta * rtg_m2 + (1 - rtg_beta) * (
        rtg_batch_var + rtg_batch_mean**2
    )  # this estimates E[G^2]
    rtg_mu = rtg_beta * rtg_mu + (1 - rtg_beta) * rtg_batch_mean  # this estimates E[G]

    # now get the variance
    rtg_var = max(rtg_m2 - rtg_mu**2, eps)
    rtg_stddev = rtg_var**0.5

    # finally, we compute the normalized rtg
    for ep in episodes:
        for move in ep["moves"]:
            # triangle swap
            move["future_reward_raw"] = move["future_reward"]
            move["future_reward"] = (move["future_reward"] - rtg_first_moment) / (
                rtg_stddev + eps
            )

    # calculate advantage using timestep-specific baselines (no normalization)
    for ep in episodes:
        for m in ep["moves"]:
            baseline = m["predicted_future_value"]
            # we want it to be disconnected at this point
            assert all(
                not isinstance(v, torch.Tensor) for v in [m["future_reward"], baseline]
            ), "one of the variables in advantage computation is a tensor"

            # computed in the normalized space
            m["advantage"] = m["future_reward"] - baseline

    # Here I tried to use the harmonic mean to get the advantage, but this still doesnt work
    # calculate the reward for each model
    # for episode in episodes:
    #     # harmonic mean
    #     episode["reward"] = len(episode["moves"]) / sum(
    #         1 / m["reward"] * max(1.0, m["highest_points"]) if m["reward"] > 0 else 0
    #         for m in episode["moves"]
    #     )

    # # calculate advantage
    # # avg_reward = len(rollout_episodes) / sum(
    # #     1 / e["reward"] if e["reward"] > 0 else 0 for e in rollout_episodes
    # # )
    # N = len(episodes)
    # avg_reward = sum(e["reward"] for e in episodes) / N
    # for ep in episodes:
    #     ep["advantage_unnormalized"] = ep["reward"] - avg_reward

    # # next, we calculate the normalized advantage
    # advantage_avg = sum(ep["advantage_unnormalized"] for ep in episodes) / N
    # advantage_stddev = (
    #     (1 / (N - 1))
    #     * sum((e["advantage_unnormalized"] - advantage_avg) ** 2 for e in episodes)
    # ) ** 0.5

    # # add an epsilon for numerical stability
    # eps = 1e-8
    # for ep in episodes:
    #     ep["advantage"] = (ep["advantage_unnormalized"] - advantage_avg) / (
    #         advantage_stddev + eps
    #     )

    # so then we rescale everything to fit this

    # for episode in rollout_episodes:
    #     episode["advantage"] = episode["reward"] - avg_reward

    # Print advantage statistics
    # Advantage statistics will be printed in the training loop

    # now that we have our rollout episodes, we need to evaluate the quality of each rollout
    # personally, my thinking is we should use the median reward as the baseline where advantage=0
    # and then when we calculate cross-entropy, we push rollouts towards the positive reward
    # rollout_episodes = sorted(rollout_episodes, key=lambda x: x["average_reward"])

    # # now we need to calculate the advantage of each  sample
    # min_r, max_r = (
    #     rollout_episodes[0]["average_reward"],
    #     rollout_episodes[-1]["average_reward"],
    # )
    # # print(f"{min_r=:.3f}, {max_r=:.3f}")

    # # calculate median
    # if len(rollout_episodes) % 2 != 0:
    #     middle_idx = len(rollout_episodes) // 2
    #     med_r = rollout_episodes[middle_idx]["average_reward"]
    # else:
    #     middle_idx = len(rollout_episodes) // 2
    #     med_r = (
    #         rollout_episodes[middle_idx]["average_reward"]
    #         + rollout_episodes[middle_idx - 1]["average_reward"]
    #     ) / 2

    # calculate advantage
    # for episode in rollout_episodes:
    #     episode["advantage"] = episode["average_reward"] - med_r

    # we return the new moments
    return episodes, rtg_first_moment, rtg_m2, rtg_mu

    # print summary statistics of the rollout episodes
    # total_points_list = [ep["total_points"] for ep in rollout_episodes]
    # total_steps_list = [ep["total_steps"] for ep in rollout_episodes]
    # average_rewards = [ep["average_reward"] for ep in rollout_episodes]
    # print(f"\n{'=' * 50}")
    # print(f"Rollout Summary Statistics:")
    # print(f"{'=' * 50}")
    # print(
    #     f"Total Points - Min: {min(total_points_list)}, Max: {max(total_points_list)}, Median: {total_points_list[len(total_points_list) // 2]}"
    # )
    # print(
    #     f"Total Steps  - Min: {min(total_steps_list)}, Max: {max(total_steps_list)}, Median: {total_steps_list[len(total_steps_list) // 2]}"
    # )
    # print(
    #     f"Avg Reward   - Min: {min(average_rewards):.3f}, Max: {max(average_rewards):.3f}, Median: {average_rewards[len(average_rewards) // 2]:.3f}"
    # )
    # # Calculate advantage for each episode
    # for episode in rollout_episodes:
    #     print(f"Episode advantage: {episode['advantage']:.3f}")
    # print(f"{'=' * 50}\n")


@app.command()
def train(
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of training steps"),
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to save/load model"
    ),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    gamma: float = typer.Option(0.99, "--gamma", help="Discount factor"),
    beta: float = typer.Option(
        0.1, "--beta", help="Parmaeter for controlling entropy regularization."
    ),
    critic_strength: float = typer.Option(
        1.0,
        "--critic",
        help="Strength of the critic (estimated reward-to-go) during loss computation",
    ),
    epsilon: float = typer.Option(1.0, "--epsilon", help="Initial exploration rate"),
    momentum: float = typer.Option(
        0.99, "--momentum", help="Momentum used for the EMA of average reward"
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", help="Number of games to play in a given batch"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of parallel workers for game execution"
    ),
    max_steps: int = typer.Option(
        None, "--max-steps", help="Maximum number of steps a game can reach."
    ),
    hidden_size: int = typer.Option(
        64, "-h", "--hidden", help="hidden capacity of the model"
    ),
    num_layers: int = typer.Option(
        2, "--num-layers", "-l", help="Number of residual hidden layers"
    ),
    model_type: str = typer.Option(
        "mlp", "--model-type", "-t", help="Model architecture: 'mlp' or 'urm'"
    ),
    # URM-specific options
    num_heads: int = typer.Option(
        4, "--num-heads", help="Number of attention heads (URM only)"
    ),
    num_loops: int = typer.Option(
        4, "--num-loops", help="Number of recurrent reasoning loops (URM only)"
    ),
    num_truncated_loops: int = typer.Option(
        1,
        "--truncated-loops",
        help="Loops without gradients for memory savings (URM only)",
    ),
    print_frequency: int = typer.Option(
        10, "--print-freq", "-p", help="Printing frequency"
    ),
    show_last_steps: int = typer.Option(
        0,
        "--show-last-steps",
        help="Show the last N steps of the best game (0 = disabled)",
    ),
    points_weight: float = typer.Option(
        0.0, "--points", help="Weight for raw game points (0 = disabled)"
    ),
    smoothness_weight: float = typer.Option(
        0.0, "--smoothness", help="Weight for smoothness reward shaping (0 = disabled)"
    ),
    max_tile_weight: float = typer.Option(
        0.0, "--tile-bonus", help="Weight for max tile created bonus (0 = disabled)"
    ),
    corner_weight: float = typer.Option(
        0.0, "--corner", help="Weight for corner bonus (max tile in corner)"
    ),
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
):
    """Watch the AI play 2048."""
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    if gpu and not torch.cuda.is_available():
        typer.echo("Warning: --gpu flag set but CUDA is not available, using CPU")
    else:
        typer.echo(f"Using device: {device}")

    model = None
    if model_path:
        typer.echo(f"Loading model from: {model_path}")
        typer.echo("this path is not available")
        import sys

        sys.exit(0)
    else:
        # Create model based on model_type
        model_type_lower = model_type.lower()
        if model_type_lower == "mlp":
            typer.echo(
                f"Creating GameMLP model (hidden={hidden_size}, layers={num_layers})"
            )
            model = GameMLP(MLPConfig(hidden_dim=hidden_size, num_layers=num_layers))
        elif model_type_lower == "urm":
            typer.echo(
                f"Creating GameURM model (hidden={hidden_size}, layers={num_layers}, "
                f"heads={num_heads}, loops={num_loops}, truncated={num_truncated_loops})"
            )
            model = GameURM(
                GameURMConfig(
                    hidden_dim=hidden_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    num_loops=num_loops,
                    num_truncated_loops=num_truncated_loops,
                )
            )
        else:
            typer.echo(f"Unknown model type: {model_type}. Use 'mlp' or 'urm'.")
            import sys

            sys.exit(1)

    # initialize variables for tracking ema on RTG
    rtg_mu = 0.0
    rtg_m2 = 1.0
    rtg_moment = 0.0

    model = model.to(device)

    # After creating the model, before training:
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.zero_()

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.01
    )

    # Add this before the training loop starts
    with torch.no_grad():
        test_game = Game2048()
        test_game.reset()
        test_input = test_game.to_model_format().unsqueeze(0).to(device)
        act_logits, val_logits = model(test_input)
        probs = torch.softmax(act_logits, dim=1)
        print(f"Initial logits: {act_logits}")
        print(f"Initial probs: {probs}")
        print(f"Initial value: {val_logits}")
        print(f"Initial entropy: {-(probs * probs.log()).sum().item():.4f}")

    # train the model
    highest_score = 0

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
            return max(
                2**cell if cell > 0 else 0 for row in ep["final_state"] for cell in row
            )
        return 0

    for train_step in tqdm(range(steps), desc=f"Running RL training"):
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
            for _ in range(batch_size):
                rollout_episodes.append(
                    play_game_for_episode(model, max_steps=max_steps, device=device)
                )

        # get the updated moments after each batch
        rollout_episodes, rtg_moment, rtg_m2, rtg_mu = calculate_advantage(
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
            topological_weight,
            win_bonus,
            rtg_beta=rtg_beta,
            rtg_m2=rtg_m2,
            rtg_mu=rtg_mu,
        )

        # Now we update the model's policy using this rollout
        stats = model_optimize_step(
            model, rollout_episodes, optimizer, beta, critic_strength, device
        )

        # Get highest score from batch
        highest_score = max(
            max([episode["total_points"] for episode in rollout_episodes]),
            highest_score,
        )

        # Calculate average number of steps in batch
        avg_steps = sum([len(episode["moves"]) for episode in rollout_episodes]) / len(
            rollout_episodes
        )

        # compute batch tile stats and update EMAs every step
        max_tiles = [get_max_tile(ep) for ep in rollout_episodes]
        num_eps = len(rollout_episodes)
        batch_avg_score = sum(ep["total_points"] for ep in rollout_episodes) / num_eps
        batch_pct_512 = sum(1 for t in max_tiles if t >= 512) / num_eps * 100
        batch_pct_1024 = sum(1 for t in max_tiles if t >= 1024) / num_eps * 100
        batch_pct_2048 = sum(1 for t in max_tiles if t >= 2048) / num_eps * 100

        # update EMAs
        ema_avg_score = (1 - ema_decay) * ema_avg_score + ema_decay * batch_avg_score
        ema_pct_512 = (1 - ema_decay) * ema_pct_512 + ema_decay * batch_pct_512
        ema_pct_1024 = (1 - ema_decay) * ema_pct_1024 + ema_decay * batch_pct_1024
        ema_pct_2048 = (1 - ema_decay) * ema_pct_2048 + ema_decay * batch_pct_2048

        # Print stats instead of updating progress bar
        if train_step % print_frequency == 0:
            # collect per-step metrics
            all_moves = [m for ep in rollout_episodes for m in ep["moves"]]
            rewards = [m["reward"] for m in all_moves]
            advantages = [m["advantage"] for m in all_moves]
            future_rewards_norm = [m["future_reward"] for m in all_moves]
            future_rewards = [m["future_reward_raw"] for m in all_moves]
            value_preds = [m["predicted_future_value"] for m in all_moves]

            # compute variances
            n = len(rewards)
            reward_mean = sum(rewards) / n
            reward_var = sum((r - reward_mean) ** 2 for r in rewards) / n
            adv_mean = sum(advantages) / n
            adv_var = sum((a - adv_mean) ** 2 for a in advantages) / n

            future_mean = sum(future_rewards) / n
            future_var = sum((f - future_mean) ** 2 for f in future_rewards) / n

            future_norm_mean = sum(future_rewards_norm) / n
            future_norm_var = (
                sum((f - future_norm_mean) ** 2 for f in future_rewards_norm) / n
            )

            # value prediction stats
            v_mean = sum(value_preds) / n
            v_var = sum((v - v_mean) ** 2 for v in value_preds) / n

            # count zero rewards (sparse signal indicator)
            zero_reward_pct = sum(1 for r in rewards if r == 0) / n * 100

            scores = sorted([ep["total_points"] for ep in rollout_episodes])
            avg_score = sum(scores) / len(scores)
            median_score = (
                scores[len(scores) // 2]
                if len(scores) % 2 == 1
                else (scores[len(scores) // 2 - 1] + scores[len(scores) // 2]) / 2
            )

            typer.echo(f"--- Step {train_step} ---")
            typer.echo(f"  samples:        {n}")
            typer.echo(f"  loss:           {stats['loss']:.4f}")
            typer.echo(f"  policy_loss:    {stats.get('policy_loss', 0):.4f}")
            typer.echo(f"  entropy_loss:   {stats.get('entropy_loss', 0):.4f}")
            typer.echo(f"  value_loss:     {stats.get('value_loss', 0):.4f}")
            typer.echo(f"  grad_norm:      {stats.get('grad_norm', 0):.4f}")
            typer.echo(f"  entropy:        {stats.get('entropy', 0):.4f}")
            typer.echo(f"  peak_score:     {highest_score}")
            typer.echo(f"  avg_score:      {avg_score:.1f} (ema: {ema_avg_score:.1f})")
            typer.echo(f"  median_score:   {median_score:.1f}")
            typer.echo(
                f"  >=512:          {batch_pct_512:.1f}% (ema: {ema_pct_512:.1f}%)"
            )
            typer.echo(
                f"  >=1024:         {batch_pct_1024:.1f}% (ema: {ema_pct_1024:.1f}%)"
            )
            typer.echo(
                f"  >=2048:         {batch_pct_2048:.1f}% (ema: {ema_pct_2048:.1f}%)"
            )
            typer.echo(f"  reward_var:     {reward_var:.2f}")
            typer.echo(f"  future_raw_var: {future_var:.2f}")
            typer.echo(f"  future_var:     {future_norm_var:.2f}")
            adv_l2_norm = sum(a**2 for a in advantages) ** 0.5
            typer.echo(f"  advantage_var:  {adv_var:.2f}")
            typer.echo(f"  advantage_l2:   {adv_l2_norm:.2f}")
            typer.echo(f"  reward_mean:    {reward_mean:.2f}")
            typer.echo(f"  zero_reward%:   {zero_reward_pct:.1f}%")
            typer.echo(
                f"  adv_range:      [{min(advantages):.2f}, {max(advantages):.2f}]"
            )
            future_norm_std = future_norm_var**0.5
            adv_std = adv_var**0.5
            typer.echo(f"  G_norm.mean:    {future_norm_mean:.4f}")
            typer.echo(f"  G_norm.std:     {future_norm_std:.4f}")
            typer.echo(
                f"  G_norm.range:   [{min(future_rewards_norm):.2f}, {max(future_rewards_norm):.2f}]"
            )

            # critic usefulness: does V reduce variance of A compared to G?
            # if std(A) < std(G), critic is learning a useful baseline
            v_std = v_var**0.5
            future_raw_std = future_var**0.5
            variance_reduction = (
                (future_norm_std - adv_std) / future_norm_std * 100
                if future_norm_std > 0
                else 0.0
            )
            # explained variance: EV = 1 - Var(G - V) / Var(G) = 1 - Var(A) / Var(G)
            explained_var = (
                1.0 - adv_var / future_norm_var if future_norm_var > 0 else 0.0
            )
            # update EMA for explained variance (only on print steps)
            ema_explained_var = (
                1 - ema_decay
            ) * ema_explained_var + ema_decay * explained_var
            typer.echo(f"  std(G_raw):     {future_raw_std:.4f}")
            typer.echo(f"  std(G_norm):    {future_norm_std:.4f}")
            typer.echo(f"  std(V):         {v_std:.4f}")
            typer.echo(f"  std(A):         {adv_std:.4f}")
            typer.echo(f"  var_reduction:  {variance_reduction:.1f}%")
            typer.echo(
                f"  explained_var:  {explained_var:.4f} (ema: {ema_explained_var:.4f})"
            )

            # Find and display the best game from this batch
            best_episode = max(rollout_episodes, key=lambda ep: ep["total_points"])
            typer.echo(
                f"\n  Best game this batch (score: {best_episode['total_points']}, steps: {best_episode['total_steps']}):"
            )

            # Calculate reward breakdown for best episode
            if best_episode["moves"]:
                moves = best_episode["moves"]
                total_points = sum(m.get("points_earned", 0) for m in moves)
                total_smoothness = sum(m.get("smoothness_delta", 0) for m in moves)
                total_tile_bonus = sum(m.get("max_tile_created", 0) for m in moves)
                total_corner = sum(m.get("corner_delta", 0) for m in moves)
                total_adjacency = sum(m.get("adjacency_delta", 0) for m in moves)
                total_chain = sum(m.get("chain_delta", 0) for m in moves)
                total_topo = sum(m.get("topological_delta", 0) for m in moves)

                # PBRS monotonicity: telescopes to γ^T * Φ(s_T) - Φ(s_0)
                mono_initial = moves[0]["monotonicity_before"]
                mono_final = moves[-1]["monotonicity_after"]
                mono_net_change = mono_final - mono_initial
                num_steps = len(moves)
                # the discount factor raised to the episode length
                gamma_T = gamma**num_steps
                # actual PBRS contribution to discounted return
                mono_pbrs_contribution = gamma_T * mono_final - mono_initial

                # Build reward breakdown table (absolute rewards only)
                reward_components = [
                    ("points_earned", total_points, points_weight),
                    ("smoothness", total_smoothness, smoothness_weight),
                    ("tile_bonus", total_tile_bonus, max_tile_weight),
                    ("corner", total_corner, corner_weight),
                    ("adjacency", total_adjacency, adjacency_weight),
                    ("chain", total_chain, chain_weight),
                    ("topological", total_topo, topological_weight),
                ]

                typer.echo("  Reward breakdown:")
                typer.echo("    ┌─────────────────┬──────────┬────────┬──────────┐")
                typer.echo("    │ Component       │      Raw │ Weight │ Weighted │")
                typer.echo("    ├─────────────────┼──────────┼────────┼──────────┤")
                total_weighted = 0.0
                for name, raw, weight in reward_components:
                    weighted = raw * weight
                    total_weighted += weighted
                    typer.echo(
                        f"    │ {name:<15} │ {raw:>8.1f} │ {weight:>6.2f} │ {weighted:>8.1f} │"
                    )
                typer.echo("    ├─────────────────┼──────────┼────────┼──────────┤")
                typer.echo(
                    f"    │ {'TOTAL':<15} │          │        │ {total_weighted:>8.1f} │"
                )
                typer.echo("    └─────────────────┴──────────┴────────┴──────────┘")

                # PBRS monotonicity breakdown (shaped reward, telescopes over episode)
                if monotonicity_weight != 0.0:
                    typer.echo("")
                    typer.echo(
                        f"  PBRS Monotonicity (weight={monotonicity_weight:.2f}, γ={gamma:.4f}):"
                    )
                    typer.echo("    ┌───────────────────────┬──────────┐")
                    typer.echo(f"    │ Φ(s₀) initial         │ {mono_initial:>8.1f} │")
                    typer.echo(f"    │ Φ(s_T) final          │ {mono_final:>8.1f} │")
                    typer.echo(
                        f"    │ Net Δ (final - init)  │ {mono_net_change:>8.1f} │"
                    )
                    typer.echo("    ├───────────────────────┼──────────┤")
                    typer.echo(f"    │ T (episode length)    │ {num_steps:>8d} │")
                    typer.echo(f"    │ γ^T                   │ {gamma_T:>8.4f} │")
                    typer.echo("    ├───────────────────────┼──────────┤")
                    typer.echo(
                        f"    │ γ^T·Φ(s_T) - Φ(s₀)    │ {mono_pbrs_contribution:>8.2f} │"
                    )
                    weighted_pbrs = mono_pbrs_contribution * monotonicity_weight
                    typer.echo(f"    │ Weighted contribution │ {weighted_pbrs:>8.2f} │")
                    typer.echo("    └───────────────────────┴──────────┘")

            # Show last N steps if requested
            if show_last_steps > 0 and best_episode["moves"]:
                direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                moves_to_show = best_episode["moves"][-show_last_steps:]
                start_idx = len(best_episode["moves"]) - len(moves_to_show)

                # Summary line with points for each step
                pts_summary = [str(m.get("points_earned", 0)) for m in moves_to_show]
                typer.echo(
                    f"\n  Last {len(moves_to_show)} steps (pts: {' → '.join(pts_summary)}):"
                )

                for i, move in enumerate(moves_to_show):
                    step_num = start_idx + i + 1
                    action = direction_names[move["selected_direction"]]
                    pts = move.get("points_earned", 0)
                    typer.echo(f"\n  Step {step_num}: {action} (+{pts} pts)")
                    if "result_state" in move:
                        typer.echo(format_grid(move["result_state"], indent="  "))

            if "final_state" in best_episode:
                typer.echo("\n  Final state:")
                typer.echo(format_grid(best_episode["final_state"], indent="  "))

            # Export visualization data if viz_dir is set
            if viz_dir and best_episode["moves"]:
                viz_path = Path(viz_dir)
                viz_path.mkdir(parents=True, exist_ok=True)

                # Convert grid from exponents to actual tile values
                def grid_to_values(grid):
                    return [
                        [2**cell if cell > 0 else 0 for cell in row] for row in grid
                    ]

                direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                viz_data = {
                    "step": train_step,
                    "score": best_episode["total_points"],
                    "total_steps": best_episode["total_steps"],
                    "moves": [],
                }

                for i, move in enumerate(best_episode["moves"]):
                    state_before = move.get("state_before", [])
                    state_after = move.get("result_state", [])
                    move_data = {
                        "step": i + 1,
                        "state_before": grid_to_values(state_before)
                        if state_before
                        else [],
                        "action": direction_names[move["selected_direction"]],
                        "state_after": grid_to_values(state_after)
                        if state_after
                        else [],
                        "points_earned": move.get("points_earned", 0),
                        "rewards": {
                            "points": move.get("points_earned", 0) * points_weight,
                            "smoothness": move.get("smoothness_delta", 0)
                            * smoothness_weight,
                            "tile_bonus": move.get("max_tile_created", 0)
                            * max_tile_weight,
                            "corner": move.get("corner_delta", 0) * corner_weight,
                            "adjacency": move.get("adjacency_delta", 0)
                            * adjacency_weight,
                            "chain": move.get("chain_delta", 0) * chain_weight,
                            "monotonicity": move.get("monotonicity_delta", 0)
                            * monotonicity_weight,
                            "topological": move.get("topological_delta", 0)
                            * topological_weight,
                        },
                        "entropy": move.get("entropy", 0.0),
                        "advantage": move.get("advantage", 0.0),
                    }
                    viz_data["moves"].append(move_data)

                export_file = viz_path / f"step_{train_step:06d}.json"
                with open(export_file, "w") as f:
                    json.dump(viz_data, f, indent=2)

            typer.echo("")  # Blank line for readability


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
        typer.echo(
            f"Move {move_count}: {direction.value.upper()} (+{points_earned} points)"
        )

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
    model_path: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to trained model (optional)"
    ),
    delay: float = typer.Option(
        0.5, "--delay", "-d", help="Delay between moves in seconds"
    ),
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
        best_direction, highest_points = max(
            possible_gains.items(), key=lambda item: item[1]
        )

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
        adjusted_action_dist = torch.softmax(
            action_probs, dim=0
        )  # this is the one we want to save
        selected_action = torch.multinomial(adjusted_action_dist, num_samples=1).item()
        action: Direction = model.directions[selected_action]

        # take step
        new_state, points_earned, done, info = game.step(action)
        move_count += 1
        total_points += points_earned

        # calculate the per-step reward
        if highest_points != 0:
            step_reward = (
                1.0 if action == best_direction else points_earned / highest_points
            )
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
        typer.echo(
            f"\nMove {move_count}: {action.value.upper()} (points earned: {points_earned})"
        )
        typer.echo(
            f"Best available: {best_direction.value.upper()} ({highest_points} points)"
        )
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
