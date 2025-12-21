"""
CLI training interface for 2048 AI agent.
Run with: python train.py [command]
"""

import typer
from typing import Optional
from pathlib import Path
import random
import time
import torch
from torch.optim import AdamW
from tqdm import tqdm

from game import Game2048, Direction, GameMLP, MLPConfig

app = typer.Typer(help="Train and evaluate 2048 AI agents")


@torch.no_grad
def play_game_for_episode(
    model: GameMLP, max_steps: int | None = None, device: torch.device = None
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

        # now agent makes a decision
        model_input = game.to_model_format()
        if device is not None:
            model_input = model_input.to(device)
        logits, _ = model(model_input.unsqueeze(0))

        # we interret the directions as being UP/DOWN/LEFT/RIGHT:
        dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
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
        step_data["selected_direction"] = selected_action
        step_data["game_state"] = model_input
        # step_data["reward"] = step_reward
        step_data["max_points_possible"] = highest_points
        step_data["points_earned"] = points_earned
        step_data["points_possible"] = possible_gains
        step_data["action_mask"] = invalid_mask

        # we dont want to save the terminal state as part of the rollout, since all possible actions would be masked
        # and unless our initial training run results in a perfect model (highly unlikely) we dont need to worry about
        # the edge case of it winning
        if done:
            break  # the `step` variable is `1-indexed` so the final step isn't incremented

        # save the data generated at the current step
        game_data.append(step_data)
        step += 1

    # add the rollout data
    episode_data = {
        "moves": game_data,
        "total_points": points_earned,
        "total_steps": step,
    }
    return episode_data


def model_optimize_step(
    model: GameMLP,
    episodes: list[dict],
    optimizer: torch.optim.Optimizer,
    beta: float = 0.1,
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
                (input_state, direction_idx, action_mask, move["advantage"])
            )

    # now we create a single batched input tensor
    input_batch = torch.stack([item[0] for item in minibatch])
    target_batch = torch.stack([item[1] for item in minibatch])
    action_mask = torch.tensor([item[2] for item in minibatch])
    advantage = torch.tensor([item[3] for item in minibatch], dtype=torch.float32)

    if device is not None:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        action_mask = action_mask.to(device)
        advantage = advantage.to(device)

    # batches += [
    #     {
    #         "inputs": input_batch,
    #         "target_batch": target_batch,
    #         "advantage": ep["advantage"],
    #     }
    # ]

    # now that the data is prepared, we compute the loss and take an optimizer step
    model.train()
    # for batch in batches:
    #     loss = None
    #     inputs = batch["inputs"]
    #     targets = batch["target_batch"]
    #     advantage = batch["advantage"]

    #     _, loss = model(inputs=inputs, targets=targets)
    #     loss *= advantage

    #     loss /= len(batches)
    #     with torch.no_grad():
    #         total_loss += loss

    #     # now we backprop
    #     loss.backward()

    # loss = None
    # targets = batch["target_batch"]
    # advantage = batch["advantage"]

    logits, unreduced_loss = model(
        inputs=input_batch,
        targets=target_batch,
        action_mask=action_mask,
        reduction="none",
    )
    unreduced_loss *= advantage

    # compute entropy here
    masked_probs = torch.softmax(logits, dim=1)
    entropy = -(masked_probs * (masked_probs + 1e-8).log()).sum(dim=1)

    unreduced_loss -= beta * entropy

    loss = unreduced_loss.sum() / unreduced_loss.numel()
    num_valid = (~action_mask).sum(dim=1).float().mean()
    # print(f" actions: {num_valid:.2f}")

    # # now we also add entropy to the loss
    # from IPython import embed

    # embed()

    # import sys

    # sys.exit()

    # now we backprop
    loss.backward()

    # clip gradnorm and take an optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    # # returns the gradnorm
    # from IPython import embed

    # embed()
    stats = {
        "loss": loss.detach().cpu().item(),
        "grad_norm": grad_norm,
        "entropy": torch.e ** (loss.item()),
    }
    return stats

    # # print optimizer step stats
    # typer.echo("\nOptimizer step completed:")
    # typer.echo(f"  Total loss: {total_loss.item():.4f}")
    # typer.echo(f"  Gradient norm: {grad_norm.item():.4f}")
    # typer.echo(f"  Number of batches: {len(batches)}")


def calculate_advantage(episodes: list[dict], gamma: float):
    """
    One thing that I also want to try is looking at the step-level reward
    and using harmonic mean to figure out how good most of the actions are.
    For example if the model had a chance to make a good merge but didn't then
    this should be penalized heavily
    """

    # Okay now im gonna try step-level rewarding instead
    # since that might be more stable

    # first we calculate the reward of each step
    for ep in episodes:
        total_points = ep["total_points"]

        # we need to be careful with this quantity as it is not strictly causal
        total_possible_points = sum(m["max_points_possible"] for m in ep["moves"])

        # cumsum lol
        # points_possible_cum = 0.0
        # points_earned = 0.0

        # hypothesis 1: we need to use step-level rewards instead of
        # global rewards for proper grading
        # hypothesis 2: rewards should encourage long-term progress rather than short-term optimization

        for move in ep["moves"]:
            # points_possible_cum += move["max_points_possible"]
            move["reward"] = move["points_earned"]

            # if max(move["points_possible"].values()) == 0:
            #     # this is a tricky case, sometimes it might not be possible to
            #     # score any points and the model needs to move it around.
            #     # in these cases, the model should move in a direction which maximizes
            #     # future reward, e.g. two policies:
            #     # P1: X + R2 + ... +
            #     # P2: Y + R2' + ... +
            #     # so if we want P1 > P2 then we want R2 + ... > R2' + ....
            #     # in this case if it moves to a worse direction then it wont get the opportunity
            #     # to make a mistake and so the overall reward should be smaller on average
            #     # but if it moves into a better direction then it gets an opportunity to score higher
            #     # where it will either be rewarded or penalized.

            #     # since this move is hard to penalize or reward because wed need to figure out future possible
            #     # states (can be done via bfs using max_depth=K), lets just keep it as 0 for now
            #     move["reward"] = 0
            #     continue

            # # simple case of grading based on what the available moves are
            # avg_possible_points = sum(move["points_possible"].values()) / 4
            # worst_case = min(move["points_possible"].values())
            # best_case = max(move["points_possible"].values())

            # # reward = move["points_earned"] / (best_case - worst_case) * 2
            # # move["reward"]

            # # simple reward
            # move["reward"] = (
            #     (move["points_earned"] - avg_possible_points)
            #     / (best_case - worst_case)
            #     * 2
            # )

    # next we calculate the discount rate
    total_future_reward = 0.0
    N = 0
    for ep in episodes:
        moves = ep["moves"]
        for i, move in list(enumerate(moves))[::-1]:
            move["future_reward"] = move["reward"]
            if i < len(moves) - 1:  # check last
                move["future_reward"] += gamma * moves[i + 1]["future_reward"]

            # for move in ep["moves"][::-1]:

        # calculate future total reward
        total_future_reward += sum(m["future_reward"] for m in moves)
        N += len(moves)

    # global average
    batch_future_reward_mean = total_future_reward / N

    # now we calculate advantage
    for ep in episodes:
        for m in ep["moves"]:
            m["advantage_unnormalized"] = m["future_reward"] - batch_future_reward_mean

    # now we rescale

    # Here I tried to use the harmonic mean to get the advantage, but this still doesnt work
    # calculate the reward for each model
    # for episode in episodes:
    #     # harmonic mean
    #     episode["reward"] = len(episode["moves"]) / sum(
    #         1 / m["reward"] * max(1.0, m["highest_points"]) if m["reward"] > 0 else 0
    #         for m in episode["moves"]
    #     )

    # calculate advantage
    # avg_reward = len(rollout_episodes) / sum(
    #     1 / e["reward"] if e["reward"] > 0 else 0 for e in rollout_episodes
    # )
    # N = len(episodes)
    # avg_reward = sum(e["reward"] for e in episodes) / N
    # for ep in episodes:
    #     ep["advantage_unnormalized"] = ep["reward"] - avg_reward

    # next, we calculate the normalized advantage

    advantage_avg = (
        sum(move["advantage_unnormalized"] for ep in episodes for move in ep["moves"])
        / N
    )
    advantage_stddev = (
        (1 / (N - 1))
        * sum(
            (m["advantage_unnormalized"] - advantage_avg) ** 2
            for e in episodes
            for m in e["moves"]
        )
    ) ** 0.5

    # add an epsilon for numerical stability
    eps = 1e-8
    for ep in episodes:
        for m in ep["moves"]:
            m["advantage"] = (m["advantage_unnormalized"] - advantage_avg) / (
                advantage_stddev + eps
            )

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

    return episodes

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
    epsilon: float = typer.Option(1.0, "--epsilon", help="Initial exploration rate"),
    momentum: float = typer.Option(
        0.99, "--momentum", help="Momentum used for the EMA of average reward"
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", help="Number of games to play in a given batch"
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
    gpu: bool = typer.Option(False, "--gpu", help="Use CUDA:0 for training"),
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
        typer.echo("this path is not availaable")
        import sys

        sys.exit(0)
    else:
        typer.echo("Playing with random agent (no model specified)")
        model = GameMLP(MLPConfig(hidden_dim=hidden_size, num_layers=num_layers))

    model = model.to(device)

    # After creating the model, before training:
    with torch.no_grad():
        model.proj_down.weight.zero_()
        model.proj_down.bias.zero_()

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.01
    )

    # Add this before the training loop starts
    with torch.no_grad():
        test_game = Game2048()
        test_game.reset()
        test_input = test_game.to_model_format().unsqueeze(0).to(device)
        logits, _ = model(test_input)
        probs = torch.softmax(logits, dim=1)
        print(f"Initial logits: {logits}")
        print(f"Initial probs: {probs}")
        print(f"Initial entropy: {-(probs * probs.log()).sum().item():.4f}")

    # train the model
    highest_score = 0
    for train_step in tqdm(range(steps), desc=f"Running RL training", disable=True):
        model.eval()

        # here we compute a single pass
        rollout_episodes = []

        # game loop
        for i in range(batch_size):
            episode_data = play_game_for_episode(
                model, max_steps=max_steps, device=device
            )
            rollout_episodes.append(episode_data)
            # total_points = episode_data["total_points"]
            # print(
            #     f"Game {i + 1} average reward: {total_reward:.3f}, points: {total_points}"
            # )

        rollout_episodes = calculate_advantage(rollout_episodes, gamma)

        # Now we update the model's policy using this rollout
        stats = model_optimize_step(model, rollout_episodes, optimizer, beta, device)

        # Get highest score from batch
        highest_score = max(
            max([episode["total_points"] for episode in rollout_episodes]),
            highest_score,
        )

        # Calculate average number of steps in batch
        avg_steps = sum([len(episode["moves"]) for episode in rollout_episodes]) / len(
            rollout_episodes
        )

        # Print stats instead of updating progress bar
        if train_step % 25 == 0:
            advantages = [
                m["advantage"] for ep in rollout_episodes for m in ep["moves"]
            ]
            min_adv = min(advantages)
            max_adv = max(advantages)
            print(
                f"Step {train_step}: loss={stats['loss']:.4f}, "
                f"entropy={stats.get('entropy', 0):.4f}, "
                f"grad_norm={stats.get('grad_norm', 0):.4f}, "
                f"peak_score={highest_score}, "
                f"avg_steps={avg_steps:.1f}, "
                f"avg_score={sum([episode['total_points'] for episode in rollout_episodes]) / len(rollout_episodes):.1f}, "
                f"highest_score={max([episode['total_points'] for episode in rollout_episodes]):.1f}, "
                f"adv_range=[{min_adv:.3f}, {max_adv:.3f}]"
            )


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


def display_board(game: Game2048) -> None:
    """Display the game board in a nice format."""
    typer.echo("\n" + "=" * 25)
    for row in game.grid:
        # convert exponents to actual values (0 stays 0, exponent k becomes 2^k)
        display_row = [str(2**cell if cell > 0 else 0).rjust(5) for cell in row]
        typer.echo("│" + "│".join(display_row) + "│")
    typer.echo("=" * 25)
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
