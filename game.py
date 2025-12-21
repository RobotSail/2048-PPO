GRID_SIZE = 4

type Grid = list[list[int]]
from enum import Enum
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


from pydantic import BaseModel


class MLPConfig(BaseModel):
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class Game2048:
    grid: Grid

    def __init__(self, state: Grid = None):
        """
        Grid stores exponent values where cell value = 2^exponent.
        Empty cells are represented as 0.
        """
        if not state:
            self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            return

        # otherwise we have to validate the game state
        # Valid values are 0 (empty) or exponents 1-16 (representing 2^1 to 2^16)
        valid_nums = list(range(0, 17))
        assert all(s in valid_nums for row in state for s in row)
        self.grid = state

    def score(self) -> int:
        return sum([2**k for row in self.grid for k in row if k > 0])

    @classmethod
    def from_tensor(cls, state: torch.Tensor) -> "Game2048":
        if state.ndim != GRID_SIZE**2:
            raise ValueError(
                f"received inappropriate dimensions for creating a new grid"
            )

        if state.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"dimensions are inappropriate for a grid")
        native_state = state.tolist()
        return cls(native_state)

    @staticmethod
    def create_random_board(generator: torch.Generator = None) -> Grid:
        # create a random board
        board = torch.zeros((GRID_SIZE, GRID_SIZE))

        # Select 2 random indices without replacement
        flat_indices = torch.randperm(GRID_SIZE * GRID_SIZE, generator=generator)[:2]

        # Convert flat indices to 2D coordinates and set to 1 (exponent for value 2)
        for idx in flat_indices:
            row = idx // GRID_SIZE
            col = idx % GRID_SIZE
            board[row, col] = 1  # exponent 1 represents 2^1 = 2

        return board.tolist()

    def to_model_format(self) -> torch.Tensor:
        outputs = torch.tensor(self.grid).to(torch.float32)  # now should in shape 4x4

        # divide by 3 so that 0 = 0, 3 = 1
        flattened_outputs = outputs.view(-1)
        row_idxs = (torch.arange(GRID_SIZE**2) // 4) / 3
        col_idxs = (torch.arange(GRID_SIZE**2) % 4) / 3

        # concatenate the indices to each cell and flatten into a 1d vec
        return torch.stack((flattened_outputs, row_idxs, col_idxs), dim=1).view(-1)

    def has_next_step(self) -> bool:
        return self.state_has_next_step(self.grid)

    @staticmethod
    def state_has_next_step(state: Grid) -> bool:
        # it has a next step when it can move left, right, up, or down
        for direction in Direction._member_map_.values():
            if Game2048.can_move_in_direction(
                state=state, direction=direction
            ) or Game2048.can_merge_in_direction(state=state, direction=direction):
                return True
        return False

    def direction_has_step(self, direction: Direction) -> bool:
        return Game2048.can_move_in_direction(
            self.grid, direction
        ) or Game2048.can_merge_in_direction(self.grid, direction)

    @staticmethod
    def simulate_move(grid: Grid, direction: Direction) -> tuple[Grid, int]:
        """
        Simulate a move on a grid copy without mutating the original.
        Returns (resulting_grid, score_gained_from_merges).
        """
        new_grid = [row[:] for row in grid]
        total_score = 0

        if direction == Direction.UP:
            working_grid = [
                [new_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
            results = [
                Game2048._merge_and_shift_left_with_score(row) for row in working_grid
            ]
            working_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
            new_grid = [
                [working_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
        elif direction == Direction.DOWN:
            working_grid = [
                [new_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
            results = [
                Game2048._merge_and_shift_right_with_score(row) for row in working_grid
            ]
            working_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
            new_grid = [
                [working_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
        elif direction == Direction.LEFT:
            results = [
                Game2048._merge_and_shift_left_with_score(row) for row in new_grid
            ]
            new_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
        elif direction == Direction.RIGHT:
            results = [
                Game2048._merge_and_shift_right_with_score(row) for row in new_grid
            ]
            new_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)

        return new_grid, total_score

    @staticmethod
    def calculate_grid_score(grid: Grid) -> int:
        """Calculate the sum of tile values on a grid."""
        return sum(2**k for row in grid for k in row if k > 0)

    def preview_move_rewards(self) -> dict[Direction, int]:
        """
        Calculate potential points gained for each direction without mutating state.

        Returns a dict mapping Direction -> points gained from merges.
        Invalid moves return 0. Each direction's calculation is independent.
        """
        results = {}

        for direction in Direction:
            if not self.direction_has_step(direction):
                results[direction] = 0
                continue

            _, score_gained = Game2048.simulate_move(self.grid, direction)
            results[direction] = score_gained

        return results

    def move(self, direction: Direction) -> Grid:
        """
        Compute the resulting grid after moving in the specified direction.
        Raises ValueError if the direction has no valid move.
        """
        if not self.direction_has_step(direction):
            raise ValueError(f"Cannot move in direction {direction.value}")

        # Create a copy of the grid to work with
        new_grid = [row[:] for row in self.grid]

        # Transform the grid based on direction
        if direction == Direction.UP:
            # Transpose to work with columns as rows
            working_grid = [
                [new_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
            working_grid = [self._merge_and_shift_left(row) for row in working_grid]
            # Transpose back
            new_grid = [
                [working_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
        elif direction == Direction.DOWN:
            # Transpose to work with columns as rows
            working_grid = [
                [new_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
            working_grid = [self._merge_and_shift_right(row) for row in working_grid]
            # Transpose back
            new_grid = [
                [working_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
        elif direction == Direction.LEFT:
            new_grid = [self._merge_and_shift_left(row) for row in new_grid]
        elif direction == Direction.RIGHT:
            new_grid = [self._merge_and_shift_right(row) for row in new_grid]

        self.grid = new_grid
        return new_grid

    @staticmethod
    def _merge_and_shift_left(row: list[int]) -> list[int]:
        """Merge and shift a row to the left. Row contains exponents."""
        result, _ = Game2048._merge_and_shift_left_with_score(row)
        return result

    @staticmethod
    def _merge_and_shift_left_with_score(row: list[int]) -> tuple[list[int], int]:
        """Merge and shift a row to the left, returning (new_row, score_gained)."""
        non_zero = [x for x in row if x != 0]

        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                new_exp = non_zero[i] + 1
                merged.append(new_exp)
                score += 2**new_exp  # points = value of merged tile
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        return merged + [0] * (GRID_SIZE - len(merged)), score

    @staticmethod
    def _merge_and_shift_right(row: list[int]) -> list[int]:
        """Merge and shift a row to the right. Row contains exponents."""
        result, _ = Game2048._merge_and_shift_right_with_score(row)
        return result

    @staticmethod
    def _merge_and_shift_right_with_score(row: list[int]) -> tuple[list[int], int]:
        """Merge and shift a row to the right, returning (new_row, score_gained)."""
        reversed_row = row[::-1]
        merged, score = Game2048._merge_and_shift_left_with_score(reversed_row)
        return merged[::-1], score

    @staticmethod
    def can_move_in_direction(state: Grid, direction: Direction) -> bool:
        # given the direction, can we make a move in this direction?

        _state = state
        if direction in [Direction.UP, Direction.DOWN]:
            state_cols = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            # transpose, so we represent our rows as colummns
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    state_cols[i][j] = state[j][i]
            _state = state_cols

            # direction DOWN --> RIGHT
            # direction UP --> LEFT
            if direction == Direction.UP:
                direction = Direction.LEFT

        # here we scan either right or left
        # if direction in [Direction.LEFT, Direction.RIGHT]:
        if direction == Direction.LEFT:
            _state = [row[::-1] for row in _state]

        can_move = False
        for row in _state:
            found_value = False
            for item in row:
                if item > 0:
                    found_value = True

                if found_value and item == 0:
                    can_move = True
                    break

        return can_move

    def current_valid_directions(self) -> list[Direction]:
        """
        Returns the list of valid directions we can take a move in under the current state.
        """
        return [
            d for d in Direction._member_map_.values() if self.direction_has_step(d)
        ]

    @staticmethod
    def can_merge_in_direction(state: Grid, direction: Direction) -> bool:
        # given the direction, are 2 elements available to be merged?

        _state = state
        if direction in [Direction.UP, Direction.DOWN]:
            state_cols = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            # transpose, so we represent our rows as colummns
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    state_cols[i][j] = state[j][i]
            _state = state_cols

            # direction DOWN --> RIGHT
            # direction UP --> LEFT
            if direction == Direction.UP:
                direction = Direction.LEFT

        # here we scan either right or left
        # if direction in [Direction.LEFT, Direction.RIGHT]:
        if direction == Direction.LEFT:
            _state = [row[::-1] for row in _state]

        can_merge = False
        for row in _state:
            for a, b in zip(row[:-1:], row[1::]):
                if a == b and a != 0:
                    can_merge = True

        return can_merge

    def get_score(self) -> int:
        """
        Returns the score of the current game
        """
        return self.score()

    def _add_tile(self) -> bool:
        """
        Add a new tile (90% chance of 2, 10% chance of 4) to a random empty cell.
        Returns True if a tile was added, False if no empty cells available.
        """
        empty_cells = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 0:
                    empty_cells.append((i, j))

        if not empty_cells:
            return False

        row, col = random.choice(empty_cells)
        # 90% chance of 2 (exponent=1), 10% chance of 4 (exponent=2)
        self.grid[row][col] = 1 if random.random() < 0.9 else 2
        return True

    def reset(self) -> Grid:
        """
        Reset the game to initial state with 2 random tiles.
        Returns the initial grid.
        """
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self._add_tile()
        self._add_tile()
        return [row[:] for row in self.grid]

    def step(self, direction: Direction) -> tuple[Grid, int, bool, dict]:
        """
        Take a step in the game by moving in the specified direction.
        Returns (new_state, reward, done, info) for RL compatibility.
        Reward is the points gained from merging tiles.
        """
        if not self.direction_has_step(direction):
            return (
                [row[:] for row in self.grid],
                0,
                not self.has_next_step(),
                {"invalid_move": True},
            )

        new_grid, points_earned = Game2048.simulate_move(self.grid, direction)
        self.grid = new_grid

        # spawn a new tile after successful move
        self._add_tile()
        done = not self.has_next_step()

        return (
            [row[:] for row in self.grid],
            points_earned,
            done,
            {"invalid_move": False},
        )


class ResidualBlock(nn.Module):
    """A residual block with skip connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear(x)
        x = self.dropout(x)
        x = self.ln(x)
        x = self.activation(x)
        return x + residual  # skip connection


class GameMLP(nn.Module):
    """
    MLP with configurable depth and residual connections.
    """

    @staticmethod
    def init_kaiming(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    N = 16  # grid size of 16
    NUM_ACTIONS = 4

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        # project input to hidden dimension
        self.proj_up = nn.Linear(
            in_features=self.N * 3, out_features=config.hidden_dim, bias=True
        )
        self.input_ln = nn.LayerNorm(config.hidden_dim)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(config.dropout)

        # stack of residual blocks
        self.hidden_layers = nn.ModuleList(
            [
                ResidualBlock(config.hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        # project to action logits (no LayerNorm on output to allow confident predictions)
        self.proj_down = nn.Linear(
            in_features=config.hidden_dim, out_features=self.NUM_ACTIONS, bias=True
        )

        self.apply(GameMLP.init_kaiming)

    @property
    def directions(self) -> list[Direction]:
        """
        These are the directions that the MLP outputs represent
        """
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        f"""
        Given the set of inputs in $R^{16}$, returns a probability function over the action
        space and optionally a loss if targets were specified


        The `inputs` tensor should be an N * 3-lenth tensor with row/col indices appended
        `T` and `B` in this case are treated as batch dimensions.
        
        `targets` should be a list of class probabilies over the action space

        """

        # dimensions of input
        B = 1

        if inputs.ndim <= 1:
            raise ValueError(
                f"input must consist of shape (batch, channel), got: {inputs.shape}"
            )

        # handles single-length vector inputs
        C = inputs.shape[-1]
        assert C == self.N * 3, f"{C} does not equal {self.N * 3}"

        # # next we concatenate the inputs to be the expected shape
        # col_idxs = torch.arange(square_width) % 4
        # row_idxs = torch.arange(square_width) // 4

        # # cast to devices
        # col_idxs = col_idxs.to(inputs.device)
        # row_idxs = row_idxs.to(inputs.device)

        # full input
        # this forms a matrix of the shape
        # [x1 c1 r1]
        # [x2 c2 r1]
        # [x3 c3 r1]
        # [x4 c4 r1]
        # [x5 c1 r2]
        # ....
        # [x16 c4 r4]
        # full_input = torch.stack((inputs, col_idxs, row_idxs), dim=1)

        # project input to hidden dimension
        x = inputs.to(dtype=torch.float32)  # (B, N*3)
        x = self.proj_up(x)  # (B, H)
        x = self.input_dropout(x)
        x = self.input_ln(x)
        x = self.input_activation(x)

        # pass through residual blocks
        for layer in self.hidden_layers:
            x = layer(x)

        # project to action logits
        x = self.proj_down(x)  # (B, NUM_ACTIONS)

        # now we normalize to logits
        loss = None
        if targets is not None:
            targets = targets.to(device=x.device)

            # convert (B, T) into (B x T)
            targets = targets.view(-1)
            x[action_mask] = -torch.inf  # mask to kill the training signal

            # compute the loss here
            loss = F.cross_entropy(input=x, target=targets, reduction=reduction)

        # returns the raw logits + loss (if applicable)
        return (x, loss)


def generate_random_games(n=1) -> list[Game2048]:
    games = []
    while len(games) < n:
        random_grid = Game2048.create_random_board()
        game = Game2048(random_grid)
        assert game.has_next_step()
        games.append(game)
    return games


# attempt a model forward
if __name__ == "__main__":
    games = generate_random_games()
    # stacked_input = torch.stack((input_list, col_idxs, row_idxs), dim=1).view(-1)

    # format the games into an input of the shape (batch, game)
    boards = [game.to_model_format() for game in generate_random_games(n=3)]

    stacked = torch.stack(boards)

    model = GameMLP(MLPConfig(hidden_dim=64))
    logits, _ = model(stacked)

    print(logits)
