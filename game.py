GRID_SIZE = 4

type Grid = list[list[int]]
from enum import Enum
import random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


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


class GameURMConfig(BaseModel):
    """Configuration for the Universal Reasoning Model adapted for 2048 game."""

    hidden_dim: int = 64
    num_layers: int = 2  # Number of transformer blocks
    num_heads: int = 4
    expansion: float = 2.67  # MLP expansion factor (SwiGLU uses 2/3 of this)
    dropout: float = 0.1
    num_loops: int = 4  # Number of recurrent reasoning loops
    num_truncated_loops: int = 1  # Loops without gradient (forward-only)
    conv_kernel: int = 2  # Short convolution kernel size
    rms_norm_eps: float = 1e-5


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
    def simulate_move(grid: Grid, direction: Direction) -> tuple[Grid, int, int]:
        """
        Simulate a move on a grid copy without mutating the original.
        Returns (resulting_grid, score_gained_from_merges, max_tile_created_exponent).
        """
        new_grid = [row[:] for row in grid]
        total_score = 0
        max_tile_created = 0

        if direction == Direction.UP:
            working_grid = [
                [new_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
            results = [
                Game2048._merge_and_shift_left_with_score(row) for row in working_grid
            ]
            working_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
            max_tile_created = max(r[2] for r in results)
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
            max_tile_created = max(r[2] for r in results)
            new_grid = [
                [working_grid[j][i] for j in range(GRID_SIZE)] for i in range(GRID_SIZE)
            ]
        elif direction == Direction.LEFT:
            results = [
                Game2048._merge_and_shift_left_with_score(row) for row in new_grid
            ]
            new_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
            max_tile_created = max(r[2] for r in results)
        elif direction == Direction.RIGHT:
            results = [
                Game2048._merge_and_shift_right_with_score(row) for row in new_grid
            ]
            new_grid = [r[0] for r in results]
            total_score = sum(r[1] for r in results)
            max_tile_created = max(r[2] for r in results)

        return new_grid, total_score, max_tile_created

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

            _, score_gained, _ = Game2048.simulate_move(self.grid, direction)
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
        result, _, _ = Game2048._merge_and_shift_left_with_score(row)
        return result

    @staticmethod
    def _merge_and_shift_left_with_score(row: list[int]) -> tuple[list[int], int, int]:
        """Merge and shift a row to the left, returning (new_row, score_gained, max_tile_created)."""
        non_zero = [x for x in row if x != 0]

        merged = []
        score = 0
        max_tile_created = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                new_exp = non_zero[i] + 1
                merged.append(new_exp)
                score += 2**new_exp  # points = value of merged tile
                max_tile_created = max(max_tile_created, new_exp)
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1

        return merged + [0] * (GRID_SIZE - len(merged)), score, max_tile_created

    @staticmethod
    def _merge_and_shift_right(row: list[int]) -> list[int]:
        """Merge and shift a row to the right. Row contains exponents."""
        result, _, _ = Game2048._merge_and_shift_right_with_score(row)
        return result

    @staticmethod
    def _merge_and_shift_right_with_score(row: list[int]) -> tuple[list[int], int, int]:
        """Merge and shift a row to the right, returning (new_row, score_gained, max_tile_created)."""
        reversed_row = row[::-1]
        merged, score, max_tile = Game2048._merge_and_shift_left_with_score(
            reversed_row
        )
        return merged[::-1], score, max_tile

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

    @staticmethod
    def smoothness_score(grid: Grid) -> float:
        """
        Measure board smoothness: how similar are adjacent tiles?
        Higher score = smoother = better (tiles clustered by similar values).

        Returns negative sum of exponent differences between adjacent non-empty cells.
        """
        score = 0.0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] == 0:
                    continue
                # check right neighbor
                if j < GRID_SIZE - 1 and grid[i][j + 1] != 0:
                    score -= abs(grid[i][j] - grid[i][j + 1])
                # check down neighbor
                if i < GRID_SIZE - 1 and grid[i + 1][j] != 0:
                    score -= abs(grid[i][j] - grid[i + 1][j])
        return score

    @staticmethod
    def corner_bonus(grid: Grid) -> float:
        """
        Returns a bonus/penalty based on max tile position.
        - Positive bonus (exponent) if ANY max tile is in a corner
        - Negative penalty (-exponent) if NO max tile is in a corner

        This creates a strong incentive to keep the max tile in a corner,
        and penalizes moving it away.
        """
        corners = {
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        }

        # Find the max value
        max_val = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] > max_val:
                    max_val = grid[i][j]

        if max_val == 0:
            return 0.0

        # Check if ANY max tile is in a corner
        max_in_corner = False
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] == max_val and (i, j) in corners:
                    max_in_corner = True
                    break
            if max_in_corner:
                break

        if max_in_corner:
            return float(max_val)  # Positive bonus for having max in corner
        else:
            return -float(max_val)  # Negative penalty for NOT having max in corner

    @staticmethod
    def adjacency_bonus(grid: Grid) -> float:
        """
        Returns a bonus for having high-value tiles adjacent to the maximum tile.
        Also rewards pairs of high-value tiles (exponent >= 5, i.e., 32+) that are adjacent.
        """
        # Find max tile position
        max_val = 0
        max_pos = (0, 0)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] > max_val:
                    max_val = grid[i][j]
                    max_pos = (i, j)

        bonus = 0.0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Bonus for tiles adjacent to max tile
        for di, dj in directions:
            ni, nj = max_pos[0] + di, max_pos[1] + dj
            if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                neighbor_val = grid[ni][nj]
                if neighbor_val > 0:
                    # Reward based on how close the neighbor is to max (smaller diff = better)
                    # Also scale by the neighbor's value
                    bonus += neighbor_val * 0.5  # Half the exponent as bonus

        # Bonus for adjacent high-value pairs (exponent >= 5 means tile >= 32)
        HIGH_VALUE_THRESHOLD = 5
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] >= HIGH_VALUE_THRESHOLD:
                    # Check right neighbor
                    if j < GRID_SIZE - 1 and grid[i][j + 1] >= HIGH_VALUE_THRESHOLD:
                        # Bonus = sum of exponents for the pair
                        bonus += (grid[i][j] + grid[i][j + 1]) * 0.25
                    # Check down neighbor
                    if i < GRID_SIZE - 1 and grid[i + 1][j] >= HIGH_VALUE_THRESHOLD:
                        bonus += (grid[i][j] + grid[i + 1][j]) * 0.25

        return bonus

    @staticmethod
    def monotonic_chain_score(grid: Grid) -> float:
        """
        Calculate the length of the longest monotonically decreasing chain
        starting from ANY maximum tile.

        A valid chain requires each step to be exactly one exponent lower than
        the previous (e.g., 512 -> 256 -> 128 -> 64).

        This rewards the "snake pattern" that strong 2048 players use.
        Returns the sum of exponents in the chain (weighted by position).
        """
        # Find max value
        max_val = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] > max_val:
                    max_val = grid[i][j]

        if max_val == 0:
            return 0.0

        # Find ALL positions with max value
        max_positions = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] == max_val:
                    max_positions.append((i, j))

        # DFS to find the longest chain of consecutive descending exponents
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def dfs(pos: tuple[int, int], expected_exp: int, visited: set) -> float:
            """DFS to find chain, returns total score of this branch."""
            if pos in visited:
                return 0.0
            i, j = pos
            if not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE):
                return 0.0
            if grid[i][j] != expected_exp:
                return 0.0

            visited.add(pos)
            # Score: exponent value, weighted slightly by depth to prefer longer chains
            score = float(expected_exp)

            # Continue chain with neighbors that are exactly one exponent lower
            best_continuation = 0.0
            for di, dj in directions:
                ni, nj = i + di, j + dj
                continuation = dfs((ni, nj), expected_exp - 1, visited)
                best_continuation = max(best_continuation, continuation)

            visited.remove(pos)  # Allow other branches to use this cell
            return score + best_continuation

        # Try starting from EACH max tile and take the best result
        best_chain_score = 0.0
        for max_pos in max_positions:
            chain_score = dfs(max_pos, max_val, set())
            best_chain_score = max(best_chain_score, chain_score)

        return best_chain_score

    @staticmethod
    def _position_multiplier(row: int, col: int) -> float:
        """
        Returns a multiplier based on position:
        - Corner: 1.0 (full bonus)
        - Edge (not corner): 0.2 (partial bonus)
        - Center: 0.0 (no bonus)
        """
        is_edge_row = row == 0 or row == GRID_SIZE - 1
        is_edge_col = col == 0 or col == GRID_SIZE - 1

        if is_edge_row and is_edge_col:
            return 1.0  # Corner
        elif is_edge_row or is_edge_col:
            return 0.2  # Edge but not corner
        else:
            return 0.0  # Center

    @staticmethod
    def _get_snake_order(corner: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Generate the ideal snake path starting from the given corner.
        Returns list of (row, col) positions in order from corner outward.
        """
        cr, cc = corner
        # Determine direction multipliers based on corner
        row_dir = 1 if cr == 0 else -1
        col_dir = 1 if cc == 0 else -1

        order = []
        for i in range(GRID_SIZE):
            row = cr + i * row_dir
            # Alternate direction each row for snake pattern
            if i % 2 == 0:
                cols = range(cc, cc + GRID_SIZE * col_dir, col_dir)
            else:
                cols = range(cc + (GRID_SIZE - 1) * col_dir, cc - col_dir, -col_dir)
            for col in cols:
                if 0 <= col < GRID_SIZE:
                    order.append((row, col))
        return order

    @staticmethod
    def _choose_anchor_corner(grid: Grid) -> tuple[int, int]:
        """
        Select a consistent anchor corner for topological scoring.
        Prefer the corner that already holds a max tile; otherwise pick
        the corner closest (Manhattan distance) to the first max tile found.
        """
        corners = [
            (0, 0),
            (0, GRID_SIZE - 1),
            (GRID_SIZE - 1, 0),
            (GRID_SIZE - 1, GRID_SIZE - 1),
        ]

        max_val = 0
        max_positions = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] > max_val:
                    max_val = grid[i][j]
                    max_positions = [(i, j)]
                elif grid[i][j] == max_val and max_val > 0:
                    max_positions.append((i, j))

        if not max_positions:
            return corners[0]

        # If any max tile is already in a corner, lock to that corner.
        for pos in max_positions:
            if pos in corners:
                return pos

        # Otherwise choose the closest corner to the first max tile (deterministic).
        target = max_positions[0]
        return min(corners, key=lambda c: abs(c[0] - target[0]) + abs(c[1] - target[1]))

    @staticmethod
    def monotonicity(grid: Grid, require_corner_max: bool = False) -> int:
        """
        Calculate a game's monotonicity score by evaluating how well tiles
        are arranged in monotonically increasing/decreasing sequences.

        The algorithm checks all four rotations of the board (0°, 90°, 180°, 270°)
        and returns the best monotonicity score found.

        For each rotation, it counts how many adjacent pairs satisfy the
        monotonicity condition (left >= right and top >= bottom).

        Args:
            grid: The game board state.
            require_corner_max: If True (default), applies a penalty when the
                maximum tile is not in a corner. Per the original algorithm
                description, high monotonicity requires the max tile to be in
                a corner. When enabled, the score is halved if max is not in
                a corner.

        Returns the highest score across all rotations (potentially penalized).
        """
        best = -1

        # Try all 4 rotations (0°, 90°, 180°, 270°)
        current_grid = [row[:] for row in grid]

        for rotation in range(4):
            current = 0

            # Check horizontal monotonicity (left to right)
            # only count pairs where both cells have tiles (skip empty cells)
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE - 1):
                    left = current_grid[row][col]
                    right = current_grid[row][col + 1]
                    if left > 0 and right > 0 and left >= right:
                        current += 1

            # Check vertical monotonicity (top to bottom)
            for col in range(GRID_SIZE):
                for row in range(GRID_SIZE - 1):
                    top = current_grid[row][col]
                    bottom = current_grid[row + 1][col]
                    if top > 0 and bottom > 0 and top >= bottom:
                        current += 1

            if current > best:
                best = current

            # Rotate the board 90 degrees clockwise for next iteration
            current_grid = [
                [current_grid[GRID_SIZE - 1 - j][i] for j in range(GRID_SIZE)]
                for i in range(GRID_SIZE)
            ]

        if require_corner_max:
            # check if max tile is in one of the four corners
            max_val = max(max(row) for row in grid)
            corners = [
                (0, 0),
                (0, GRID_SIZE - 1),
                (GRID_SIZE - 1, 0),
                (GRID_SIZE - 1, GRID_SIZE - 1),
            ]
            max_in_corner = any(grid[r][c] == max_val for r, c in corners)

            if not max_in_corner:
                best = 0

        return best

    @staticmethod
    def topological_score(
        grid: Grid, anchor_corner: tuple[int, int] | None = None
    ) -> float:
        """
        Calculate a gradient-based topological score.

        The ideal 2048 board has tiles arranged in a "snake" pattern from a corner,
        with values monotonically decreasing along the path. This function measures
        how well the current board matches this ideal.

        Scoring components:
        1. Gradient consistency: Reward tiles that follow monotonic decrease along snake
        2. Inversion penalty: Penalize when higher tiles are "behind" lower ones
        3. Trapped tile penalty: High tiles surrounded by much lower tiles
        4. Corner anchoring: Bonus for max tile in corner

        Returns a score where higher = better organized board.
        """
        # Find all non-zero tiles
        tiles = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] > 0:
                    tiles.append((grid[i][j], i, j))

        if not tiles:
            return 0.0

        # Find the max tile value and its positions
        max_val = max(t[0] for t in tiles)

        # Define corners and find the best one (where max tile is, or best gradient)
        corners = (
            [anchor_corner]
            if anchor_corner is not None
            else [
                (0, 0),
                (0, GRID_SIZE - 1),
                (GRID_SIZE - 1, 0),
                (GRID_SIZE - 1, GRID_SIZE - 1),
            ]
        )

        best_score = float("-inf")

        for corner in corners:
            snake_order = Game2048._get_snake_order(corner)

            # Create position-to-snake-index mapping
            pos_to_idx = {pos: idx for idx, pos in enumerate(snake_order)}

            score = 0.0

            # 1. Gradient consistency score
            # For each tile, check if tiles earlier in snake order are >= its value
            for val, row, col in tiles:
                tile_idx = pos_to_idx[(row, col)]

                # Bonus for being in a good position relative to value rank
                # Higher values should be earlier in the snake order
                position_bonus = (16 - tile_idx) * val * 0.1
                score += position_bonus

            # 2. Monotonicity along snake path
            prev_val = float("inf")
            monotonic_bonus = 0.0
            inversion_penalty = 0.0

            for pos in snake_order:
                row, col = pos
                val = grid[row][col]
                if val == 0:
                    continue

                if val <= prev_val:
                    # Good: value is decreasing or equal (merge potential)
                    monotonic_bonus += val * 0.2
                else:
                    # Bad: inversion - higher value appears later in snake
                    # Penalty proportional to how "wrong" this is
                    inversion_penalty += (val - prev_val) * 0.5

                prev_val = val

            score += monotonic_bonus - inversion_penalty

            # 3. Max tile anchoring bonus
            cr, cc = corner
            if grid[cr][cc] == max_val:
                score += max_val * 2.0  # Strong bonus for max in corner

            # 4. Trapped tile penalty
            # A tile is "trapped" if it's high value but surrounded by much lower tiles
            # and not in a good snake position
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for val, row, col in tiles:
                if val < 4:  # Only check tiles >= 16 (exponent 4)
                    continue

                tile_idx = pos_to_idx[(row, col)]
                neighbors_lower = 0
                total_neighbors = 0

                for di, dj in directions:
                    ni, nj = row + di, col + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        neighbor_val = grid[ni][nj]
                        if neighbor_val > 0:
                            total_neighbors += 1
                            if neighbor_val < val - 2:  # 2+ exponents lower
                                neighbors_lower += 1

                # Trapped: high tile surrounded mostly by much lower tiles
                # and in a bad snake position (late in order)
                if total_neighbors >= 2 and neighbors_lower >= total_neighbors - 1:
                    if tile_idx > 4:  # Not in first few snake positions
                        score -= val * 1.0  # Penalty for trapped high tile

            best_score = max(best_score, score)

        return best_score

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
        Info includes deltas for reward shaping heuristics.
        """
        if not self.direction_has_step(direction):
            return (
                [row[:] for row in self.grid],
                0,
                not self.has_next_step(),
                {
                    "invalid_move": True,
                    "smoothness_delta": 0.0,
                    "max_tile_created": 0,
                    "corner_delta": 0.0,
                    "adjacency_delta": 0.0,
                    "chain_delta": 0.0,
                    # "monotonicity_delta": 0.0,
                    "monotonicity_before": 0.0,
                    "monotonicity_after": 0.0,
                    "topological_delta": 0.0,
                },
            )

        # compute heuristics before move
        smoothness_before = Game2048.smoothness_score(self.grid)
        corner_before = Game2048.corner_bonus(self.grid)
        adjacency_before = Game2048.adjacency_bonus(self.grid)
        chain_before = Game2048.monotonic_chain_score(self.grid)
        monotonicity_before = Game2048.monotonicity(self.grid)
        anchor_corner = Game2048._choose_anchor_corner(self.grid)
        topological_before = Game2048.topological_score(self.grid, anchor_corner)
        max_exp_before = max(max(row) for row in self.grid)

        new_grid, points_earned, max_tile_created = Game2048.simulate_move(
            self.grid, direction
        )
        self.grid = new_grid

        # compute heuristics after move but before random spawn (to avoid reward hacking)
        smoothness_after = Game2048.smoothness_score(new_grid)
        corner_after = Game2048.corner_bonus(new_grid)
        adjacency_after = Game2048.adjacency_bonus(new_grid)
        chain_after = Game2048.monotonic_chain_score(new_grid)
        monotonicity_after = Game2048.monotonicity(new_grid)
        topological_after = Game2048.topological_score(new_grid, anchor_corner)
        max_exp_after = max(max(row) for row in new_grid)

        # spawn a new tile after successful move
        self._add_tile()
        done = not self.has_next_step()

        return (
            [row[:] for row in self.grid],
            points_earned,
            done,
            {
                "invalid_move": False,
                "smoothness_delta": smoothness_after - smoothness_before,
                "max_tile_created": max_tile_created,
                "max_exponent_before": max_exp_before,
                "max_exponent_after": max_exp_after,
                "corner_delta": corner_after - corner_before,
                "adjacency_delta": adjacency_after - adjacency_before,
                "chain_delta": chain_after - chain_before,
                # well intentioned but i dont need this
                # "monotonicity_delta": monotonicity_after - monotonicity_before,
                "monotonicity_before": monotonicity_before,
                "monotonicity_after": monotonicity_after,
                "topological_delta": topological_after - topological_before,
                "topological_anchor": anchor_corner,
            },
        )


class ResidualBlock(nn.Module):
    """A residual block with skip connection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)  # skip connection


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

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(
                in_features=self.N * 3, out_features=config.hidden_dim, bias=False
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )

        # stack of residual blocks
        self.backbone = nn.ModuleList(
            [
                ResidualBlock(config.hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        # action head
        self.action_head = nn.Linear(
            in_features=config.hidden_dim, out_features=self.NUM_ACTIONS, bias=True
        )
        # value head
        self.value_head = nn.Linear(
            in_features=config.hidden_dim, out_features=1, bias=True
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = self.stem(x)

        # pass through residual blocks
        for layer in self.backbone:
            x = layer(x)

        # project to action logits
        action_logits = self.action_head(x)  # (B, NUM_ACTIONS)
        value_logit = self.value_head(x)

        # now we normalize to logits
        # loss = None
        # if targets is not None:
        #     targets = targets.to(device=action_logits.device)

        #     # convert (B, T) into (B x T)
        #     targets = targets.view(-1)
        #     x[action_mask] = -torch.inf  # mask to kill the training signal

        #     # compute the loss here
        #     loss = F.cross_entropy(
        #         input=action_logits, target=targets, reduction=reduction
        #     )

        # # returns the raw logits + loss (if applicable)
        return (action_logits, value_logit)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """RMS normalization without learnable parameters."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class GameConvSwiGLU(nn.Module):
    """
    SwiGLU with depthwise short convolution for local context mixing.

    This is the key innovation from URM: adds a 1D depthwise convolution
    after the gated activation to inject local token interactions.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 2,
    ):
        super().__init__()
        # Calculate intermediate size (SwiGLU uses 2/3 of expansion)
        inter = round(expansion * hidden_size * 2 / 3)
        # Round up to multiple of 8 for efficiency
        inter = ((inter + 7) // 8) * 8
        self.inter = inter

        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter,  # Depthwise: each channel has its own kernel
            bias=True,
        )
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up  # SwiGLU gating

        # Apply depthwise convolution for local mixing
        # (batch, seq, inter) -> (batch, inter, seq) for conv1d
        x_conv = self.dwconv(x_ffn.transpose(1, 2))
        x_conv = x_conv[..., : x_ffn.size(1)]  # Trim to original seq length
        x_conv = F.silu(x_conv)  # Additional nonlinearity after conv
        x_conv = x_conv.transpose(1, 2).contiguous()

        return self.down_proj(x_conv)


class GameURMAttention(nn.Module):
    """
    Multi-head self-attention for the 2048 game board.

    Uses scaled dot-product attention (no causal masking needed for 2048).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention (no causal mask for 2048)
        attn_output = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class GameURMBlock(nn.Module):
    """
    A single URM transformer block with attention + ConvSwiGLU.

    Architecture:
        x -> Attention -> + -> RMSNorm -> ConvSwiGLU -> + -> RMSNorm -> out
             └──────────┘              └───────────────┘
    """

    def __init__(self, config: GameURMConfig):
        super().__init__()
        self.attn = GameURMAttention(
            hidden_size=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.mlp = GameConvSwiGLU(
            hidden_size=config.hidden_dim,
            expansion=config.expansion,
            conv_kernel=config.conv_kernel,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        attn_output = self.attn(hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, self.norm_eps)

        # Pre-norm MLP with residual
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, self.norm_eps)

        return hidden_states


class GameURM(nn.Module):
    """
    Universal Reasoning Model adapted for the 2048 game.

    This implements the key URM innovations:
    1. Recurrent loops: Same transformer blocks applied multiple times
    2. ConvSwiGLU: SwiGLU MLP with short convolution for local context
    3. Truncated backprop: First N loops are forward-only (no gradients)

    Input: (batch, 48) tensor with 16 cells × 3 features (value, row_idx, col_idx)
    Output: (action_logits, value_logit) tuple
    """

    N = 16  # Grid size (4x4 = 16 cells)
    NUM_ACTIONS = 4

    def __init__(self, config: GameURMConfig):
        super().__init__()
        self.config = config

        # Input projection: 3 features per cell -> hidden_dim
        self.stem = nn.Sequential(
            nn.Linear(3, config.hidden_dim, bias=False),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

        # Stack of URM transformer blocks (reused in recurrent loops)
        self.layers = nn.ModuleList(
            [GameURMBlock(config) for _ in range(config.num_layers)]
        )

        # Learnable initial hidden state for recurrent loops
        self.init_hidden = nn.Parameter(torch.zeros(1, self.N, config.hidden_dim))
        nn.init.trunc_normal_(self.init_hidden, std=0.02)

        # Action head: pool over sequence -> 4 actions
        self.action_head = nn.Linear(config.hidden_dim, self.NUM_ACTIONS, bias=True)

        # Value head: pool over sequence -> 1 value
        self.value_head = nn.Linear(config.hidden_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def directions(self) -> list[Direction]:
        """Direction order matching output logits."""
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with recurrent reasoning loops.

        Args:
            inputs: (batch, 48) tensor with cell features

        Returns:
            (action_logits, value_logit) tuple
        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        batch_size = inputs.shape[0]

        # Reshape: (batch, 48) -> (batch, 16, 3)
        x = inputs.view(batch_size, self.N, 3)

        # Project input to hidden dimension
        input_embeddings = self.stem(x)  # (batch, 16, hidden_dim)

        # Initialize hidden state (expand for batch)
        hidden_states = self.init_hidden.expand(batch_size, -1, -1).clone()

        # Recurrent reasoning loops with truncated backprop
        total_loops = self.config.num_loops
        truncated_loops = self.config.num_truncated_loops

        # Phase 1: Truncated loops (forward-only, no gradients)
        if truncated_loops > 0:
            with torch.no_grad():
                for _ in range(truncated_loops):
                    hidden_states = hidden_states + input_embeddings
                    for layer in self.layers:
                        hidden_states = layer(hidden_states)

        # Phase 2: Remaining loops with gradients
        for _ in range(total_loops - truncated_loops):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states)

        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Output heads
        action_logits = self.action_head(pooled)
        value_logit = self.value_head(pooled)

        return action_logits, value_logit


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

    print("=== Testing GameMLP ===")
    model = GameMLP(MLPConfig(hidden_dim=64))
    logits, value = model(stacked)
    print(f"Action logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Action logits:\n{logits}")

    print("\n=== Testing GameURM ===")
    urm_model = GameURM(
        GameURMConfig(hidden_dim=64, num_loops=4, num_truncated_loops=1)
    )
    urm_logits, urm_value = urm_model(stacked)
    print(f"Action logits shape: {urm_logits.shape}")
    print(f"Value shape: {urm_value.shape}")
    print(f"Action logits:\n{urm_logits}")

    # Compare parameter counts
    mlp_params = sum(p.numel() for p in model.parameters())
    urm_params = sum(p.numel() for p in urm_model.parameters())
    print(f"\n=== Parameter Counts ===")
    print(f"GameMLP: {mlp_params:,} parameters")
    print(f"GameURM: {urm_params:,} parameters")
