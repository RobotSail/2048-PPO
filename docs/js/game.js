/**
 * 2048 Game Logic - JavaScript port of game.py
 *
 * Grid representation: 4x4 array of exponents
 * - 0 = empty cell
 * - 1 = 2 (2^1)
 * - 2 = 4 (2^2)
 * - etc.
 */

const GRID_SIZE = 4;
const DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT'];

class Game2048 {
    constructor(initialGrid = null) {
        if (initialGrid) {
            this.grid = initialGrid.map(row => [...row]);
        } else {
            this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
        }
        this.score = 0;
        this.moveCount = 0;
    }

    /**
     * Reset the game to initial state with 2 random tiles.
     */
    reset() {
        this.grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
        this.score = 0;
        this.moveCount = 0;
        this.addRandomTile();
        this.addRandomTile();
        return this.grid;
    }

    /**
     * Add a random tile (90% chance of 2, 10% chance of 4) to an empty cell.
     */
    addRandomTile() {
        const emptyCells = [];
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                if (this.grid[i][j] === 0) {
                    emptyCells.push([i, j]);
                }
            }
        }

        if (emptyCells.length === 0) return false;

        const [row, col] = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        this.grid[row][col] = Math.random() < 0.9 ? 1 : 2; // exponent 1 = 2, exponent 2 = 4
        return true;
    }

    /**
     * Check if a move in the given direction is valid.
     */
    canMove(direction) {
        return this.canMoveInDirection(direction) || this.canMergeInDirection(direction);
    }

    /**
     * Check if tiles can slide in the given direction (has empty space).
     */
    canMoveInDirection(direction) {
        let workingGrid = this.grid.map(row => [...row]);

        // Transpose for vertical moves
        if (direction === 'UP' || direction === 'DOWN') {
            workingGrid = this.transpose(workingGrid);
        }

        // Reverse rows for right/down moves
        if (direction === 'LEFT' || direction === 'UP') {
            workingGrid = workingGrid.map(row => [...row].reverse());
        }

        for (const row of workingGrid) {
            let foundValue = false;
            for (const cell of row) {
                if (cell > 0) foundValue = true;
                if (foundValue && cell === 0) return true;
            }
        }
        return false;
    }

    /**
     * Check if tiles can merge in the given direction.
     */
    canMergeInDirection(direction) {
        let workingGrid = this.grid.map(row => [...row]);

        // Transpose for vertical moves
        if (direction === 'UP' || direction === 'DOWN') {
            workingGrid = this.transpose(workingGrid);
        }

        // Reverse rows for right/down moves
        if (direction === 'LEFT' || direction === 'UP') {
            workingGrid = workingGrid.map(row => [...row].reverse());
        }

        for (const row of workingGrid) {
            for (let i = 0; i < row.length - 1; i++) {
                if (row[i] !== 0 && row[i] === row[i + 1]) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Get list of valid directions.
     */
    getValidDirections() {
        return DIRECTIONS.filter(dir => this.canMove(dir));
    }

    /**
     * Check if the game has any valid moves left.
     */
    hasNextStep() {
        return DIRECTIONS.some(dir => this.canMove(dir));
    }

    /**
     * Transpose a grid (swap rows and columns).
     */
    transpose(grid) {
        return grid[0].map((_, i) => grid.map(row => row[i]));
    }

    /**
     * Merge and shift a row to the left.
     * Returns [newRow, scoreGained, maxTileCreated]
     */
    mergeAndShiftLeft(row) {
        const nonZero = row.filter(x => x !== 0);
        const merged = [];
        let score = 0;
        let maxTileCreated = 0;

        let i = 0;
        while (i < nonZero.length) {
            if (i + 1 < nonZero.length && nonZero[i] === nonZero[i + 1]) {
                const newExp = nonZero[i] + 1;
                merged.push(newExp);
                score += Math.pow(2, newExp);
                maxTileCreated = Math.max(maxTileCreated, newExp);
                i += 2;
            } else {
                merged.push(nonZero[i]);
                i += 1;
            }
        }

        // Pad with zeros
        while (merged.length < GRID_SIZE) {
            merged.push(0);
        }

        return [merged, score, maxTileCreated];
    }

    /**
     * Execute a move in the given direction.
     * Returns { grid, pointsEarned, done }
     */
    move(direction) {
        if (!this.canMove(direction)) {
            return { grid: this.grid, pointsEarned: 0, done: !this.hasNextStep() };
        }

        let workingGrid = this.grid.map(row => [...row]);
        let needsTranspose = false;
        let needsReverse = false;

        // Transform grid for the move
        if (direction === 'UP' || direction === 'DOWN') {
            workingGrid = this.transpose(workingGrid);
            needsTranspose = true;
        }

        if (direction === 'RIGHT' || direction === 'DOWN') {
            workingGrid = workingGrid.map(row => [...row].reverse());
            needsReverse = true;
        }

        // Merge all rows
        let totalScore = 0;
        const results = workingGrid.map(row => {
            const [newRow, score, _] = this.mergeAndShiftLeft(row);
            totalScore += score;
            return newRow;
        });

        // Reverse transform
        workingGrid = results;
        if (needsReverse) {
            workingGrid = workingGrid.map(row => [...row].reverse());
        }
        if (needsTranspose) {
            workingGrid = this.transpose(workingGrid);
        }

        this.grid = workingGrid;
        this.score += totalScore;
        this.moveCount++;

        // Add new random tile
        this.addRandomTile();

        return {
            grid: this.grid,
            pointsEarned: totalScore,
            done: !this.hasNextStep()
        };
    }

    /**
     * Convert grid to model input format (48 features).
     * Matches Python's to_model_format():
     * - 16 cells × 3 features (value, row_idx/3, col_idx/3)
     */
    toModelFormat() {
        const features = [];
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                features.push(this.grid[i][j]);  // exponent value
                features.push(i / 3);             // row index normalized
                features.push(j / 3);             // col index normalized
            }
        }
        return new Float32Array(features);
    }

    /**
     * Get the current score.
     */
    getScore() {
        return this.score;
    }

    /**
     * Get the maximum tile value on the board.
     */
    getMaxTile() {
        let maxExp = 0;
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                maxExp = Math.max(maxExp, this.grid[i][j]);
            }
        }
        return maxExp > 0 ? Math.pow(2, maxExp) : 0;
    }

    /**
     * Create a deep copy of the current game state.
     */
    clone() {
        const copy = new Game2048(this.grid);
        copy.score = this.score;
        return copy;
    }
}

// Export for use in other modules
window.Game2048 = Game2048;
window.DIRECTIONS = DIRECTIONS;
window.GRID_SIZE = GRID_SIZE;
