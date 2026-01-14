/**
 * UI Controller for 2048 RL Demo
 *
 * Handles:
 * - Replay mode: Step through recorded best game
 * - Live mode: Watch the AI play in real-time
 */

// Color scheme matching classic 2048
const TILE_COLORS = {
    0: '#cdc1b4',
    2: '#eee4da',
    4: '#ede0c8',
    8: '#f2b179',
    16: '#f59563',
    32: '#f67c5f',
    64: '#f65e3b',
    128: '#edcf72',
    256: '#edcc61',
    512: '#edc850',
    1024: '#edc53f',
    2048: '#edc22e',
    4096: '#3c3a32',
    8192: '#3c3a32',
};

const TEXT_COLORS = {
    0: '#cdc1b4',
    2: '#776e65',
    4: '#776e65',
    8: '#f9f6f2',
    16: '#f9f6f2',
    32: '#f9f6f2',
    64: '#f9f6f2',
    128: '#f9f6f2',
    256: '#f9f6f2',
    512: '#f9f6f2',
    1024: '#f9f6f2',
    2048: '#f9f6f2',
    4096: '#f9f6f2',
    8192: '#f9f6f2',
};

class UIController {
    constructor() {
        this.mode = 'replay'; // 'replay' or 'live'
        this.model = new GameModel();
        this.game = new Game2048();

        // Replay state
        this.replayData = null;
        this.replayIndex = 0;
        this.replayInterval = null;
        this.replaySpeed = 500;

        // Live mode state
        this.liveInterval = null;
        this.liveSpeed = 200;
        this.isLivePlaying = false;

        this.initializeBoard();
        this.bindEvents();
        this.loadAssets();
    }

    /**
     * Initialize the game board grid.
     */
    initializeBoard() {
        const board = document.getElementById('game-board');
        board.innerHTML = '';

        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.id = `cell-${i}-${j}`;
                board.appendChild(cell);
            }
        }
    }

    /**
     * Render a grid state to the board.
     */
    renderBoard(grid) {
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                const cell = document.getElementById(`cell-${i}-${j}`);
                const value = grid[i][j];

                cell.textContent = value > 0 ? value : '';
                cell.style.backgroundColor = TILE_COLORS[value] || TILE_COLORS[8192];
                cell.style.color = TEXT_COLORS[value] || TEXT_COLORS[8192];

                // Adjust font size for large numbers
                if (value >= 1000) {
                    cell.style.fontSize = '28px';
                } else if (value >= 100) {
                    cell.style.fontSize = '36px';
                } else {
                    cell.style.fontSize = '48px';
                }
            }
        }
    }

    /**
     * Update the score display.
     */
    updateScore(score, moveCount) {
        document.getElementById('current-score').textContent = score;
        document.getElementById('move-count').textContent = moveCount;
    }

    /**
     * Update the action display.
     */
    updateAction(action, pointsEarned = 0) {
        const actionSymbols = { 'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→' };
        document.getElementById('last-action').textContent = actionSymbols[action] || '-';

        const pointsEl = document.getElementById('points-earned');
        if (pointsEarned > 0) {
            pointsEl.textContent = `+${pointsEarned}`;
            pointsEl.classList.add('show');
            setTimeout(() => pointsEl.classList.remove('show'), 500);
        } else {
            pointsEl.textContent = '';
        }
    }

    /**
     * Update action probability bars (live mode).
     */
    updateProbabilities(probs) {
        const directions = ['up', 'down', 'left', 'right'];
        for (let i = 0; i < 4; i++) {
            const prob = probs[i] * 100;
            document.getElementById(`prob-${directions[i]}`).style.width = `${prob}%`;
            document.getElementById(`prob-${directions[i]}-val`).textContent = `${prob.toFixed(1)}%`;
        }
    }

    /**
     * Bind UI event handlers.
     */
    bindEvents() {
        // Mode switching
        document.getElementById('replay-mode-btn').addEventListener('click', () => this.setMode('replay'));
        document.getElementById('live-mode-btn').addEventListener('click', () => this.setMode('live'));

        // Replay controls
        document.getElementById('replay-start').addEventListener('click', () => this.replayGoToStart());
        document.getElementById('replay-prev').addEventListener('click', () => this.replayPrev());
        document.getElementById('replay-play').addEventListener('click', () => this.replayTogglePlay());
        document.getElementById('replay-next').addEventListener('click', () => this.replayNext());
        document.getElementById('replay-end').addEventListener('click', () => this.replayGoToEnd());

        document.getElementById('replay-speed').addEventListener('input', (e) => {
            this.replaySpeed = parseInt(e.target.value);
            document.getElementById('replay-speed-value').textContent = `${this.replaySpeed}ms`;
        });

        // Live controls
        document.getElementById('live-start').addEventListener('click', () => this.liveStart());
        document.getElementById('live-stop').addEventListener('click', () => this.liveStop());
        document.getElementById('live-reset').addEventListener('click', () => this.liveReset());

        document.getElementById('live-speed').addEventListener('input', (e) => {
            this.liveSpeed = parseInt(e.target.value);
            document.getElementById('live-speed-value').textContent = `${this.liveSpeed}ms`;
        });
    }

    /**
     * Load best game data and model.
     */
    async loadAssets() {
        console.log('Loading assets...');

        // Load best game replay data
        try {
            console.log('Fetching best_game.json...');
            const response = await fetch('data/best_game.json');
            console.log('Response status:', response.status);
            if (response.ok) {
                this.replayData = await response.json();
                console.log('Loaded replay data:', this.replayData.score, 'points,', this.replayData.total_steps, 'moves');
                document.getElementById('best-game-score').textContent = this.replayData.score;
                document.getElementById('best-game-moves').textContent = this.replayData.total_steps;

                // Show initial state
                if (this.replayData.moves && this.replayData.moves.length > 0) {
                    console.log('Rendering initial board state...');
                    this.renderBoard(this.replayData.moves[0].state_before);
                    this.updateScore(0, 0);
                }
            } else {
                console.error('Best game data not found, status:', response.status);
            }
        } catch (error) {
            console.error('Could not load best game:', error);
        }

        // Load model
        console.log('Loading ONNX model...');
        const modelLoaded = await this.model.load('data/model.onnx');
        console.log('Model loaded:', modelLoaded);
        if (modelLoaded) {
            document.getElementById('model-status').textContent = 'Ready';
            document.getElementById('model-status').style.color = '#4CAF50';
        } else {
            document.getElementById('model-status').textContent = 'Failed to load';
            document.getElementById('model-status').style.color = '#f44336';
            console.error('Model load error:', this.model.loadError);
        }
    }

    /**
     * Switch between replay and live modes.
     */
    setMode(mode) {
        this.mode = mode;

        // Update button states
        document.getElementById('replay-mode-btn').classList.toggle('active', mode === 'replay');
        document.getElementById('live-mode-btn').classList.toggle('active', mode === 'live');

        // Show/hide control panels
        document.getElementById('replay-controls').classList.toggle('hidden', mode !== 'replay');
        document.getElementById('live-controls').classList.toggle('hidden', mode !== 'live');

        // Stop any running playback
        this.replayPause();
        this.liveStop();

        // Initialize the mode
        if (mode === 'replay') {
            this.replayGoToStart();
        } else {
            this.liveReset();
        }
    }

    // ==================== REPLAY MODE ====================

    replayGoToStart() {
        this.replayIndex = 0;
        this.replayShowFrame();
    }

    replayGoToEnd() {
        if (!this.replayData) return;
        this.replayIndex = this.replayData.moves.length - 1;
        this.replayShowFrame();
    }

    replayPrev() {
        if (this.replayIndex > 0) {
            this.replayIndex--;
            this.replayShowFrame();
        }
    }

    replayNext() {
        if (!this.replayData) return;
        if (this.replayIndex < this.replayData.moves.length - 1) {
            this.replayIndex++;
            this.replayShowFrame();
        } else {
            this.replayPause();
        }
    }

    replayTogglePlay() {
        if (this.replayInterval) {
            this.replayPause();
        } else {
            this.replayPlay();
        }
    }

    replayPlay() {
        if (!this.replayData) return;

        document.getElementById('replay-play').textContent = '⏸';
        this.replayInterval = setInterval(() => {
            this.replayNext();
        }, this.replaySpeed);
    }

    replayPause() {
        if (this.replayInterval) {
            clearInterval(this.replayInterval);
            this.replayInterval = null;
        }
        document.getElementById('replay-play').textContent = '▶';
    }

    replayShowFrame() {
        if (!this.replayData || !this.replayData.moves) return;

        const move = this.replayData.moves[this.replayIndex];
        if (!move) return;

        // Show state after move (or state_before for first frame)
        const grid = this.replayIndex === 0 ? move.state_before : move.state_after;
        this.renderBoard(grid);

        // Calculate cumulative score up to this point
        let cumulativeScore = 0;
        for (let i = 0; i <= this.replayIndex; i++) {
            cumulativeScore += this.replayData.moves[i].points_earned || 0;
        }

        this.updateScore(cumulativeScore, this.replayIndex + 1);
        this.updateAction(move.action, move.points_earned);
    }

    // ==================== LIVE MODE ====================

    liveReset() {
        this.liveStop();
        this.game.reset();
        this.renderBoard(this.game.grid);
        this.updateScore(0, 0);
        this.updateAction('-');
        this.updateProbabilities([0.25, 0.25, 0.25, 0.25]);
    }

    async liveStart() {
        if (!this.model.isLoaded) {
            alert('Model not loaded yet. Please wait or check console for errors.');
            return;
        }

        this.isLivePlaying = true;
        document.getElementById('live-start').disabled = true;
        document.getElementById('live-stop').disabled = false;

        this.liveStep();
    }

    liveStop() {
        this.isLivePlaying = false;
        if (this.liveInterval) {
            clearTimeout(this.liveInterval);
            this.liveInterval = null;
        }
        document.getElementById('live-start').disabled = false;
        document.getElementById('live-stop').disabled = true;
    }

    async liveStep() {
        if (!this.isLivePlaying || !this.game.hasNextStep()) {
            this.liveStop();
            return;
        }

        try {
            // Get model prediction
            const { action, probs, value } = await this.model.selectMove(this.game, false);

            // Update probability display
            this.updateProbabilities(probs);

            // Execute the move
            const { pointsEarned, done } = this.game.move(action);

            // Update display
            this.renderBoard(this.game.grid);
            this.updateScore(this.game.score, this.game.score > 0 ? Math.floor(Math.log2(this.game.score)) : 0);
            this.updateAction(action, pointsEarned);

            // Calculate move count (approximate)
            const moveCount = document.getElementById('move-count');
            moveCount.textContent = parseInt(moveCount.textContent || 0) + 1;

            if (done) {
                this.liveStop();
                return;
            }

            // Schedule next step
            this.liveInterval = setTimeout(() => this.liveStep(), this.liveSpeed);

        } catch (error) {
            console.error('Error during live step:', error);
            this.liveStop();
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.ui = new UIController();
});
