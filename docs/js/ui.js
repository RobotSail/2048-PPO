/**
 * UI Controller for 2048 RL Demo
 *
 * Handles:
 * - Replay mode: Step through recorded best game
 * - Live mode: Watch the AI play in real-time
 */

// Speed presets: slider value (1-10) -> milliseconds delay
// Higher slider value = faster speed
const SPEED_PRESETS = {
    1: { ms: 2000, label: '0.5x' },
    2: { ms: 1000, label: '1x' },
    3: { ms: 500, label: '2x' },
    4: { ms: 250, label: '4x' },
    5: { ms: 150, label: '6x' },
    6: { ms: 100, label: '10x' },
    7: { ms: 50, label: '20x' },
    8: { ms: 25, label: '40x' },
    9: { ms: 10, label: '100x' },
    10: { ms: 5, label: '200x' },
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
        this.replaySpeedLevel = 5; // Default to 6x speed
        this.replaySpeed = SPEED_PRESETS[5].ms;

        // Live mode state
        this.liveInterval = null;
        this.liveSpeedLevel = 6; // Default to 10x speed
        this.liveSpeed = SPEED_PRESETS[6].ms;
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
     * @param {number[][]} grid - Grid of values (already 2^exp for replay, exponents for live)
     * @param {boolean} isExponent - If true, values are exponents and need conversion
     */
    renderBoard(grid, isExponent = false) {
        for (let i = 0; i < GRID_SIZE; i++) {
            for (let j = 0; j < GRID_SIZE; j++) {
                const cell = document.getElementById(`cell-${i}-${j}`);
                const rawValue = grid[i][j];

                // Convert exponent to actual value if needed
                const displayValue = isExponent && rawValue > 0 ? Math.pow(2, rawValue) : rawValue;

                // Reset classes and set tile class based on display value
                cell.className = 'cell';
                if (displayValue > 0) {
                    cell.classList.add(`tile-${displayValue}`);
                }

                cell.textContent = displayValue > 0 ? displayValue : '';

                // Adjust font size for large numbers
                if (displayValue >= 1000) {
                    cell.style.fontSize = '24px';
                } else if (displayValue >= 100) {
                    cell.style.fontSize = '32px';
                } else {
                    cell.style.fontSize = '';
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
     * Update the step slider and label.
     */
    updateStepSlider() {
        if (!this.replayData || !this.replayData.moves) return;

        const maxStep = this.replayData.moves.length;
        const slider = document.getElementById('replay-step-slider');
        const label = document.getElementById('replay-step-label');

        slider.max = maxStep;
        slider.value = this.replayIndex;
        label.textContent = `Step ${this.replayIndex} / ${maxStep}`;
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

        // Replay step slider (draggable progress bar)
        document.getElementById('replay-step-slider').addEventListener('input', (e) => {
            this.replayPause();
            this.replayIndex = parseInt(e.target.value);
            this.replayShowFrame();
        });

        // Replay speed slider (1-10 scale, higher = faster)
        document.getElementById('replay-speed').addEventListener('input', (e) => {
            this.replaySpeedLevel = parseInt(e.target.value);
            const preset = SPEED_PRESETS[this.replaySpeedLevel];
            this.replaySpeed = preset.ms;
            document.getElementById('replay-speed-value').textContent = preset.label;

            // If playing, restart with new speed
            if (this.replayInterval) {
                this.replayPause();
                this.replayPlay();
            }
        });

        // Live controls
        document.getElementById('live-start').addEventListener('click', () => this.liveStart());
        document.getElementById('live-stop').addEventListener('click', () => this.liveStop());
        document.getElementById('live-reset').addEventListener('click', () => this.liveReset());

        // Live speed slider (1-10 scale, higher = faster)
        document.getElementById('live-speed').addEventListener('input', (e) => {
            this.liveSpeedLevel = parseInt(e.target.value);
            const preset = SPEED_PRESETS[this.liveSpeedLevel];
            this.liveSpeed = preset.ms;
            document.getElementById('live-speed-value').textContent = preset.label;
        });

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (this.mode === 'replay' && this.replayData) {
                switch (e.key) {
                    case 'ArrowLeft':
                        this.replayPrev();
                        break;
                    case 'ArrowRight':
                        this.replayNext();
                        break;
                    case ' ':
                        e.preventDefault();
                        this.replayTogglePlay();
                        break;
                    case 'Home':
                        this.replayGoToStart();
                        break;
                    case 'End':
                        this.replayGoToEnd();
                        break;
                }
            }
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

                // Initialize step slider
                this.updateStepSlider();

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
            document.getElementById('model-status').style.color = '#3fb950';
        } else {
            document.getElementById('model-status').textContent = 'Failed to load';
            document.getElementById('model-status').style.color = '#f85149';
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
        this.replayIndex = this.replayData.moves.length;
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
        if (this.replayIndex < this.replayData.moves.length) {
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

        // If at end, restart from beginning
        if (this.replayIndex >= this.replayData.moves.length) {
            this.replayIndex = 0;
        }

        document.getElementById('replay-play').textContent = '⏸';
        document.getElementById('replay-play').title = 'Pause';
        this.replayInterval = setInterval(() => {
            if (this.replayIndex < this.replayData.moves.length) {
                this.replayIndex++;
                this.replayShowFrame();
            } else {
                this.replayPause();
            }
        }, this.replaySpeed);
    }

    replayPause() {
        if (this.replayInterval) {
            clearInterval(this.replayInterval);
            this.replayInterval = null;
        }
        document.getElementById('replay-play').textContent = '▶';
        document.getElementById('replay-play').title = 'Play';
    }

    replayShowFrame() {
        if (!this.replayData || !this.replayData.moves) return;

        const moves = this.replayData.moves;

        // Update step slider
        this.updateStepSlider();

        if (this.replayIndex === 0) {
            // Show initial state (before first move)
            if (moves[0] && moves[0].state_before) {
                this.renderBoard(moves[0].state_before);
            }
            this.updateScore(0, 0);
            this.updateAction('-');
        } else {
            // Show state after move at replayIndex - 1
            const move = moves[this.replayIndex - 1];
            if (move) {
                this.renderBoard(move.state_after);

                // Calculate cumulative score up to this point
                let cumulativeScore = 0;
                for (let i = 0; i < this.replayIndex; i++) {
                    cumulativeScore += moves[i].points_earned || 0;
                }

                this.updateScore(cumulativeScore, this.replayIndex);
                this.updateAction(move.action, move.points_earned);
            }
        }
    }

    // ==================== LIVE MODE ====================

    liveReset() {
        this.liveStop();
        this.game.reset();
        this.renderBoard(this.game.grid, true);  // Live mode uses exponents
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
            this.renderBoard(this.game.grid, true);  // Live mode uses exponents
            this.updateScore(this.game.score, this.game.moveCount);
            this.updateAction(action, pointsEarned);

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
