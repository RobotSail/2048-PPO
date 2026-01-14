/**
 * ONNX Model Wrapper for 2048 RL Agent
 *
 * Loads and runs the trained model using ONNX Runtime Web.
 */

class GameModel {
    constructor() {
        this.session = null;
        this.isLoaded = false;
        this.loadError = null;
    }

    /**
     * Load the ONNX model from the given path.
     */
    async load(modelPath = 'data/model.onnx') {
        try {
            this.session = await ort.InferenceSession.create(modelPath);
            this.isLoaded = true;
            console.log('Model loaded successfully');
            return true;
        } catch (error) {
            this.loadError = error;
            console.error('Failed to load model:', error);
            return false;
        }
    }

    /**
     * Run inference on the given board state.
     *
     * @param {Float32Array} boardState - 48-element feature vector from Game2048.toModelFormat()
     * @returns {Object} - { actionLogits: Float32Array(4), value: number }
     */
    async predict(boardState) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', boardState, [1, 48]);

        // Run inference
        const feeds = { 'board_state': inputTensor };
        const results = await this.session.run(feeds);

        // Extract outputs
        const actionLogits = results['action_logits'].data;
        const value = results['value'].data[0];

        return {
            actionLogits: new Float32Array(actionLogits),
            value: value
        };
    }

    /**
     * Get action probabilities from logits, masking invalid moves.
     *
     * @param {Float32Array} logits - Raw action logits from model
     * @param {Array<string>} validMoves - List of valid direction strings
     * @returns {Object} - { probs: Float32Array(4), selectedAction: number }
     */
    getActionProbabilities(logits, validMoves) {
        // Create mask for invalid moves
        const maskedLogits = new Float32Array(4);
        const directionMap = { 'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3 };

        for (let i = 0; i < 4; i++) {
            const direction = DIRECTIONS[i];
            if (validMoves.includes(direction)) {
                maskedLogits[i] = logits[i];
            } else {
                maskedLogits[i] = -Infinity;
            }
        }

        // Compute softmax
        const probs = this.softmax(maskedLogits);

        return probs;
    }

    /**
     * Sample an action from the probability distribution.
     *
     * @param {Float32Array} probs - Action probabilities
     * @returns {number} - Selected action index (0-3)
     */
    sampleAction(probs) {
        const rand = Math.random();
        let cumSum = 0;

        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) {
                return i;
            }
        }

        // Fallback to last valid action
        return probs.length - 1;
    }

    /**
     * Get the greedy (highest probability) action.
     *
     * @param {Float32Array} probs - Action probabilities
     * @returns {number} - Selected action index (0-3)
     */
    greedyAction(probs) {
        let maxIdx = 0;
        let maxProb = probs[0];

        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /**
     * Compute softmax over the given logits.
     */
    softmax(logits) {
        // Find max for numerical stability
        let maxLogit = -Infinity;
        for (const l of logits) {
            if (l > maxLogit) maxLogit = l;
        }

        // Compute exp and sum
        const exps = new Float32Array(logits.length);
        let sum = 0;

        for (let i = 0; i < logits.length; i++) {
            if (logits[i] === -Infinity) {
                exps[i] = 0;
            } else {
                exps[i] = Math.exp(logits[i] - maxLogit);
                sum += exps[i];
            }
        }

        // Normalize
        const probs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            probs[i] = sum > 0 ? exps[i] / sum : 0;
        }

        return probs;
    }

    /**
     * Select the best move for the given game state.
     *
     * @param {Game2048} game - Current game instance
     * @param {boolean} greedy - If true, always select highest probability action
     * @returns {Object} - { action: string, probs: Float32Array, value: number }
     */
    async selectMove(game, greedy = false) {
        const boardState = game.toModelFormat();
        const { actionLogits, value } = await this.predict(boardState);

        const validMoves = game.getValidDirections();
        const probs = this.getActionProbabilities(actionLogits, validMoves);

        const actionIdx = greedy ? this.greedyAction(probs) : this.sampleAction(probs);
        const action = DIRECTIONS[actionIdx];

        return {
            action: action,
            probs: probs,
            value: value
        };
    }
}

// Export for use in other modules
window.GameModel = GameModel;
