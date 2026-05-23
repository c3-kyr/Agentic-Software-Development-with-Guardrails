/**
 * Parameter Optimizer (REINFORCE / Finite-Difference Policy Gradient)
 *
 * Optimizes the continuous parameters [w₁, w₂, w₃, λ] using
 * the DQN agent's episode return as the objective signal.
 *
 * Uses finite-difference gradient estimation:
 *   ∂J/∂θᵢ ≈ (J(θ + δeᵢ) − J(θ − δeᵢ)) / (2δ)
 *
 * Applies softmax constraint to weights (must sum to 1).
 */

export class ParamOptimizer {
    /**
     * @param {Object} config
     * @param {number[]} config.initialWeights - Starting [w1, w2, w3]
     * @param {number} config.initialLambda - Starting λ (human cost weight)
     * @param {number} config.lr - Learning rate for parameter updates (default 0.01)
     * @param {number} config.delta - Perturbation size for gradient estimation (default 0.05)
     * @param {number} config.momentum - Momentum coefficient (default 0.9)
     */
    constructor(config = {}) {
        // Raw (pre-softmax) weight parameters
        this.weightLogits = [...(config.initialWeights || [0.4, 0.35, 0.25])];
        this.lambda = config.initialLambda ?? 0.5;

        this.lr = config.lr || 0.01;
        this.delta = config.delta || 0.05;
        this.momentum = config.momentum || 0.9;

        // Momentum accumulators
        this.velocityWeights = [0, 0, 0];
        this.velocityLambda = 0;

        // History for tracking
        this.history = [];
        this.updateCount = 0;
    }

    /**
     * Get the current normalized weights (softmax of logits)
     * @returns {number[]}
     */
    getWeights() {
        return this._softmax(this.weightLogits);
    }

    /**
     * Get the current lambda
     * @returns {number}
     */
    getLambda() {
        return Math.max(0.01, Math.min(1.0, this.lambda));
    }

    /**
     * Get all current parameters
     * @returns {{ weights: number[], lambda: number }}
     */
    getParams() {
        return {
            weights: this.getWeights(),
            lambda: this.getLambda(),
        };
    }

    /**
     * Update parameters using finite-difference gradient estimation
     *
     * @param {Function} evaluateFn - async (weights, lambda) => episodeReturn
     *   This function runs one full episode with the given parameters
     *   and returns the total reward.
     * @returns {Promise<{ params: Object, gradient: Object, improvement: number }>}
     */
    async update(evaluateFn) {
        const currentWeights = this.getWeights();
        const currentLambda = this.getLambda();

        // Evaluate current parameters
        const baseReturn = await evaluateFn(currentWeights, currentLambda);

        // Estimate gradient for each weight logit
        const weightGrads = [];
        for (let i = 0; i < 3; i++) {
            // Perturb positively
            const logitsPlus = [...this.weightLogits];
            logitsPlus[i] += this.delta;
            const weightsPlus = this._softmax(logitsPlus);
            const returnPlus = await evaluateFn(weightsPlus, currentLambda);

            // Perturb negatively
            const logitsMinus = [...this.weightLogits];
            logitsMinus[i] -= this.delta;
            const weightsMinus = this._softmax(logitsMinus);
            const returnMinus = await evaluateFn(weightsMinus, currentLambda);

            // Central difference
            weightGrads.push((returnPlus - returnMinus) / (2 * this.delta));
        }

        // Estimate gradient for lambda
        const returnLambdaPlus = await evaluateFn(
            currentWeights,
            Math.min(1.0, currentLambda + this.delta)
        );
        const returnLambdaMinus = await evaluateFn(
            currentWeights,
            Math.max(0.01, currentLambda - this.delta)
        );
        const lambdaGrad = (returnLambdaPlus - returnLambdaMinus) / (2 * this.delta);

        // Apply momentum + gradient ascent (maximizing return)
        for (let i = 0; i < 3; i++) {
            this.velocityWeights[i] =
                this.momentum * this.velocityWeights[i] + this.lr * weightGrads[i];
            this.weightLogits[i] += this.velocityWeights[i];
        }

        this.velocityLambda = this.momentum * this.velocityLambda + this.lr * lambdaGrad;
        this.lambda += this.velocityLambda;
        this.lambda = Math.max(0.01, Math.min(1.0, this.lambda));

        this.updateCount++;

        const newParams = this.getParams();
        const improvement = (await evaluateFn(newParams.weights, newParams.lambda)) - baseReturn;

        const record = {
            step: this.updateCount,
            params: { ...newParams },
            baseReturn,
            gradient: { weights: weightGrads, lambda: lambdaGrad },
            improvement,
        };
        this.history.push(record);

        return record;
    }

    /**
     * Get optimization history
     */
    getHistory() {
        return this.history;
    }

    /**
     * Reset optimizer state
     */
    reset(initialWeights, initialLambda) {
        this.weightLogits = [...(initialWeights || [0.4, 0.35, 0.25])];
        this.lambda = initialLambda ?? 0.5;
        this.velocityWeights = [0, 0, 0];
        this.velocityLambda = 0;
        this.history = [];
        this.updateCount = 0;
    }

    /**
     * Softmax: ensures weights sum to 1 and are positive
     * @param {number[]} logits
     * @returns {number[]}
     */
    _softmax(logits) {
        const maxLogit = Math.max(...logits);
        const exps = logits.map((l) => Math.exp(l - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map((e) => e / sumExps);
    }
}
