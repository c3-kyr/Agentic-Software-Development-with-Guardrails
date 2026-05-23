/**
 * Token-Cost Grounded Reward Model
 *
 * Derives RL reward values from the estimated token economics of each
 * decision outcome, replacing arbitrary hand-tuned constants.
 *
 * ────────────────────────────────────────────────────────────────────
 * Mathematical Foundation:
 *
 *   T_base(I,P)  = T_min + (T_max − T_min) · φ(I,P)
 *                   where φ(I,P) = (I + P) / 2
 *
 *   κ(R)         = 1 + α · R        (redo multiplier)
 *
 *   Rewards:
 *     R_TP =  +T_base            / T_norm     (tokens saved)
 *     R_TN =  +T_base · η       / T_norm     (efficient automation)
 *     R_FP =  −T_review          / T_norm     (review overhead)
 *     R_FN =  −T_base · (1 + κ) / T_norm     (wasted + redo cost)
 *
 *   T_norm = T_max · (2 + α)   (normalization ceiling)
 *
 * See: token_cost_reward_derivation.md for full proof.
 * ────────────────────────────────────────────────────────────────────
 */

/**
 * Default parameters for the Token Cost Model.
 * These can be calibrated from real sprint data over time.
 */
export const DEFAULT_TOKEN_PARAMS = {
    T_min: 500,       // Minimal task: config change, feature flag (~500 tokens)
    T_max: 10000,     // Maximum task: full architectural refactor (~10K tokens)
    T_review: 800,    // Human review overhead: reading context + responding (~800 tokens equiv.)
    alpha: 2.5,       // Redo overhead ceiling: fixing costs up to 3.5× the original
    eta: 0.6,         // Efficiency discount: correct automation is 0.6× the value of crisis aversion
};

/**
 * Token Cost Model — computes mathematically grounded rewards
 * for each decision outcome in the HITL RL environment.
 */
export class TokenCostModel {
    /**
     * @param {Object} params - Model parameters (uses defaults if omitted)
     * @param {number} params.T_min - Minimum token cost for a task
     * @param {number} params.T_max - Maximum token cost for a task
     * @param {number} params.T_review - Fixed token cost of human review
     * @param {number} params.alpha - Maximum redo overhead multiplier
     * @param {number} params.eta - Efficiency discount factor for TN
     */
    constructor(params = {}) {
        this.params = { ...DEFAULT_TOKEN_PARAMS, ...params };

        // Precompute normalization constant: T_norm = T_max · (2 + α)
        this.T_norm = this.params.T_max * (2 + this.params.alpha);

        // Calibration tracking (for self-calibration from real data)
        this._calibrationBuffer = [];
        this._calibrationWindowSize = 100;
    }

    /**
     * Compute the base token complexity for a task.
     *
     *   T_base(I, P) = T_min + (T_max − T_min) · φ(I, P)
     *   φ(I, P) = (I + P) / 2
     *
     * @param {number} impact - Impact factor ∈ [0, 1]
     * @param {number} propagation - Propagation factor ∈ [0, 1]
     * @returns {number} Estimated base token cost
     */
    computeBaseTokenCost(impact, propagation) {
        const I = Math.max(0, Math.min(1, impact));
        const P = Math.max(0, Math.min(1, propagation));

        const phi = (I + P) / 2;
        return this.params.T_min + (this.params.T_max - this.params.T_min) * phi;
    }

    /**
     * Compute the redo multiplier.
     *
     *   κ(R) = 1 + α · R
     *
     * @param {number} irreversibility - Irreversibility factor ∈ [0, 1]
     * @returns {number} Redo cost multiplier
     */
    computeRedoMultiplier(irreversibility) {
        const R = Math.max(0, Math.min(1, irreversibility));
        return 1 + this.params.alpha * R;
    }

    /**
     * Compute all four reward values for a given scenario.
     *
     * @param {number} impact - Impact factor ∈ [0, 1]
     * @param {number} irreversibility - Irreversibility factor ∈ [0, 1]
     * @param {number} propagation - Propagation factor ∈ [0, 1]
     * @returns {{ TP: number, TN: number, FP: number, FN: number, meta: Object }}
     */
    computeRewards(impact, irreversibility, propagation) {
        const T_base = this.computeBaseTokenCost(impact, propagation);
        const kappa = this.computeRedoMultiplier(irreversibility);

        const TP = T_base / this.T_norm;
        const TN = (T_base * this.params.eta) / this.T_norm;
        const FP = -(this.params.T_review / this.T_norm);
        const FN = -(T_base * (1 + kappa)) / this.T_norm;

        return {
            TP,
            TN,
            FP,
            FN,
            meta: {
                T_base,
                kappa,
                T_norm: this.T_norm,
                T_review: this.params.T_review,
                // Diagnostic ratios
                FN_to_FP_ratio: Math.abs(FN / FP),
                TP_to_TN_ratio: TP / TN,
            },
        };
    }

    /**
     * Compute the reward for a specific outcome.
     *
     * @param {string} classification - 'TP' | 'TN' | 'FP' | 'FN'
     * @param {number} impact
     * @param {number} irreversibility
     * @param {number} propagation
     * @returns {number} Reward value
     */
    getReward(classification, impact, irreversibility, propagation) {
        const rewards = this.computeRewards(impact, irreversibility, propagation);
        return rewards[classification] ?? 0;
    }

    /**
     * Record an observed token cost for self-calibration.
     * Call this after a real sprint task completes.
     *
     * @param {Object} observation
     * @param {number} observation.tokensUsed - Actual tokens consumed
     * @param {number} observation.impact - Task impact factor
     * @param {number} observation.propagation - Task propagation factor
     * @param {string} observation.category - Task category for segmented calibration
     */
    recordObservation(observation) {
        this._calibrationBuffer.push({
            ...observation,
            timestamp: Date.now(),
        });

        // Maintain window size
        if (this._calibrationBuffer.length > this._calibrationWindowSize) {
            this._calibrationBuffer.shift();
        }
    }

    /**
     * Self-calibrate T_min and T_max from observed data.
     * Uses exponential moving average of observed token costs.
     *
     * @param {number} [smoothing=0.1] - EMA smoothing factor
     * @returns {{ T_min: number, T_max: number, samples: number } | null}
     */
    calibrate(smoothing = 0.1) {
        if (this._calibrationBuffer.length < 10) {
            return null; // Not enough data
        }

        const tokenCosts = this._calibrationBuffer.map((o) => o.tokensUsed);
        const sorted = [...tokenCosts].sort((a, b) => a - b);

        // Use 10th/90th percentiles to avoid outliers
        const p10 = sorted[Math.floor(sorted.length * 0.1)];
        const p90 = sorted[Math.floor(sorted.length * 0.9)];

        // EMA update
        this.params.T_min = this.params.T_min * (1 - smoothing) + p10 * smoothing;
        this.params.T_max = this.params.T_max * (1 - smoothing) + p90 * smoothing;

        // Recompute normalization
        this.T_norm = this.params.T_max * (2 + this.params.alpha);

        return {
            T_min: this.params.T_min,
            T_max: this.params.T_max,
            T_norm: this.T_norm,
            samples: this._calibrationBuffer.length,
        };
    }

    /**
     * Get current model parameters (for display / serialization).
     */
    getParams() {
        return {
            ...this.params,
            T_norm: this.T_norm,
        };
    }

    /**
     * Update model parameters.
     * @param {Object} newParams
     */
    setParams(newParams) {
        this.params = { ...this.params, ...newParams };
        this.T_norm = this.params.T_max * (2 + this.params.alpha);
    }

    /**
     * Generate a human-readable breakdown of the reward computation
     * for a given scenario. Useful for UI display and debugging.
     *
     * @param {number} impact
     * @param {number} irreversibility
     * @param {number} propagation
     * @returns {Object} Detailed breakdown
     */
    explain(impact, irreversibility, propagation) {
        const T_base = this.computeBaseTokenCost(impact, propagation);
        const kappa = this.computeRedoMultiplier(irreversibility);
        const rewards = this.computeRewards(impact, irreversibility, propagation);

        return {
            inputs: {
                impact,
                irreversibility,
                propagation,
            },
            intermediates: {
                phi: (impact + propagation) / 2,
                T_base: Math.round(T_base),
                kappa: kappa.toFixed(2),
                T_norm: this.T_norm,
                T_review: this.params.T_review,
            },
            rewards: {
                TP: rewards.TP.toFixed(4),
                TN: rewards.TN.toFixed(4),
                FP: rewards.FP.toFixed(4),
                FN: rewards.FN.toFixed(4),
            },
            ratios: {
                "FN/FP": Math.abs(rewards.FN / rewards.FP).toFixed(1) + "×",
                "TP/TN": (rewards.TP / rewards.TN).toFixed(2) + "×",
            },
            narrative: [
                `Task complexity: φ = ${((impact + propagation) / 2).toFixed(2)} → T_base ≈ ${Math.round(T_base)} tokens`,
                `Redo multiplier: κ = 1 + ${this.params.alpha} × ${irreversibility.toFixed(1)} = ${kappa.toFixed(2)}`,
                `If we miss this (FN): waste ${Math.round(T_base)} + redo ${Math.round(T_base * kappa)} = ${Math.round(T_base * (1 + kappa))} tokens`,
                `If we catch it (TP): save ${Math.round(T_base)} tokens from being wasted`,
                `If we automate correctly (TN): ${Math.round(T_base)} tokens used efficiently (×${this.params.eta} discount)`,
                `If we interrupt needlessly (FP): ${this.params.T_review} tokens of review overhead`,
            ],
        };
    }
}

/**
 * Verify the mathematical properties of the model.
 * Used for testing / validation.
 *
 * @returns {{ passed: boolean, tests: Object[] }}
 */
export function verifyModelProperties() {
    const model = new TokenCostModel();
    const tests = [];

    // Test 1: FN magnitude >> FP magnitude for all scenarios
    const gravityConfigs = [
        { I: 0.2, R: 0.2, P: 0.2, label: "Low gravity" },
        { I: 0.5, R: 0.5, P: 0.5, label: "Medium gravity" },
        { I: 0.8, R: 0.8, P: 0.8, label: "High gravity" },
        { I: 0.9, R: 0.9, P: 1.0, label: "Critical gravity" },
    ];

    for (const cfg of gravityConfigs) {
        const r = model.computeRewards(cfg.I, cfg.R, cfg.P);
        const fnFpRatio = Math.abs(r.FN / r.FP);
        tests.push({
            name: `|FN| >> |FP| (${cfg.label})`,
            passed: fnFpRatio > 3,
            value: `|FN/FP| = ${fnFpRatio.toFixed(1)}×`,
        });
    }

    // Test 2: TP > TN always (constant ratio = 1/η)
    for (const cfg of gravityConfigs) {
        const r = model.computeRewards(cfg.I, cfg.R, cfg.P);
        tests.push({
            name: `TP > TN (${cfg.label})`,
            passed: r.TP > r.TN,
            value: `TP/TN = ${(r.TP / r.TN).toFixed(2)}×`,
        });
    }

    // Test 3: Higher gravity → higher |FN|
    const lowR = model.computeRewards(0.2, 0.2, 0.2);
    const highR = model.computeRewards(0.9, 0.9, 0.9);
    tests.push({
        name: "Higher gravity → higher |FN|",
        passed: Math.abs(highR.FN) > Math.abs(lowR.FN),
        value: `|FN_high/FN_low| = ${(Math.abs(highR.FN) / Math.abs(lowR.FN)).toFixed(1)}×`,
    });

    // Test 4: FP is constant (independent of gravity)
    tests.push({
        name: "FP is constant across gravity levels",
        passed: Math.abs(lowR.FP - highR.FP) < 1e-10,
        value: `FP_low = ${lowR.FP.toFixed(4)}, FP_high = ${highR.FP.toFixed(4)}`,
    });

    // Test 5: All rewards in reasonable range [-1, 1]
    for (const cfg of gravityConfigs) {
        const r = model.computeRewards(cfg.I, cfg.R, cfg.P);
        const inRange = r.TP <= 1 && r.TN <= 1 && r.FP >= -1 && r.FN >= -1;
        tests.push({
            name: `Rewards in [-1, 1] (${cfg.label})`,
            passed: inRange,
            value: `[${r.FN.toFixed(3)}, ${r.TP.toFixed(3)}]`,
        });
    }

    return {
        passed: tests.every((t) => t.passed),
        tests,
    };
}
