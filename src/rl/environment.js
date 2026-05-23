/**
 * RL Environment for HITL Threshold Optimization
 *
 * Wraps the scenario simulation as a Markov Decision Process:
 *   State:  [U_semantic, U_entropy, U_reflection, G_impact, G_irrev, G_prop, S_score]
 *   Action: {0: automate, 1: intervene}
 *   Reward: Derived from Token-Cost Model (see tokenCostModel.js)
 *
 * ────────────────────────────────────────────────────────────────────
 * Reward Philosophy (Token-Cost Grounded):
 *
 *   Instead of arbitrary constants, rewards are derived from the
 *   estimated token expenditure of each decision path:
 *
 *     R_TP =  +T_base            / T_norm    (tokens saved)
 *     R_TN =  +T_base · η       / T_norm    (efficient automation)
 *     R_FP =  −T_review          / T_norm    (review overhead)
 *     R_FN =  −T_base · (1 + κ) / T_norm    (wasted + redo cost)
 *
 *   where T_base, κ, T_norm are functions of (Impact, Irrev., Prop.)
 *
 *   This produces mathematically proven asymmetry:
 *     |FN| >> |FP|   (missing a problem is far costlier than a false alarm)
 *     TP > TN        (catching a problem is more valuable than smooth automation)
 *
 *   See: token_cost_reward_derivation.md for the full proof.
 * ────────────────────────────────────────────────────────────────────
 *
 * One episode = evaluating all scenarios in the batch.
 */

import { computeCompositeUncertainty } from "../uncertainty/compositeUncertainty.js";
import { computeGravity } from "../gravity/gravityCalc.js";
import { PRESET_SCENARIOS, generateRandomBatch } from "../simulation/scenarios.js";
import { TokenCostModel, DEFAULT_TOKEN_PARAMS } from "./tokenCostModel.js";

/**
 * Legacy reward structure — kept as fallback for comparison experiments.
 * @deprecated Use TokenCostModel instead.
 */
export const REWARDS_LEGACY = {
    TP: 1.0,   // Correctly intervened on a real problem
    TN: 0.8,   // Correctly automated a safe scenario
    FP: -0.3,  // Unnecessarily interrupted human
    FN: -2.0,  // Missed a real problem (worst outcome)
};

export class RLEnvironment {
    /**
     * @param {Object} config
     * @param {string} config.scenarioMode - 'preset' | 'random' | 'mixed'
     * @param {number} config.randomCount - Number of random scenarios (for 'random' and 'mixed')
     * @param {number[]} config.weights - Uncertainty weights [w1, w2, w3]
     * @param {Object} config.rewards - Custom reward overrides (legacy mode only)
     * @param {string} config.rewardMode - 'token-cost' (default) | 'legacy'
     * @param {Object} config.tokenCostParams - Token cost model parameters (overrides)
     */
    constructor(config = {}) {
        this.scenarioMode = config.scenarioMode || "mixed";
        this.randomCount = config.randomCount || 50;
        this.weights = config.weights || [0.4, 0.35, 0.25];

        // Reward mode selection
        this.rewardMode = config.rewardMode || "token-cost";

        if (this.rewardMode === "token-cost") {
            // Token-Cost Grounded Reward Model
            this.tokenCostModel = new TokenCostModel(config.tokenCostParams || {});
        } else {
            // Legacy: fixed reward constants
            this.rewards = { ...REWARDS_LEGACY, ...(config.rewards || {}) };
        }

        // Episode state
        this.scenarios = [];
        this.currentIndex = 0;
        this.episodeReward = 0;
        this.episodeMetrics = { tp: 0, tn: 0, fp: 0, fn: 0 };

        // Token cost tracking (for reward analysis)
        this.episodeRewardBreakdown = [];

        // Precomputed features for current scenario batch
        this.featureCache = [];

        this.reset();
    }

    /**
     * State dimensionality
     */
    get stateSize() {
        return 7;
    }

    /**
     * Number of possible actions
     */
    get actionSize() {
        return 2;
    }

    /**
     * Reset environment for a new episode
     * @param {number[]} [weights] - Optionally update weights
     * @returns {number[]} Initial state
     */
    reset(weights) {
        if (weights) {
            this.weights = weights;
        }

        // Generate scenarios based on mode
        switch (this.scenarioMode) {
            case "preset":
                this.scenarios = [...PRESET_SCENARIOS];
                break;
            case "random":
                this.scenarios = generateRandomBatch(this.randomCount, Date.now());
                break;
            case "mixed":
            default:
                this.scenarios = [
                    ...PRESET_SCENARIOS,
                    ...generateRandomBatch(this.randomCount, Date.now()),
                ];
                break;
        }

        // Shuffle scenarios for each episode (prevents ordering bias)
        this._shuffle(this.scenarios);

        // Precompute features for all scenarios
        this.featureCache = this.scenarios.map((s) => this._computeFeatures(s));

        this.currentIndex = 0;
        this.episodeReward = 0;
        this.episodeMetrics = { tp: 0, tn: 0, fp: 0, fn: 0 };
        this.episodeRewardBreakdown = [];

        return this.featureCache[0];
    }

    /**
     * Take an action for the current scenario
     * @param {number} action - 0 (automate) or 1 (intervene)
     * @returns {{ state: number[], reward: number, done: boolean, info: Object }}
     */
    step(action) {
        const scenario = this.scenarios[this.currentIndex];
        const predicted = action === 1; // 1 = intervene
        const actual = scenario.groundTruthNeedsHuman;

        // Determine outcome classification
        let classification;
        if (predicted && actual) {
            classification = "TP";
            this.episodeMetrics.tp++;
        } else if (!predicted && !actual) {
            classification = "TN";
            this.episodeMetrics.tn++;
        } else if (predicted && !actual) {
            classification = "FP";
            this.episodeMetrics.fp++;
        } else {
            classification = "FN";
            this.episodeMetrics.fn++;
        }

        // Compute reward based on selected mode
        let reward;
        let rewardMeta = null;

        if (this.rewardMode === "token-cost") {
            // ═══════════════════════════════════════════════════════════
            // TOKEN-COST GROUNDED REWARDS
            //
            // Each reward is derived from the estimated token cost of
            // the decision path. The gravity factors (I, R, P) of the
            // scenario determine T_base and κ, which in turn determine
            // the reward magnitude.
            //
            // This produces scenario-adaptive rewards where:
            //   - Missing a high-gravity problem is severely penalized
            //   - Missing a low-gravity problem is mildly penalized
            //   - FP cost is always small and constant (human review overhead)
            //   - TP/TN scale with task complexity
            // ═══════════════════════════════════════════════════════════
            const scenarioRewards = this.tokenCostModel.computeRewards(
                scenario.impact,
                scenario.irreversibility,
                scenario.propagation
            );

            reward = scenarioRewards[classification];
            rewardMeta = scenarioRewards.meta;
        } else {
            // ═══════════════════════════════════════════════════════════
            // LEGACY MODE: Fixed rewards with gravity scaling
            // ═══════════════════════════════════════════════════════════
            const gravityObj = computeGravity(
                scenario.impact,
                scenario.irreversibility,
                scenario.propagation
            );
            const G = gravityObj.G;
            const baseReward = this.rewards[classification];

            if (classification === "TP" || classification === "FN") {
                reward = baseReward * Math.max(0.1, G);
            } else {
                reward = baseReward;
            }
        }

        this.episodeReward += reward;

        // Track reward breakdown for analysis
        this.episodeRewardBreakdown.push({
            scenarioId: scenario.id,
            classification,
            reward,
            impact: scenario.impact,
            irreversibility: scenario.irreversibility,
            propagation: scenario.propagation,
            meta: rewardMeta,
        });

        this.currentIndex++;

        const done = this.currentIndex >= this.scenarios.length;
        const nextState = done ? new Array(this.stateSize).fill(0) : this.featureCache[this.currentIndex];

        return {
            state: nextState,
            reward,
            done,
            info: {
                scenarioId: scenario.id,
                scenarioLabel: scenario.label,
                classification,
                predicted,
                actual,
                episodeReward: this.episodeReward,
                step: this.currentIndex,
                totalSteps: this.scenarios.length,
                rewardMeta,
            },
        };
    }

    /**
     * Get current state without stepping
     */
    getCurrentState() {
        if (this.currentIndex >= this.featureCache.length) {
            return new Array(this.stateSize).fill(0);
        }
        return this.featureCache[this.currentIndex];
    }

    /**
     * Get episode summary after completion
     */
    getEpisodeSummary() {
        const { tp, tn, fp, fn } = this.episodeMetrics;
        const total = tp + tn + fp + fn;
        const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
        const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        const accuracy = total > 0 ? (tp + tn) / total : 0;
        const interruptRate = total > 0 ? (tp + fp) / total : 0;
        const errorRate = total > 0 ? fn / total : 0;

        return {
            totalReward: this.episodeReward,
            metrics: { ...this.episodeMetrics },
            precision,
            recall,
            f1,
            accuracy,
            interruptRate,
            errorRate,
            totalScenarios: total,
            rewardBreakdown: this.episodeRewardBreakdown,
        };
    }

    /**
     * Update the uncertainty weights (used by parameter optimizer)
     * @param {number[]} newWeights
     */
    setWeights(newWeights) {
        this.weights = [...newWeights];
    }

    /**
     * Update the reward structure (legacy mode only)
     * @param {Object} newRewards
     */
    setRewards(newRewards) {
        if (this.rewardMode === "legacy") {
            this.rewards = { ...this.rewards, ...newRewards };
        }
    }

    /**
     * Update token cost model parameters (token-cost mode only)
     * @param {Object} newParams - Partial params to update
     */
    setTokenCostParams(newParams) {
        if (this.rewardMode === "token-cost" && this.tokenCostModel) {
            this.tokenCostModel.setParams(newParams);
        }
    }

    /**
     * Get the current reward model configuration for display
     * @returns {Object}
     */
    getRewardModelInfo() {
        if (this.rewardMode === "token-cost") {
            return {
                mode: "token-cost",
                params: this.tokenCostModel.getParams(),
                description: "Rewards derived from token economics (T_base, κ, T_review)",
            };
        } else {
            return {
                mode: "legacy",
                rewards: { ...this.rewards },
                description: "Fixed reward constants with gravity scaling",
            };
        }
    }

    // ─── Internal ───

    /**
     * Compute the 7-dimensional feature vector for a scenario
     */
    _computeFeatures(scenario) {
        const uncertainty = computeCompositeUncertainty(scenario, this.weights);
        const gravity = computeGravity(
            scenario.impact,
            scenario.irreversibility,
            scenario.propagation
        );
        const score = uncertainty.U * gravity.G;

        return [
            uncertainty.components.semantic,
            uncertainty.components.entropy,
            uncertainty.components.reflection,
            gravity.factors.impact,
            gravity.factors.irreversibility,
            gravity.factors.propagation,
            score,
        ];
    }

    /**
     * Fisher-Yates shuffle
     */
    _shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }
}
