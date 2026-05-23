/**
 * Composite Uncertainty Calculator
 *
 * U = w₁ * U_semantic + w₂ * U_entropy + w₃ * U_reflection
 */

import { computeSemanticDensity } from "./semanticDensity.js";
import { computeEntropy } from "./entropyCalc.js";
import { computeReflectionScore } from "./reflectionScore.js";

/**
 * Compute composite uncertainty from all three signals
 *
 * @param {Object} scenario - A scenario object with all required fields
 * @param {number[]} weights - [w1, w2, w3] (auto-normalized to sum=1)
 * @returns {Object} Full uncertainty analysis
 */
export function computeCompositeUncertainty(scenario, weights = [0.4, 0.35, 0.25]) {
    // Normalize weights
    const wSum = weights.reduce((a, b) => a + b, 0);
    const w = weights.map((x) => (wSum > 0 ? x / wSum : 1 / weights.length));

    // Compute each component
    const semantic = computeSemanticDensity(scenario.embeddingVectors);
    const entropy = computeEntropy(scenario.confidenceDistribution);
    const reflection = computeReflectionScore(
        scenario.reflectionAgreement,
        scenario.reflectionConflicts,
        scenario.conflictSeverities
    );

    // Use weighted reflection score for composite
    const U =
        w[0] * semantic.score +
        w[1] * entropy.score +
        w[2] * reflection.weightedScore;

    return {
        U: Math.min(1, Math.max(0, U)),
        components: {
            semantic: semantic.score,
            entropy: entropy.score,
            reflection: reflection.weightedScore,
        },
        details: {
            semantic,
            entropy,
            reflection,
        },
        weights: { w1: w[0], w2: w[1], w3: w[2] },
    };
}
