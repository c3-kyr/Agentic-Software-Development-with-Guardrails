/**
 * Decision-Level Entropy Calculator
 *
 * U_entropy = -Σ p_i * log₂(p_i)
 * Normalized to [0, 1] by dividing by log₂(numOptions)
 */

/**
 * Compute normalized decision-level entropy
 *
 * @param {Object} confidenceDistribution - e.g. { "SQL": 0.55, "NoSQL": 0.45 }
 * @returns {{ score: number, rawEntropy: number, maxEntropy: number, distribution: Object }}
 */
export function computeEntropy(confidenceDistribution) {
    const keys = Object.keys(confidenceDistribution);
    const n = keys.length;

    if (n <= 1) {
        return { score: 0, rawEntropy: 0, maxEntropy: 0, distribution: confidenceDistribution };
    }

    // Normalize probabilities to sum = 1
    const raw = keys.map((k) => Math.max(confidenceDistribution[k], 0));
    const sum = raw.reduce((a, b) => a + b, 0);
    const probs = sum > 0 ? raw.map((p) => p / sum) : raw.map(() => 1 / n);

    // H = -Σ p_i * log₂(p_i)
    let entropy = 0;
    for (const p of probs) {
        if (p > 0) {
            entropy -= p * Math.log2(p);
        }
    }

    const maxEntropy = Math.log2(n);
    const normalizedScore = maxEntropy > 0 ? entropy / maxEntropy : 0;

    // Rebuild normalized distribution
    const normalizedDist = {};
    keys.forEach((k, i) => {
        normalizedDist[k] = probs[i];
    });

    return {
        score: normalizedScore,
        rawEntropy: entropy,
        maxEntropy: maxEntropy,
        distribution: normalizedDist,
    };
}
