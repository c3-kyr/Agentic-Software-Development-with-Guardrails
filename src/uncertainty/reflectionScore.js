/**
 * System-2 Reflection / Structured Disagreement Score
 *
 * U_reflection = 1 - agreement_score
 * Weighted version factors in severity of conflict points.
 */

/**
 * Compute reflection-based uncertainty score
 *
 * @param {number} agreementScore - Model B's agreement with Model A [0, 1]
 * @param {string[]} conflictPoints - List of identified conflicts
 * @param {number[]} conflictSeverities - Severity of each conflict [0, 1]
 * @returns {{ score: number, weightedScore: number, conflictCount: number, severityBreakdown: Object[] }}
 */
export function computeReflectionScore(agreementScore, conflictPoints = [], conflictSeverities = []) {
    // Base score: simple disagreement
    const baseScore = 1 - Math.max(0, Math.min(1, agreementScore));

    // Weighted score: incorporate conflict severity
    let weightedScore = baseScore;
    if (conflictPoints.length > 0 && conflictSeverities.length > 0) {
        const avgSeverity =
            conflictSeverities.reduce((a, b) => a + b, 0) / conflictSeverities.length;
        // Blend base disagreement with severity-weighted component
        // More conflicts with higher severity → higher uncertainty
        const severityFactor = avgSeverity * Math.min(conflictPoints.length / 5, 1);
        weightedScore = 0.6 * baseScore + 0.4 * severityFactor;
    }

    const severityBreakdown = conflictPoints.map((point, i) => ({
        conflict: point,
        severity: conflictSeverities[i] || 0.5,
    }));

    return {
        score: baseScore,
        weightedScore: Math.min(1, weightedScore),
        conflictCount: conflictPoints.length,
        severityBreakdown,
    };
}
