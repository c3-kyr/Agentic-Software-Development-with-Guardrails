/**
 * Decision Gravity Calculator
 *
 * G = Impact × Irreversibility × Propagation
 *
 * Each factor ∈ [0, 1]
 */

/** Reference scales for each gravity factor */
export const GRAVITY_SCALES = {
    impact: {
        label: "Impact",
        description: "How severe is failure?",
        levels: [
            { value: 0.1, label: "Cosmetic", description: "Visual/UX only" },
            { value: 0.3, label: "Minor", description: "Reduced functionality" },
            { value: 0.5, label: "Moderate", description: "Performance degradation" },
            { value: 0.8, label: "Major", description: "Feature unavailable" },
            { value: 1.0, label: "Critical", description: "Data corruption / loss" },
        ],
    },
    irreversibility: {
        label: "Irreversibility",
        description: "How hard to fix later?",
        levels: [
            { value: 0.2, label: "Trivial", description: "Config change" },
            { value: 0.4, label: "Easy", description: "Code refactor" },
            { value: 0.6, label: "Moderate", description: "API migration" },
            { value: 0.8, label: "Hard", description: "Schema migration" },
            { value: 1.0, label: "Permanent", description: "Data loss / broken contracts" },
        ],
    },
    propagation: {
        label: "Propagation",
        description: "How far does the error spread?",
        levels: [
            { value: 0.2, label: "Isolated", description: "Single function" },
            { value: 0.4, label: "Component", description: "One module" },
            { value: 0.6, label: "Service", description: "Multiple modules" },
            { value: 0.8, label: "Cross-service", description: "Multiple services" },
            { value: 1.0, label: "System-wide", description: "Entire architecture" },
        ],
    },
};

/**
 * Compute Decision Gravity
 *
 * @param {number} impact - Impact factor [0, 1]
 * @param {number} irreversibility - Irreversibility factor [0, 1]
 * @param {number} propagation - Propagation factor [0, 1]
 * @returns {{ G: number, factors: Object }}
 */
export function computeGravity(impact, irreversibility, propagation) {
    const i = Math.max(0, Math.min(1, impact));
    const r = Math.max(0, Math.min(1, irreversibility));
    const p = Math.max(0, Math.min(1, propagation));

    return {
        G: i * r * p,
        factors: {
            impact: i,
            irreversibility: r,
            propagation: p,
        },
    };
}

/**
 * Get human-readable level label for a gravity factor value
 */
export function getGravityLevel(factor, value) {
    const scale = GRAVITY_SCALES[factor];
    if (!scale) return "Unknown";

    let closest = scale.levels[0];
    let minDist = Math.abs(value - closest.value);

    for (const level of scale.levels) {
        const d = Math.abs(value - level.value);
        if (d < minDist) {
            minDist = d;
            closest = level;
        }
    }

    return closest;
}
