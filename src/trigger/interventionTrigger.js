/**
 * Intervention Trigger
 *
 * S = U × G
 * interrupt = S > threshold
 */

/**
 * Evaluate whether to trigger HITL intervention
 *
 * @param {number} U - Composite uncertainty [0, 1]
 * @param {number} G - Decision gravity [0, 1]
 * @param {number} threshold - Intervention threshold [0, 1]
 * @returns {{ score: number, interrupt: boolean, margin: number }}
 */
export function evaluateTrigger(U, G, threshold) {
    const score = U * G;
    const interrupt = score > threshold;
    const margin = score - threshold;

    return {
        score,
        interrupt,
        margin,
    };
}
