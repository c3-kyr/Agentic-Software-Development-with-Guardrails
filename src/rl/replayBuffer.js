/**
 * Experience Replay Buffer
 *
 * Stores (state, action, reward, nextState, done) transitions
 * and provides uniform random sampling for mini-batch training.
 * Uses a circular buffer to bound memory usage.
 */

export class ReplayBuffer {
    /**
     * @param {number} capacity - Maximum number of transitions to store
     */
    constructor(capacity = 10000) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    /**
     * Store a transition
     * @param {number[]} state
     * @param {number} action
     * @param {number} reward
     * @param {number[]} nextState
     * @param {boolean} done
     */
    push(state, action, reward, nextState, done) {
        const transition = { state, action, reward, nextState, done };

        if (this.buffer.length < this.capacity) {
            this.buffer.push(transition);
        } else {
            this.buffer[this.position] = transition;
        }
        this.position = (this.position + 1) % this.capacity;
    }

    /**
     * Sample a random mini-batch
     * @param {number} batchSize
     * @returns {Array<{state, action, reward, nextState, done}>}
     */
    sample(batchSize) {
        const samples = [];
        const len = this.buffer.length;

        for (let i = 0; i < batchSize; i++) {
            const idx = Math.floor(Math.random() * len);
            samples.push(this.buffer[idx]);
        }

        return samples;
    }

    /**
     * Current number of stored transitions
     */
    get size() {
        return this.buffer.length;
    }

    /**
     * Check if buffer has enough samples for training
     * @param {number} minSize
     */
    isReady(minSize = 64) {
        return this.buffer.length >= minSize;
    }

    /**
     * Clear the buffer
     */
    clear() {
        this.buffer = [];
        this.position = 0;
    }

    /**
     * Get statistics
     */
    stats() {
        if (this.buffer.length === 0) {
            return { size: 0, avgReward: 0, rewardStd: 0 };
        }

        const rewards = this.buffer.map((t) => t.reward);
        const avg = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        const variance = rewards.reduce((s, r) => s + (r - avg) ** 2, 0) / rewards.length;

        return {
            size: this.buffer.length,
            avgReward: avg,
            rewardStd: Math.sqrt(variance),
        };
    }
}
