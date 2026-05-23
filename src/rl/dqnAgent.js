/**
 * Deep Q-Network (DQN) Agent
 *
 * Uses an online network for action selection and a target network
 * for stable Q-value computation. Employs ε-greedy exploration
 * with configurable decay schedule.
 *
 * Architecture: Input(7) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(2, Linear)
 */

import { NeuralNetwork } from "./neuralNet.js";
import { ReplayBuffer } from "./replayBuffer.js";

export class DQNAgent {
    /**
     * @param {Object} config
     * @param {number} config.stateSize - Dimension of state vector
     * @param {number} config.actionSize - Number of actions
     * @param {number} config.lr - Learning rate (default 0.001)
     * @param {number} config.gamma - Discount factor (default 0.99)
     * @param {number} config.epsilonStart - Initial exploration rate (default 1.0)
     * @param {number} config.epsilonEnd - Final exploration rate (default 0.05)
     * @param {number} config.epsilonDecay - Decay rate per step (default 0.995)
     * @param {number} config.batchSize - Mini-batch size (default 64)
     * @param {number} config.bufferSize - Replay buffer capacity (default 10000)
     * @param {number} config.targetUpdateFreq - Steps between target network updates (default 100)
     * @param {number} config.hiddenSizes - Hidden layer sizes (default [64, 32])
     */
    constructor(config = {}) {
        this.stateSize = config.stateSize || 7;
        this.actionSize = config.actionSize || 2;
        this.lr = config.lr || 0.001;
        this.gamma = config.gamma || 0.99;
        this.batchSize = config.batchSize || 64;
        this.targetUpdateFreq = config.targetUpdateFreq || 100;

        // Epsilon schedule
        this.epsilon = config.epsilonStart ?? 1.0;
        this.epsilonEnd = config.epsilonEnd ?? 0.05;
        this.epsilonDecay = config.epsilonDecay ?? 0.995;

        // Hidden layer sizes
        const hidden = config.hiddenSizes || [64, 32];

        // Build network architecture
        const layerConfigs = [
            { size: this.stateSize },
            ...hidden.map((s) => ({ size: s, activation: "relu" })),
            { size: this.actionSize, activation: "linear" },
        ];

        // Online (policy) network
        this.onlineNet = new NeuralNetwork(layerConfigs);

        // Target network (initialized as copy)
        this.targetNet = new NeuralNetwork(layerConfigs);
        this.targetNet.copyFrom(this.onlineNet);

        // Replay buffer
        this.replayBuffer = new ReplayBuffer(config.bufferSize || 10000);

        // Training metrics
        this.totalSteps = 0;
        this.trainingLosses = [];
        this.avgQValues = [];
    }

    /**
     * Select an action using ε-greedy policy
     * @param {number[]} state
     * @param {boolean} [training=true] - If false, uses greedy policy (no exploration)
     * @returns {{ action: number, qValues: number[], explorative: boolean }}
     */
    selectAction(state, training = true) {
        // Exploration
        if (training && Math.random() < this.epsilon) {
            return {
                action: Math.floor(Math.random() * this.actionSize),
                qValues: null,
                explorative: true,
            };
        }

        // Exploitation — forward pass through online network
        const qValues = this.onlineNet.forward(state);
        const action = qValues[0] >= qValues[1] ? 0 : 1;

        return { action, qValues, explorative: false };
    }

    /**
     * Store a transition in the replay buffer
     * @param {number[]} state
     * @param {number} action
     * @param {number} reward
     * @param {number[]} nextState
     * @param {boolean} done
     */
    remember(state, action, reward, nextState, done) {
        this.replayBuffer.push(state, action, reward, nextState, done);
    }

    /**
     * Train on a mini-batch from the replay buffer
     * @returns {number|null} Training loss, or null if buffer not ready
     */
    train() {
        if (!this.replayBuffer.isReady(this.batchSize)) {
            return null;
        }

        const batch = this.replayBuffer.sample(this.batchSize);
        let totalLoss = 0;
        let totalQ = 0;

        // Build training batch: compute target Q-values
        const trainBatch = batch.map((transition) => {
            const { state, action, reward, nextState, done } = transition;

            // Current Q-values from online network
            const currentQ = this.onlineNet.forward(state);

            // Target Q-values: use target network for next state
            const nextQ = this.targetNet.forward(nextState);
            const maxNextQ = Math.max(...nextQ);

            // Bellman target
            const target = [...currentQ];
            target[action] = done ? reward : reward + this.gamma * maxNextQ;

            totalQ += Math.max(...currentQ);

            return { input: state, target };
        });

        // Train online network
        const loss = this.onlineNet.trainOnBatch(trainBatch, this.lr);
        totalLoss = loss;

        // Track metrics
        this.trainingLosses.push(totalLoss);
        this.avgQValues.push(totalQ / batch.length);
        this.totalSteps++;

        // Decay epsilon
        this.epsilon = Math.max(this.epsilonEnd, this.epsilon * this.epsilonDecay);

        // Update target network periodically
        if (this.totalSteps % this.targetUpdateFreq === 0) {
            this.targetNet.softUpdate(this.onlineNet, 0.01);
        }

        return totalLoss;
    }

    /**
     * Get the agent's learned decision boundary
     * Analyzes what composite score (U×G) the agent uses to decide.
     * Probes the network at various score levels to find the crossover point.
     *
     * @returns {{ threshold: number, confidence: number }}
     */
    getLearnedThreshold() {
        // Probe the network with varying composite scores
        // Use average values for other features
        const probeResults = [];

        for (let s = 0; s <= 100; s++) {
            const score = s / 100;
            // Create a synthetic state with proportional uncertainty/gravity
            const state = [
                score * 0.6,  // semantic
                score * 0.5,  // entropy
                score * 0.4,  // reflection
                score * 0.8,  // impact
                score * 0.7,  // irreversibility
                score * 0.6,  // propagation
                score,        // composite score
            ];

            const qValues = this.onlineNet.forward(state);
            const prefersIntervene = qValues[1] > qValues[0];

            probeResults.push({ score, qValues, prefersIntervene });
        }

        // Find the crossover point (where action switches from automate to intervene)
        let threshold = 0.5; // default
        for (let i = 1; i < probeResults.length; i++) {
            if (!probeResults[i - 1].prefersIntervene && probeResults[i].prefersIntervene) {
                threshold = probeResults[i].score;
                break;
            }
        }

        // Confidence: how consistent is the boundary?
        let consistent = 0;
        for (const r of probeResults) {
            if ((r.score < threshold && !r.prefersIntervene) ||
                (r.score >= threshold && r.prefersIntervene)) {
                consistent++;
            }
        }
        const confidence = consistent / probeResults.length;

        return { threshold, confidence };
    }

    /**
     * Get recent training metrics
     * @param {number} window - Number of recent steps to average
     */
    getMetrics(window = 50) {
        const recentLosses = this.trainingLosses.slice(-window);
        const recentQ = this.avgQValues.slice(-window);

        return {
            epsilon: this.epsilon,
            totalSteps: this.totalSteps,
            bufferSize: this.replayBuffer.size,
            avgLoss: recentLosses.length > 0
                ? recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length
                : 0,
            avgQ: recentQ.length > 0
                ? recentQ.reduce((a, b) => a + b, 0) / recentQ.length
                : 0,
        };
    }

    /**
     * Serialize the agent's state (for saving/loading)
     */
    serialize() {
        return {
            config: {
                stateSize: this.stateSize,
                actionSize: this.actionSize,
                lr: this.lr,
                gamma: this.gamma,
                batchSize: this.batchSize,
                targetUpdateFreq: this.targetUpdateFreq,
                epsilonEnd: this.epsilonEnd,
                epsilonDecay: this.epsilonDecay,
            },
            epsilon: this.epsilon,
            totalSteps: this.totalSteps,
            onlineNet: this.onlineNet.serialize(),
            targetNet: this.targetNet.serialize(),
        };
    }

    /**
     * Restore from serialized state
     */
    static deserialize(data) {
        const agent = new DQNAgent(data.config);
        agent.epsilon = data.epsilon;
        agent.totalSteps = data.totalSteps;
        agent.onlineNet = NeuralNetwork.deserialize(data.onlineNet);
        agent.targetNet = NeuralNetwork.deserialize(data.targetNet);
        return agent;
    }
}
