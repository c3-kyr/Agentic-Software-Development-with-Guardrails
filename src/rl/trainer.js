/**
 * RL Trainer — Orchestrates the DQN training loop
 *
 * Runs episodes, collects experience, trains the DQN agent,
 * and periodically invokes the parameter optimizer.
 *
 * Uses requestAnimationFrame for non-blocking UI updates.
 * Emits progress events for dashboard visualization.
 */

import { RLEnvironment } from "./environment.js";
import { DQNAgent } from "./dqnAgent.js";
import { ParamOptimizer } from "./paramOptimizer.js";

/**
 * @typedef {Object} TrainerConfig
 * @property {number} totalEpisodes - Total episodes to train (default 500)
 * @property {number} paramUpdateEvery - Episodes between parameter optimization (default 25)
 * @property {number} logEvery - Episodes between metric logging (default 5)
 * @property {string} scenarioMode - 'preset' | 'random' | 'mixed'
 * @property {number} randomCount - Random scenarios per episode
 * @property {Object} agentConfig - DQN agent configuration overrides
 * @property {Object} paramConfig - Parameter optimizer configuration overrides
 * @property {Function} onProgress - Callback: (progressData) => void
 * @property {Function} onEpisode - Callback: (episodeData) => void
 * @property {Function} onParamUpdate - Callback: (paramData) => void
 * @property {Function} onComplete - Callback: (finalResults) => void
 */

export class RLTrainer {
    /**
     * @param {TrainerConfig} config
     */
    constructor(config = {}) {
        this.totalEpisodes = config.totalEpisodes || 500;
        this.paramUpdateEvery = config.paramUpdateEvery || 25;
        this.logEvery = config.logEvery || 5;

        // Callbacks
        this.onProgress = config.onProgress || (() => {});
        this.onEpisode = config.onEpisode || (() => {});
        this.onParamUpdate = config.onParamUpdate || (() => {});
        this.onComplete = config.onComplete || (() => {});

        // Environment (with Token-Cost Grounded Rewards by default)
        this.env = new RLEnvironment({
            scenarioMode: config.scenarioMode || "mixed",
            randomCount: config.randomCount || 50,
            weights: config.initialWeights || [0.4, 0.35, 0.25],
            rewardMode: config.rewardMode || "token-cost",
            tokenCostParams: config.tokenCostParams || {},
        });

        // DQN Agent
        this.agent = new DQNAgent({
            stateSize: this.env.stateSize,
            actionSize: this.env.actionSize,
            ...(config.agentConfig || {}),
        });

        // Parameter Optimizer
        this.paramOptimizer = new ParamOptimizer({
            initialWeights: config.initialWeights || [0.4, 0.35, 0.25],
            initialLambda: config.initialLambda || 0.5,
            ...(config.paramConfig || {}),
        });

        // Training state
        this.currentEpisode = 0;
        this.isRunning = false;
        this.isPaused = false;
        this.trainingHistory = [];
        this.paramHistory = [];
        this.bestResult = null;

        // For async control
        this._resolveStop = null;
        this._animFrameId = null;
    }

    /**
     * Start or resume training
     * @returns {Promise<Object>} Final results when training completes
     */
    async start() {
        if (this.isRunning && !this.isPaused) return;

        this.isRunning = true;
        this.isPaused = false;

        return new Promise((resolve) => {
            this._resolveStop = resolve;
            this._runNextEpisode();
        });
    }

    /**
     * Pause training (can be resumed)
     */
    pause() {
        this.isPaused = true;
        if (this._animFrameId) {
            cancelAnimationFrame(this._animFrameId);
            this._animFrameId = null;
        }
    }

    /**
     * Stop training completely
     */
    stop() {
        this.isRunning = false;
        this.isPaused = false;
        if (this._animFrameId) {
            cancelAnimationFrame(this._animFrameId);
            this._animFrameId = null;
        }
        if (this._resolveStop) {
            this._resolveStop(this.getFinalResults());
            this._resolveStop = null;
        }
    }

    /**
     * Reset training state
     * @param {Object} [config] - Optional new configuration
     */
    reset(config = {}) {
        this.stop();
        this.currentEpisode = 0;
        this.trainingHistory = [];
        this.paramHistory = [];
        this.bestResult = null;

        if (config.totalEpisodes) this.totalEpisodes = config.totalEpisodes;

        // Reinitialize agent
        this.agent = new DQNAgent({
            stateSize: this.env.stateSize,
            actionSize: this.env.actionSize,
            ...(config.agentConfig || {}),
        });

        // Reinitialize param optimizer
        this.paramOptimizer.reset(
            config.initialWeights || [0.4, 0.35, 0.25],
            config.initialLambda || 0.5
        );

        this.env.setWeights(
            config.initialWeights || this.paramOptimizer.getWeights()
        );
    }

    // ─── Internal Training Loop ───

    _runNextEpisode() {
        if (!this.isRunning || this.isPaused || this.currentEpisode >= this.totalEpisodes) {
            if (this.currentEpisode >= this.totalEpisodes) {
                this.isRunning = false;
                const results = this.getFinalResults();
                this.onComplete(results);
                if (this._resolveStop) {
                    this._resolveStop(results);
                    this._resolveStop = null;
                }
            }
            return;
        }

        // Run one episode
        const episodeData = this._runEpisode();
        this.trainingHistory.push(episodeData);
        this.currentEpisode++;

        // Track best result
        if (!this.bestResult || episodeData.f1 > this.bestResult.f1) {
            this.bestResult = { ...episodeData, episode: this.currentEpisode };
        }

        // Report episode
        this.onEpisode(episodeData);

        // Report progress
        if (this.currentEpisode % this.logEvery === 0) {
            this.onProgress(this._getProgressData());
        }

        // Parameter optimization (on a slower timescale)
        if (this.currentEpisode % this.paramUpdateEvery === 0 && this.currentEpisode > 0) {
            this._runParamUpdate();
        }

        // Schedule next episode via rAF (non-blocking)
        this._animFrameId = requestAnimationFrame(() => this._runNextEpisode());
    }

    /**
     * Run a single episode
     */
    _runEpisode() {
        const currentWeights = this.paramOptimizer.getWeights();
        let state = this.env.reset(currentWeights);
        let totalReward = 0;
        let stepCount = 0;
        let trainLoss = 0;
        let trainSteps = 0;

        while (true) {
            // Agent selects action
            const { action } = this.agent.selectAction(state);

            // Environment step
            const result = this.env.step(action);

            // Store transition
            this.agent.remember(state, action, result.reward, result.state, result.done);

            // Train agent
            const loss = this.agent.train();
            if (loss !== null) {
                trainLoss += loss;
                trainSteps++;
            }

            totalReward += result.reward;
            stepCount++;
            state = result.state;

            if (result.done) break;
        }

        const summary = this.env.getEpisodeSummary();
        const agentMetrics = this.agent.getMetrics();
        const learnedThreshold = this.agent.getLearnedThreshold();

        return {
            episode: this.currentEpisode,
            totalReward: summary.totalReward,
            f1: summary.f1,
            precision: summary.precision,
            recall: summary.recall,
            accuracy: summary.accuracy,
            interruptRate: summary.interruptRate,
            errorRate: summary.errorRate,
            metrics: summary.metrics,
            epsilon: agentMetrics.epsilon,
            avgLoss: trainSteps > 0 ? trainLoss / trainSteps : 0,
            avgQ: agentMetrics.avgQ,
            learnedThreshold: learnedThreshold.threshold,
            thresholdConfidence: learnedThreshold.confidence,
            weights: [...currentWeights],
            lambda: this.paramOptimizer.getLambda(),
            steps: stepCount,
        };
    }

    /**
     * Run parameter optimization step
     */
    _runParamUpdate() {
        // Quick evaluation function: run one episode and return total reward
        const evaluateFn = (weights, _lambda) => {
            const evalEnv = new RLEnvironment({
                scenarioMode: "preset", // Use preset for consistent evaluation
                weights,
                rewardMode: this.env.rewardMode || "token-cost",
            });

            let state = evalEnv.reset();
            let totalReward = 0;

            while (true) {
                const { action } = this.agent.selectAction(state, false); // Greedy
                const result = evalEnv.step(action);
                totalReward += result.reward;
                state = result.state;
                if (result.done) break;
            }

            return totalReward;
        };

        // Synchronous version using direct evaluation (not async for simplicity)
        const currentWeights = this.paramOptimizer.getWeights();
        const currentLambda = this.paramOptimizer.getLambda();
        const baseReturn = evaluateFn(currentWeights, currentLambda);

        // Estimate weight gradients
        const weightGrads = [];
        const delta = this.paramOptimizer.delta;

        for (let i = 0; i < 3; i++) {
            const logitsPlus = [...this.paramOptimizer.weightLogits];
            logitsPlus[i] += delta;
            const wPlus = this.paramOptimizer._softmax(logitsPlus);
            const rPlus = evaluateFn(wPlus, currentLambda);

            const logitsMinus = [...this.paramOptimizer.weightLogits];
            logitsMinus[i] -= delta;
            const wMinus = this.paramOptimizer._softmax(logitsMinus);
            const rMinus = evaluateFn(wMinus, currentLambda);

            weightGrads.push((rPlus - rMinus) / (2 * delta));
        }

        // Apply gradients with momentum
        for (let i = 0; i < 3; i++) {
            this.paramOptimizer.velocityWeights[i] =
                this.paramOptimizer.momentum * this.paramOptimizer.velocityWeights[i] +
                this.paramOptimizer.lr * weightGrads[i];
            this.paramOptimizer.weightLogits[i] += this.paramOptimizer.velocityWeights[i];
        }

        const paramRecord = {
            step: this.paramOptimizer.updateCount++,
            episode: this.currentEpisode,
            params: this.paramOptimizer.getParams(),
            baseReturn,
            gradient: { weights: weightGrads },
        };

        this.paramHistory.push(paramRecord);
        this.onParamUpdate(paramRecord);

        // Update environment with new weights
        this.env.setWeights(this.paramOptimizer.getWeights());
    }

    /**
     * Build progress data snapshot
     */
    _getProgressData() {
        const recent = this.trainingHistory.slice(-50);
        const avgReward = recent.reduce((s, e) => s + e.totalReward, 0) / (recent.length || 1);
        const avgF1 = recent.reduce((s, e) => s + e.f1, 0) / (recent.length || 1);

        return {
            episode: this.currentEpisode,
            totalEpisodes: this.totalEpisodes,
            progress: this.currentEpisode / this.totalEpisodes,
            avgReward,
            avgF1,
            epsilon: this.agent.epsilon,
            bestF1: this.bestResult?.f1 || 0,
            currentWeights: this.paramOptimizer.getWeights(),
            currentLambda: this.paramOptimizer.getLambda(),
            learnedThreshold: this.bestResult?.learnedThreshold || 0.15,
        };
    }

    /**
     * Get final training results
     */
    getFinalResults() {
        const recent = this.trainingHistory.slice(-50);
        const avgReward = recent.reduce((s, e) => s + e.totalReward, 0) / (recent.length || 1);
        const avgF1 = recent.reduce((s, e) => s + e.f1, 0) / (recent.length || 1);
        const avgPrecision = recent.reduce((s, e) => s + e.precision, 0) / (recent.length || 1);
        const avgRecall = recent.reduce((s, e) => s + e.recall, 0) / (recent.length || 1);

        const learnedThreshold = this.agent.getLearnedThreshold();
        const optimizedParams = this.paramOptimizer.getParams();

        return {
            totalEpisodes: this.currentEpisode,
            converged: avgF1 > 0.8,
            learnedThreshold: learnedThreshold.threshold,
            thresholdConfidence: learnedThreshold.confidence,
            optimizedWeights: optimizedParams.weights,
            optimizedLambda: optimizedParams.lambda,
            rewardModel: this.env.getRewardModelInfo(),
            performance: {
                avgReward,
                avgF1,
                avgPrecision,
                avgRecall,
                bestF1: this.bestResult?.f1 || 0,
                bestEpisode: this.bestResult?.episode || 0,
            },
            trainingHistory: this.trainingHistory,
            paramHistory: this.paramHistory,
        };
    }

    /**
     * Check if currently training
     */
    get isActive() {
        return this.isRunning && !this.isPaused;
    }

    /**
     * Get current status
     */
    get status() {
        if (!this.isRunning) return "stopped";
        if (this.isPaused) return "paused";
        return "running";
    }
}
