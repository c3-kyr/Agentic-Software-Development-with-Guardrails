/**
 * Pure JavaScript Neural Network
 *
 * Minimal implementation sufficient for DQN with small state spaces.
 * Supports: Dense layers, ReLU/Sigmoid/Linear activations,
 * Xavier initialization, forward + backprop with SGD/Adam.
 */

// ─── Activation Functions ───

const ACTIVATIONS = {
    relu: {
        fn: (x) => Math.max(0, x),
        deriv: (x) => (x > 0 ? 1 : 0),
    },
    sigmoid: {
        fn: (x) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))),
        deriv: (x) => {
            const s = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
            return s * (1 - s);
        },
    },
    tanh: {
        fn: (x) => Math.tanh(x),
        deriv: (x) => 1 - Math.tanh(x) ** 2,
    },
    linear: {
        fn: (x) => x,
        deriv: () => 1,
    },
};

// ─── Dense Layer ───

class DenseLayer {
    /**
     * @param {number} inputSize
     * @param {number} outputSize
     * @param {string} activation - 'relu' | 'sigmoid' | 'tanh' | 'linear'
     */
    constructor(inputSize, outputSize, activation = "relu") {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = ACTIVATIONS[activation] || ACTIVATIONS.relu;
        this.activationName = activation;

        // Xavier initialization
        const scale = Math.sqrt(2 / (inputSize + outputSize));
        this.weights = Array.from({ length: outputSize }, () =>
            Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * scale)
        );
        this.biases = new Array(outputSize).fill(0);

        // Gradient accumulators
        this.weightGrads = null;
        this.biasGrads = null;

        // Adam optimizer state
        this.mWeights = Array.from({ length: outputSize }, () => new Array(inputSize).fill(0));
        this.vWeights = Array.from({ length: outputSize }, () => new Array(inputSize).fill(0));
        this.mBiases = new Array(outputSize).fill(0);
        this.vBiases = new Array(outputSize).fill(0);

        // Cache for backprop
        this.lastInput = null;
        this.lastPreActivation = null;
        this.lastOutput = null;
    }

    /**
     * Forward pass
     * @param {number[]} input
     * @returns {number[]} output
     */
    forward(input) {
        this.lastInput = [...input];
        const preAct = new Array(this.outputSize);
        const output = new Array(this.outputSize);

        for (let j = 0; j < this.outputSize; j++) {
            let sum = this.biases[j];
            for (let i = 0; i < this.inputSize; i++) {
                sum += this.weights[j][i] * input[i];
            }
            preAct[j] = sum;
            output[j] = this.activation.fn(sum);
        }

        this.lastPreActivation = preAct;
        this.lastOutput = output;
        return output;
    }

    /**
     * Backward pass
     * @param {number[]} outputGrad - gradient of loss w.r.t. layer output
     * @returns {number[]} inputGrad - gradient of loss w.r.t. layer input
     */
    backward(outputGrad) {
        const inputGrad = new Array(this.inputSize).fill(0);
        this.weightGrads = Array.from({ length: this.outputSize }, () =>
            new Array(this.inputSize).fill(0)
        );
        this.biasGrads = new Array(this.outputSize).fill(0);

        for (let j = 0; j < this.outputSize; j++) {
            const dAct = this.activation.deriv(this.lastPreActivation[j]);
            const delta = outputGrad[j] * dAct;

            this.biasGrads[j] = delta;
            for (let i = 0; i < this.inputSize; i++) {
                this.weightGrads[j][i] = delta * this.lastInput[i];
                inputGrad[i] += delta * this.weights[j][i];
            }
        }

        return inputGrad;
    }

    /**
     * Update weights using Adam optimizer
     * @param {number} lr - learning rate
     * @param {number} t - timestep (for bias correction)
     * @param {number} beta1 - first moment decay (default 0.9)
     * @param {number} beta2 - second moment decay (default 0.999)
     */
    updateWeights(lr, t = 1, beta1 = 0.9, beta2 = 0.999) {
        const epsilon = 1e-8;
        const bc1 = 1 - Math.pow(beta1, t);
        const bc2 = 1 - Math.pow(beta2, t);

        for (let j = 0; j < this.outputSize; j++) {
            // Bias update
            this.mBiases[j] = beta1 * this.mBiases[j] + (1 - beta1) * this.biasGrads[j];
            this.vBiases[j] = beta2 * this.vBiases[j] + (1 - beta2) * this.biasGrads[j] ** 2;
            const mHatB = this.mBiases[j] / bc1;
            const vHatB = this.vBiases[j] / bc2;
            this.biases[j] -= lr * mHatB / (Math.sqrt(vHatB) + epsilon);

            // Weight update
            for (let i = 0; i < this.inputSize; i++) {
                this.mWeights[j][i] = beta1 * this.mWeights[j][i] + (1 - beta1) * this.weightGrads[j][i];
                this.vWeights[j][i] = beta2 * this.vWeights[j][i] + (1 - beta2) * this.weightGrads[j][i] ** 2;
                const mHatW = this.mWeights[j][i] / bc1;
                const vHatW = this.vWeights[j][i] / bc2;
                this.weights[j][i] -= lr * mHatW / (Math.sqrt(vHatW) + epsilon);
            }
        }
    }

    /**
     * Copy weights from another layer (for target network)
     * @param {DenseLayer} source
     */
    copyFrom(source) {
        for (let j = 0; j < this.outputSize; j++) {
            this.biases[j] = source.biases[j];
            for (let i = 0; i < this.inputSize; i++) {
                this.weights[j][i] = source.weights[j][i];
            }
        }
    }

    /**
     * Serialize layer to plain object
     */
    serialize() {
        return {
            inputSize: this.inputSize,
            outputSize: this.outputSize,
            activation: this.activationName,
            weights: this.weights.map((row) => [...row]),
            biases: [...this.biases],
        };
    }

    /**
     * Restore from serialized data
     */
    static deserialize(data) {
        const layer = new DenseLayer(data.inputSize, data.outputSize, data.activation);
        layer.weights = data.weights.map((row) => [...row]);
        layer.biases = [...data.biases];
        return layer;
    }
}

// ─── Neural Network ───

export class NeuralNetwork {
    /**
     * @param {Array<{size: number, activation: string}>} layerConfigs
     *   First element is input size (activation ignored).
     *   Example: [{size: 7}, {size: 64, activation: 'relu'}, {size: 32, activation: 'relu'}, {size: 2, activation: 'linear'}]
     */
    constructor(layerConfigs) {
        this.layers = [];
        this.trainStep = 0;

        for (let i = 1; i < layerConfigs.length; i++) {
            this.layers.push(
                new DenseLayer(
                    layerConfigs[i - 1].size,
                    layerConfigs[i].size,
                    layerConfigs[i].activation || "relu"
                )
            );
        }
    }

    /**
     * Forward pass through all layers
     * @param {number[]} input
     * @returns {number[]} output
     */
    forward(input) {
        let x = input;
        for (const layer of this.layers) {
            x = layer.forward(x);
        }
        return x;
    }

    /**
     * Train on a single (input, target) pair using MSE loss
     * @param {number[]} input
     * @param {number[]} target
     * @param {number} lr - learning rate
     * @returns {number} loss
     */
    trainOnSample(input, target, lr = 0.001) {
        this.trainStep++;
        const output = this.forward(input);

        // MSE loss gradient: 2 * (output - target) / n
        const n = output.length;
        const lossGrad = output.map((o, i) => (2 * (o - target[i])) / n);

        // Compute loss for monitoring
        const loss = output.reduce((sum, o, i) => sum + (o - target[i]) ** 2, 0) / n;

        // Backprop through layers in reverse
        let grad = lossGrad;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);
        }

        // Update weights
        for (const layer of this.layers) {
            layer.updateWeights(lr, this.trainStep);
        }

        return loss;
    }

    /**
     * Train on a batch using Huber loss (more robust for Q-learning)
     * @param {Array<{input: number[], target: number[]}>} batch
     * @param {number} lr
     * @param {number} delta - Huber loss threshold
     * @returns {number} average loss
     */
    trainOnBatch(batch, lr = 0.001, delta = 1.0) {
        this.trainStep++;
        let totalLoss = 0;

        // Accumulate gradients across batch
        const batchGrads = this.layers.map((layer) => ({
            weights: Array.from({ length: layer.outputSize }, () =>
                new Array(layer.inputSize).fill(0)
            ),
            biases: new Array(layer.outputSize).fill(0),
        }));

        for (const { input, target } of batch) {
            const output = this.forward(input);

            // Huber loss gradient
            const lossGrad = output.map((o, i) => {
                const err = o - target[i];
                if (Math.abs(err) <= delta) {
                    return err / output.length;
                } else {
                    return (delta * Math.sign(err)) / output.length;
                }
            });

            // Huber loss value
            const loss = output.reduce((sum, o, i) => {
                const err = Math.abs(o - target[i]);
                return sum + (err <= delta ? 0.5 * err ** 2 : delta * (err - 0.5 * delta));
            }, 0) / output.length;
            totalLoss += loss;

            // Backprop
            let grad = lossGrad;
            for (let l = this.layers.length - 1; l >= 0; l--) {
                grad = this.layers[l].backward(grad);

                // Accumulate gradients
                for (let j = 0; j < this.layers[l].outputSize; j++) {
                    batchGrads[l].biases[j] += this.layers[l].biasGrads[j];
                    for (let i = 0; i < this.layers[l].inputSize; i++) {
                        batchGrads[l].weights[j][i] += this.layers[l].weightGrads[j][i];
                    }
                }
            }
        }

        // Average gradients and apply
        const batchSize = batch.length;
        for (let l = 0; l < this.layers.length; l++) {
            for (let j = 0; j < this.layers[l].outputSize; j++) {
                this.layers[l].biasGrads = batchGrads[l].biases.map((g) => g / batchSize);
                this.layers[l].weightGrads = batchGrads[l].weights.map((row) =>
                    row.map((g) => g / batchSize)
                );
            }
            this.layers[l].updateWeights(lr, this.trainStep);
        }

        return totalLoss / batchSize;
    }

    /**
     * Copy all weights from another network (for target network sync)
     * @param {NeuralNetwork} source
     */
    copyFrom(source) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].copyFrom(source.layers[i]);
        }
    }

    /**
     * Soft update: θ_target ← τ * θ_online + (1 - τ) * θ_target
     * @param {NeuralNetwork} source - online network
     * @param {number} tau - interpolation factor (e.g. 0.005)
     */
    softUpdate(source, tau = 0.005) {
        for (let l = 0; l < this.layers.length; l++) {
            const srcLayer = source.layers[l];
            const tgtLayer = this.layers[l];
            for (let j = 0; j < tgtLayer.outputSize; j++) {
                tgtLayer.biases[j] = tau * srcLayer.biases[j] + (1 - tau) * tgtLayer.biases[j];
                for (let i = 0; i < tgtLayer.inputSize; i++) {
                    tgtLayer.weights[j][i] =
                        tau * srcLayer.weights[j][i] + (1 - tau) * tgtLayer.weights[j][i];
                }
            }
        }
    }

    /**
     * Serialize to JSON-safe object
     */
    serialize() {
        return {
            layers: this.layers.map((l) => l.serialize()),
            trainStep: this.trainStep,
        };
    }

    /**
     * Restore from serialized data
     */
    static deserialize(data) {
        const configs = [{ size: data.layers[0].inputSize }];
        for (const l of data.layers) {
            configs.push({ size: l.outputSize, activation: l.activation });
        }
        const net = new NeuralNetwork(configs);
        for (let i = 0; i < net.layers.length; i++) {
            net.layers[i].weights = data.layers[i].weights.map((r) => [...r]);
            net.layers[i].biases = [...data.layers[i].biases];
        }
        net.trainStep = data.trainStep || 0;
        return net;
    }
}
