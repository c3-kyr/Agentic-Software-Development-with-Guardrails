/**
 * Threshold Optimizer
 *
 * Sweeps threshold from 0 → 1 and computes:
 * 1. Interrupt Frequency vs Threshold
 * 2. Error Rate vs Threshold
 * 3. Net Utility = Quality - λ * HumanEffort
 */

import { computeCompositeUncertainty } from "../uncertainty/compositeUncertainty.js";
import { computeGravity } from "../gravity/gravityCalc.js";
import { evaluateTrigger } from "../trigger/interventionTrigger.js";

/**
 * Evaluate all scenarios at a given threshold
 *
 * @param {Object[]} scenarios - Array of scenario objects
 * @param {number} threshold - Current threshold
 * @param {number[]} weights - Uncertainty weights [w1, w2, w3]
 * @returns {Object} Evaluation results with confusion matrix
 */
export function evaluateAtThreshold(scenarios, threshold, weights) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    const results = [];

    for (const scenario of scenarios) {
        const uncertainty = computeCompositeUncertainty(scenario, weights);
        const gravity = computeGravity(scenario.impact, scenario.irreversibility, scenario.propagation);
        const trigger = evaluateTrigger(uncertainty.U, gravity.G, threshold);

        const predicted = trigger.interrupt;
        const actual = scenario.groundTruthNeedsHuman;

        if (predicted && actual) tp++;
        else if (!predicted && !actual) tn++;
        else if (predicted && !actual) fp++;
        else fn++;

        results.push({
            scenario,
            uncertainty,
            gravity,
            trigger,
            predicted,
            actual,
            classification: predicted && actual ? "TP" : !predicted && !actual ? "TN" : predicted && !actual ? "FP" : "FN",
        });
    }

    const total = scenarios.length;
    const interruptFreq = (tp + fp) / total;
    const errorRate = fn / total; // Missed interventions
    const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
        threshold,
        confusionMatrix: { tp, tn, fp, fn },
        interruptFreq,
        errorRate,
        precision,
        recall,
        f1,
        results,
    };
}

/**
 * Sweep threshold and generate curve data
 *
 * @param {Object[]} scenarios - Array of scenarios
 * @param {number[]} weights - Uncertainty weights
 * @param {number} lambda - Human cost weight for utility function
 * @param {number} steps - Number of threshold steps (default 100)
 * @returns {{ curveData: Object[], optimalThreshold: number, optimalUtility: number }}
 */
export function sweepThreshold(scenarios, weights, lambda = 0.5, steps = 100) {
    const curveData = [];
    let optimalThreshold = 0;
    let optimalUtility = -Infinity;

    for (let i = 0; i <= steps; i++) {
        const threshold = i / steps;
        const eval_ = evaluateAtThreshold(scenarios, threshold, weights);

        const quality = 1 - eval_.errorRate;
        const humanEffort = eval_.interruptFreq;
        const utility = quality - lambda * humanEffort;

        const point = {
            threshold,
            interruptFreq: eval_.interruptFreq,
            errorRate: eval_.errorRate,
            utility,
            quality,
            humanEffort,
            precision: eval_.precision,
            recall: eval_.recall,
            f1: eval_.f1,
            confusionMatrix: eval_.confusionMatrix,
        };

        curveData.push(point);

        if (utility > optimalUtility) {
            optimalUtility = utility;
            optimalThreshold = threshold;
        }
    }

    return {
        curveData,
        optimalThreshold,
        optimalUtility,
    };
}
