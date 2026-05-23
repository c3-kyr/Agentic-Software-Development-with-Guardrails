import { PRESET_SCENARIOS } from './src/simulation/scenarios.js';
import { sweepThreshold } from './src/optimizer/thresholdOptimizer.js';

const weights = [0.4, 0.35, 0.25];
const lambda = 0.5;
const sweep = sweepThreshold(PRESET_SCENARIOS, weights, lambda, 100);
console.log(`Optimal T: ${sweep.optimalThreshold}, Utility: ${sweep.optimalUtility}`);
