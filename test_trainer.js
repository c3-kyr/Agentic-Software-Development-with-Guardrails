import { RLTrainer } from './src/rl/trainer.js';

async function test() {
    const trainer = new RLTrainer({
        totalEpisodes: 50,
        paramUpdateEvery: 10,
        logEvery: 5,
        scenarioMode: "preset",
        randomCount: 50,
        initialWeights: [0.4, 0.35, 0.25],
        initialLambda: 0.5,
        agentConfig: { lr: 0.01, gamma: 0.99, epsilonDecay: 0.9 },
    });
    
    await trainer.start();
    const results = trainer.getFinalResults();
    console.log('Optimized Weights:', results.optimizedWeights);
    console.log('Optimized Lambda:', results.optimizedLambda);
    console.log('Learned Threshold:', results.learnedThreshold);
}

test();
