import { RLTrainer } from '../rl/trainer.js';
import { Chart } from 'chart.js';

export function initRLDashboard(updateResearchParamsCallback) {
    const $ = (sel) => document.querySelector(sel);

    // Controls
    const btnStart = $("#rl-btn-start");
    const btnPause = $("#rl-btn-pause");
    const btnReset = $("#rl-btn-reset");
    const btnApply = $("#rl-btn-apply");
    const statusText = $("#rl-progress-text");

    // Hyperparameters
    const sliders = {
        episodes: $("#rl-episodes"),
        batchSize: $("#rl-batch-size"),
        lr: $("#rl-lr"),
        gamma: $("#rl-gamma"),
        epsilonDecay: $("#rl-epsilon-decay")
    };
    
    // Results output
    const res = {
        threshold: $("#rl-metric-tau"),
        w1: $("#rl-metric-w1"),
        w2: $("#rl-metric-w2"),
        w3: $("#rl-metric-w3"),
        lambda: $("#rl-metric-lambda") // wait, is there rl-metric-lambda?
    };

    // Update slider displays no longer needed for number inputs
    function updateSliderDisplays() {
        // no-op, number inputs display themselves
    }

    // Chart configs
    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 }, // no animation for performance
        plugins: { legend: { display: true, labels: { color: '#94a3b8' } } },
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
        }
    };

    const charts = {
        reward: new Chart($("#chart-rl-reward"), {
            type: 'line',
            data: { datasets: [{ label: 'Reward', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
            options: { ...commonOpts, plugins: { legend: { display: false } } }
        }),
        f1: new Chart($("#chart-rl-f1"), {
            type: 'line',
            data: { datasets: [{ label: 'F1 Score', data: [], borderColor: '#6366f1', borderWidth: 2, tension: 0.3, pointRadius: 0 }] },
            options: { ...commonOpts, plugins: { legend: { display: false } }, scales: { ...commonOpts.scales, y: { ...commonOpts.scales.y, min: 0, max: 1 } } }
        }),
        lossAndEpsilon: new Chart($("#chart-rl-loss"), {
            type: 'line',
            data: { datasets: [
                { label: 'Epsilon', data: [], borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0, yAxisID: 'y' },
                { label: 'Loss', data: [], borderColor: '#ef4444', borderWidth: 1, borderDash: [5, 5], pointRadius: 0, yAxisID: 'y1' }
            ]},
            options: { ...commonOpts, scales: {
                x: commonOpts.scales.x,
                y: { type: 'linear', display: true, position: 'left', min: 0, max: 1, ticks: { color: '#64748b' } },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#64748b' } }
            }}
        }),
        params: new Chart($("#chart-rl-params"), {
            type: 'line',
            data: { datasets: [
                { label: 'w1', data: [], borderColor: '#3b82f6', borderWidth: 2, pointRadius: 0 },
                { label: 'w2', data: [], borderColor: '#8b5cf6', borderWidth: 2, pointRadius: 0 },
                { label: 'w3', data: [], borderColor: '#ec4899', borderWidth: 2, pointRadius: 0 },
                { label: 'λ', data: [], borderColor: '#f59e0b', borderWidth: 2, borderDash: [5, 5], pointRadius: 0 }
            ]},
            options: { ...commonOpts, scales: { ...commonOpts.scales, y: { ...commonOpts.scales.y, min: 0, max: 1 } } }
        })
    };

    const liveMetrics = {
        confidence: $("#rl-metric-confidence"),
        f1: $("#rl-metric-f1"),
        reward: $("#rl-metric-reward"),
        epsilon: $("#rl-metric-epsilon"),
        tp: $("#rl-cm-tp"),
        fn: $("#rl-cm-fn"),
        fp: $("#rl-cm-fp"),
        tn: $("#rl-cm-tn")
    };

    let trainer = null;
    let finalOptimizedParams = null;

    function resetCharts() {
        Object.values(charts).forEach(chart => {
            chart.data.datasets.forEach(ds => ds.data = []);
            chart.update();
        });
        res.threshold.textContent = "—";
        res.w1.textContent = "—";
        res.w2.textContent = "—";
        res.w3.textContent = "—";
        res.lambda.textContent = "—";
        
        liveMetrics.confidence.textContent = "—";
        liveMetrics.f1.textContent = "—";
        liveMetrics.reward.textContent = "—";
        liveMetrics.epsilon.textContent = "—";
        liveMetrics.tp.textContent = "0";
        liveMetrics.fn.textContent = "0";
        liveMetrics.fp.textContent = "0";
        liveMetrics.tn.textContent = "0";
        
        btnApply.disabled = true;
    }

    function initTrainer() {
        if (trainer) trainer.stop();

        trainer = new RLTrainer({
            totalEpisodes: parseInt(sliders.episodes.value),
            paramUpdateEvery: 20,
            logEvery: 5,
            agentConfig: {
                batchSize: parseInt(sliders.batchSize.value),
                lr: parseFloat(sliders.lr.value),
                gamma: parseFloat(sliders.gamma.value),
                epsilonDecay: parseFloat(sliders.epsilonDecay.value)
            },
            onEpisode: (data) => {
                const x = data.episode;
                charts.reward.data.datasets[0].data.push({x, y: data.totalReward});
                charts.f1.data.datasets[0].data.push({x, y: data.f1});
                charts.lossAndEpsilon.data.datasets[0].data.push({x, y: data.epsilon});
                charts.lossAndEpsilon.data.datasets[1].data.push({x, y: data.avgLoss});
                
                charts.params.data.datasets[0].data.push({x, y: data.weights[0]});
                charts.params.data.datasets[1].data.push({x, y: data.weights[1]});
                charts.params.data.datasets[2].data.push({x, y: data.weights[2]});
                charts.params.data.datasets[3].data.push({x, y: data.lambda});
                
                if (data.episode % 5 === 0) {
                    liveMetrics.confidence.textContent = data.thresholdConfidence ? data.thresholdConfidence.toFixed(2) : "—";
                    liveMetrics.f1.textContent = data.f1 ? data.f1.toFixed(3) : "0.000";
                    liveMetrics.reward.textContent = data.totalReward ? data.totalReward.toFixed(2) : "0.00";
                    liveMetrics.epsilon.textContent = data.epsilon ? data.epsilon.toFixed(3) : "—";
                    if (data.metrics) {
                        liveMetrics.tp.textContent = data.metrics.tp;
                        liveMetrics.fn.textContent = data.metrics.fn;
                        liveMetrics.fp.textContent = data.metrics.fp;
                        liveMetrics.tn.textContent = data.metrics.tn;
                    }
                }
            },
            onProgress: (data) => {
                const pct = (data.progress * 100).toFixed(1);
                statusText.textContent = `Training: Episode ${data.episode} / ${data.totalEpisodes} (${pct}%) — Avg Reward: ${data.avgReward.toFixed(2)}`;
                
                // Redraw charts periodically
                Object.values(charts).forEach(c => c.update());

                // Update current estimates
                res.threshold.textContent = data.learnedThreshold.toFixed(2);
                res.w1.textContent = data.currentWeights[0].toFixed(2);
                res.w2.textContent = data.currentWeights[1].toFixed(2);
                res.w3.textContent = data.currentWeights[2].toFixed(2);
                res.lambda.textContent = data.currentLambda.toFixed(2);
            },
            onComplete: (results) => {
                statusText.textContent = `Training Complete! Avg F1: ${results.performance.avgF1.toFixed(3)}`;
                btnStart.disabled = false;
                btnPause.disabled = true;
                btnStart.textContent = "▶ Retrain";
                btnApply.disabled = false;
                finalOptimizedParams = results;

                Object.values(charts).forEach(c => c.update());
            }
        });
    }

    btnStart.addEventListener('click', () => {
        if (!trainer || (!trainer.isRunning && !trainer.isPaused)) {
            resetCharts();
            initTrainer();
        }
        trainer.start();
        btnStart.disabled = true;
        btnPause.disabled = false;
        btnReset.disabled = false;
        statusText.textContent = "Training...";
    });

    btnPause.addEventListener('click', () => {
        if (trainer) trainer.pause();
        btnStart.disabled = false;
        btnPause.disabled = true;
        btnStart.textContent = "▶ Resume";
        statusText.textContent = "Paused";
    });

    btnReset.addEventListener('click', () => {
        if (trainer) trainer.stop();
        trainer = null;
        resetCharts();
        btnStart.disabled = false;
        btnPause.disabled = true;
        btnReset.disabled = true;
        btnStart.textContent = "▶ Start Training";
        statusText.textContent = "Ready to train";
    });

    btnApply.addEventListener('click', () => {
        if (finalOptimizedParams && updateResearchParamsCallback) {
            updateResearchParamsCallback(finalOptimizedParams);
            document.querySelector('#tab-research').click(); // Switch tab
        }
    });
}
