/**
 * RL Training Dashboard Panel
 *
 * Wires the DQN trainer to real-time Chart.js visualizations.
 * Runs entirely on pre-computed scenario data (Experience Buffer)
 * so training completes in seconds, not hours.
 *
 * Key charts:
 *   1. Reward curve (rolling average)
 *   2. Learned τ evolution
 *   3. Weight [w1, w2, w3] evolution
 *   4. F1 / Precision / Recall
 *   5. Live confusion matrix
 */

import { Chart } from "chart.js";
import { RLTrainer } from "../rl/trainer.js";

// ─── State ───
let trainer = null;
let charts = {};
let speedMs = 16; // ms between episodes (lower = faster)
let episodeHistory = [];
let lastTrainingResults = null; // Stored separately to avoid massive JSON in DOM

// ─── Chart Color Palette ───
const COLORS = {
    reward:    { border: "#10b981", bg: "rgba(16, 185, 129, 0.12)" },
    threshold: { border: "#f59e0b", bg: "rgba(245, 158, 11, 0.12)" },
    f1:        { border: "#6366f1", bg: "rgba(99, 102, 241, 0.12)" },
    precision: { border: "#06b6d4", bg: "rgba(6, 182, 212, 0.10)" },
    recall:    { border: "#a78bfa", bg: "rgba(167, 139, 250, 0.10)" },
    w1:        { border: "#ef4444", bg: "rgba(239, 68, 68, 0.10)" },
    w2:        { border: "#3b82f6", bg: "rgba(59, 130, 246, 0.10)" },
    w3:        { border: "#8b5cf6", bg: "rgba(139, 92, 246, 0.10)" },
    epsilon:   { border: "#94a3b8", bg: "rgba(148, 163, 184, 0.10)" },
};

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
        legend: {
            display: true,
            labels: {
                color: "#94a3b8",
                font: { family: "Inter", size: 10 },
                usePointStyle: true,
                boxWidth: 6,
                padding: 10,
            },
        },
        tooltip: {
            backgroundColor: "rgba(17, 24, 39, 0.95)",
            titleColor: "#f1f5f9",
            bodyColor: "#94a3b8",
            titleFont: { family: "Inter", weight: "600", size: 11 },
            bodyFont: { family: "JetBrains Mono", size: 10 },
            borderColor: "rgba(99, 102, 241, 0.2)",
            borderWidth: 1,
            padding: 8,
            cornerRadius: 6,
        },
    },
    scales: {
        x: {
            title: { display: true, text: "Episode", color: "#64748b", font: { family: "Inter", size: 10 } },
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, maxTicksLimit: 10 },
        },
        y: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } },
        },
    },
};

// ─── Public Init ───

let onApplyCallback = null;

export function initRLPanel(onApply) {
    onApplyCallback = onApply;
    initCharts();
    setupListeners();
    updateLiveMetrics(null);
}

// ─── Chart Initialization ───

function initCharts() {
    // 1. Reward Curve
    charts.reward = new Chart(document.getElementById("rl-chart-reward"), {
        type: "line",
        data: {
            datasets: [
                {
                    label: "Episode Reward",
                    data: [],
                    borderColor: COLORS.reward.border,
                    backgroundColor: COLORS.reward.bg,
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "Rolling Avg (50)",
                    data: [],
                    borderColor: "#f1f5f9",
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: { ...chartDefaults.scales.y, title: { display: true, text: "Reward", color: "#64748b", font: { family: "Inter", size: 10 } } },
            },
        },
    });

    // 2. Learned τ
    charts.threshold = new Chart(document.getElementById("rl-chart-threshold"), {
        type: "line",
        data: {
            datasets: [
                {
                    label: "Learned τ",
                    data: [],
                    borderColor: COLORS.threshold.border,
                    backgroundColor: COLORS.threshold.bg,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "Initial τ (0.15)",
                    data: [],
                    borderColor: "rgba(255,255,255,0.2)",
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    min: 0, max: 0.5,
                    title: { display: true, text: "Threshold (τ)", color: "#64748b", font: { family: "Inter", size: 10 } },
                },
            },
        },
    });

    // 3. F1 / Precision / Recall
    charts.f1 = new Chart(document.getElementById("rl-chart-f1"), {
        type: "line",
        data: {
            datasets: [
                {
                    label: "F1",
                    data: [],
                    borderColor: COLORS.f1.border,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "Precision",
                    data: [],
                    borderColor: COLORS.precision.border,
                    borderWidth: 1.5,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "Recall",
                    data: [],
                    borderColor: COLORS.recall.border,
                    borderWidth: 1.5,
                    tension: 0.3,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    min: 0, max: 1,
                    title: { display: true, text: "Score", color: "#64748b", font: { family: "Inter", size: 10 } },
                },
            },
        },
    });

    // 4. Weight Evolution
    charts.weights = new Chart(document.getElementById("rl-chart-weights"), {
        type: "line",
        data: {
            datasets: [
                {
                    label: "w₁ Semantic",
                    data: [],
                    borderColor: COLORS.w1.border,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "w₂ Entropy",
                    data: [],
                    borderColor: COLORS.w2.border,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "w₃ Reflection",
                    data: [],
                    borderColor: COLORS.w3.border,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    min: 0, max: 0.7,
                    title: { display: true, text: "Weight", color: "#64748b", font: { family: "Inter", size: 10 } },
                },
            },
        },
    });
}

// ─── Trainer Callbacks ───

function onEpisode(data) {
    episodeHistory.push(data);
    const ep = data.episode;

    // Reward chart
    charts.reward.data.datasets[0].data.push({ x: ep, y: data.totalReward });
    // Rolling average
    const recent = episodeHistory.slice(-50);
    const avg = recent.reduce((s, e) => s + e.totalReward, 0) / recent.length;
    charts.reward.data.datasets[1].data.push({ x: ep, y: avg });

    // Threshold chart
    charts.threshold.data.datasets[0].data.push({ x: ep, y: data.learnedThreshold });
    // Baseline reference line
    if (charts.threshold.data.datasets[1].data.length === 0) {
        charts.threshold.data.datasets[1].data.push({ x: 0, y: 0.15 });
    }
    charts.threshold.data.datasets[1].data = [
        { x: 0, y: 0.15 },
        { x: ep, y: 0.15 },
    ];

    // F1 chart
    charts.f1.data.datasets[0].data.push({ x: ep, y: data.f1 });
    charts.f1.data.datasets[1].data.push({ x: ep, y: data.precision });
    charts.f1.data.datasets[2].data.push({ x: ep, y: data.recall });

    // Weights chart
    if (data.weights && data.weights.length >= 3) {
        charts.weights.data.datasets[0].data.push({ x: ep, y: data.weights[0] });
        charts.weights.data.datasets[1].data.push({ x: ep, y: data.weights[1] });
        charts.weights.data.datasets[2].data.push({ x: ep, y: data.weights[2] });
    }

    // Update all charts every 5 episodes (perf optimization)
    if (ep % 5 === 0 || ep <= 10) {
        Object.values(charts).forEach(c => c.update("none"));
    }

    // Update live metrics
    updateLiveMetrics(data);

    // Update confusion matrix
    if (data.metrics) {
        updateConfusionMatrix(data.metrics);
    }

    // Update progress bar
    const totalEp = trainer?.totalEpisodes || 500;
    const pct = Math.round((ep / totalEp) * 100);
    const bar = document.getElementById("rl-progress-bar");
    const txt = document.getElementById("rl-progress-text");
    if (bar) bar.style.width = `${pct}%`;
    if (txt) txt.textContent = `Episode ${ep} / ${totalEp}`;
}

function onProgress(data) {
    // Already handled per-episode
}

function onParamUpdate(data) {
    const pEl = document.getElementById("rl-param-status");
    if (pEl) {
        const w = data.params.weights;
        pEl.textContent = `Param update #${data.step}: W=[${w.map(v => v.toFixed(3)).join(", ")}]`;
    }
}

function onComplete(results) {
    // Final chart update
    Object.values(charts).forEach(c => c.update());

    updateLiveMetrics({
        learnedThreshold: results.learnedThreshold,
        thresholdConfidence: results.thresholdConfidence,
        f1: results.performance.avgF1,
        precision: results.performance.avgPrecision,
        recall: results.performance.avgRecall,
        epsilon: 0,
        weights: results.optimizedWeights,
        totalReward: results.performance.avgReward,
    });

    // Show final result card
    const resultEl = document.getElementById("rl-final-result");
    if (resultEl) {
        resultEl.classList.remove("hidden");

        // Build reward model section if token-cost mode
        let rewardModelHtml = "";
        if (results.rewardModel && results.rewardModel.mode === "token-cost") {
            const p = results.rewardModel.params;
            rewardModelHtml = `
                <div class="rl-result-reward-model">
                    <div class="rl-reward-model-title">📐 Token-Cost Reward Model</div>
                    <div class="rl-reward-model-params">
                        <div class="rl-reward-param">T<sub>min</sub> = ${Math.round(p.T_min)}</div>
                        <div class="rl-reward-param">T<sub>max</sub> = ${Math.round(p.T_max)}</div>
                        <div class="rl-reward-param">T<sub>review</sub> = ${Math.round(p.T_review)}</div>
                        <div class="rl-reward-param">α = ${p.alpha}</div>
                        <div class="rl-reward-param">η = ${p.eta}</div>
                        <div class="rl-reward-param">T<sub>norm</sub> = ${Math.round(p.T_norm)}</div>
                    </div>
                    <div class="rl-reward-model-formulas">
                        <code>R<sub>TP</sub> = +T<sub>base</sub> / T<sub>norm</sub></code>
                        <code>R<sub>TN</sub> = +T<sub>base</sub>·η / T<sub>norm</sub></code>
                        <code>R<sub>FP</sub> = −T<sub>review</sub> / T<sub>norm</sub></code>
                        <code>R<sub>FN</sub> = −T<sub>base</sub>·(1+κ) / T<sub>norm</sub></code>
                    </div>
                </div>
            `;
        }

        resultEl.innerHTML = `
            <div class="rl-result-header">🎯 Training Complete — ${results.totalEpisodes} Episodes</div>
            <div class="rl-result-grid">
                <div class="rl-result-item">
                    <span class="rl-result-label">Learned τ</span>
                    <span class="rl-result-value accent">${results.learnedThreshold.toFixed(3)}</span>
                </div>
                <div class="rl-result-item">
                    <span class="rl-result-label">Confidence</span>
                    <span class="rl-result-value">${(results.thresholdConfidence * 100).toFixed(1)}%</span>
                </div>
                <div class="rl-result-item">
                    <span class="rl-result-label">Best F1</span>
                    <span class="rl-result-value">${results.performance.bestF1.toFixed(3)}</span>
                </div>
                <div class="rl-result-item">
                    <span class="rl-result-label">Avg Reward</span>
                    <span class="rl-result-value">${results.performance.avgReward.toFixed(2)}</span>
                </div>
                <div class="rl-result-item">
                    <span class="rl-result-label">Optimized W</span>
                    <span class="rl-result-value">[${results.optimizedWeights.map(w => w.toFixed(2)).join(", ")}]</span>
                </div>
                <div class="rl-result-item">
                    <span class="rl-result-label">Converged</span>
                    <span class="rl-result-value">${results.converged ? "✅ Yes" : "❌ No"}</span>
                </div>
            </div>
            ${rewardModelHtml}
            <div class="rl-result-comparison">
                <div class="rl-compare-col">
                    <span class="rl-compare-title">Initial</span>
                    <div>τ = 0.15</div>
                    <div>W = [0.40, 0.35, 0.25]</div>
                </div>
                <div class="rl-compare-arrow">→</div>
                <div class="rl-compare-col learned">
                    <span class="rl-compare-title">Learned</span>
                    <div>τ = ${results.learnedThreshold.toFixed(3)}</div>
                    <div>W = [${results.optimizedWeights.map(w => w.toFixed(2)).join(", ")}]</div>
                </div>
            </div>
        `;
    }

    // Enable Apply button and store results in module scope
    lastTrainingResults = results;
    const btnApply = document.getElementById("rl-btn-apply");
    if (btnApply) {
        btnApply.disabled = false;
    }

    // Re-enable buttons
    setButtonStates("stopped");
}

// ─── Live Metrics ───

function updateLiveMetrics(data) {
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };

    if (!data) {
        set("rl-metric-tau", "—");
        set("rl-metric-f1", "—");
        set("rl-metric-reward", "—");
        set("rl-metric-epsilon", "—");
        set("rl-metric-w1", "—");
        set("rl-metric-w2", "—");
        set("rl-metric-w3", "—");
        set("rl-metric-confidence", "—");
        return;
    }

    set("rl-metric-tau", data.learnedThreshold?.toFixed(3) ?? "—");
    set("rl-metric-f1", data.f1?.toFixed(3) ?? "—");
    set("rl-metric-reward", data.totalReward?.toFixed(2) ?? "—");
    set("rl-metric-epsilon", data.epsilon?.toFixed(3) ?? "—");
    set("rl-metric-confidence", data.thresholdConfidence ? `${(data.thresholdConfidence * 100).toFixed(0)}%` : "—");

    if (data.weights && data.weights.length >= 3) {
        set("rl-metric-w1", data.weights[0].toFixed(3));
        set("rl-metric-w2", data.weights[1].toFixed(3));
        set("rl-metric-w3", data.weights[2].toFixed(3));
    }
}

function updateConfusionMatrix(m) {
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    set("rl-cm-tp", m.tp);
    set("rl-cm-tn", m.tn);
    set("rl-cm-fp", m.fp);
    set("rl-cm-fn", m.fn);
}

// ─── Button States ───

function setButtonStates(state) {
    const btnStart = document.getElementById("rl-btn-start");
    const btnPause = document.getElementById("rl-btn-pause");
    const btnReset = document.getElementById("rl-btn-reset");

    if (!btnStart) return;

    switch (state) {
        case "running":
            btnStart.disabled = true;
            btnStart.textContent = "⏳ Training...";
            btnPause.disabled = false;
            btnReset.disabled = true;
            break;
        case "paused":
            btnStart.disabled = false;
            btnStart.textContent = "▶ Resume";
            btnPause.disabled = true;
            btnReset.disabled = false;
            break;
        case "stopped":
        default:
            btnStart.disabled = false;
            btnStart.textContent = "▶ Start Training";
            btnPause.disabled = true;
            btnReset.disabled = false;
            break;
    }
}

// ─── Event Listeners ───

function setupListeners() {
    // Start / Resume
    document.getElementById("rl-btn-start")?.addEventListener("click", async () => {
        if (!trainer || trainer.status === "stopped") {
            // Fresh start
            resetChartData();
            episodeHistory = [];
            document.getElementById("rl-final-result")?.classList.add("hidden");

            const episodes = parseInt(document.getElementById("rl-episodes")?.value || "500");
            const lr = parseFloat(document.getElementById("rl-lr")?.value || "0.001");
            const gamma = parseFloat(document.getElementById("rl-gamma")?.value || "0.99");
            const epsilonDecay = parseFloat(document.getElementById("rl-epsilon-decay")?.value || "0.995");

            // Try to fetch empirical token params from telemetry
            let tokenCostParams = {};
            try {
                const res = await fetch('http://localhost:3001/api/telemetry/parameters');
                const data = await res.json();
                if (data.status === 'success') {
                    tokenCostParams = data.params;
                    console.log("Loaded empirical token parameters from telemetry:", tokenCostParams);
                }
            } catch (err) {
                console.warn("Could not fetch empirical telemetry. Using default token model parameters.", err);
            }

            trainer = new RLTrainer({
                totalEpisodes: episodes,
                paramUpdateEvery: 25,
                logEvery: 5,
                scenarioMode: "mixed",
                randomCount: 50,
                initialWeights: [0.4, 0.35, 0.25],
                initialLambda: 0.5,
                agentConfig: { lr, gamma, epsilonDecay },
                tokenCostParams,
                onEpisode,
                onProgress,
                onParamUpdate,
                onComplete,
            });
        }

        setButtonStates("running");
        trainer.start();
    });

    // Pause
    document.getElementById("rl-btn-pause")?.addEventListener("click", () => {
        if (trainer) {
            trainer.pause();
            setButtonStates("paused");
        }
    });

    // Reset
    document.getElementById("rl-btn-reset")?.addEventListener("click", () => {
        if (trainer) {
            trainer.stop();
            trainer = null;
        }
        resetChartData();
        episodeHistory = [];
        updateLiveMetrics(null);
        updateConfusionMatrix({ tp: 0, tn: 0, fp: 0, fn: 0 });
        document.getElementById("rl-final-result")?.classList.add("hidden");
        document.getElementById("rl-progress-bar").style.width = "0%";
        document.getElementById("rl-progress-text").textContent = "Ready";
        document.getElementById("rl-param-status").textContent = "";
        setButtonStates("stopped");
    });

    // Speed slider
    document.getElementById("rl-speed")?.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        // Map 1-100 to 100ms-0ms (inverted: higher = faster)
        speedMs = Math.max(0, 100 - val);
        const label = document.getElementById("rl-speed-val");
        if (label) label.textContent = val >= 90 ? "Max" : `${val}%`;
    });

    // Apply to System
    document.getElementById("rl-btn-apply")?.addEventListener("click", () => {
        if (onApplyCallback && lastTrainingResults) {
            onApplyCallback(lastTrainingResults);
            document.getElementById("tab-research")?.click(); // Switch to research tab
        }
    });
}

function resetChartData() {
    Object.values(charts).forEach(c => {
        c.data.datasets.forEach(ds => { ds.data = []; });
        c.update("none");
    });
}
