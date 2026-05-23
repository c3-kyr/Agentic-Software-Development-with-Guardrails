/**
 * HITL Agentic Sprint — Main Application
 * Manages tab switching between Sprint Runner and Research Dashboard,
 * and initializes both panels.
 */

import { Chart, registerables } from "chart.js";
import { PRESET_SCENARIOS, generateRandomBatch } from "./simulation/scenarios.js";
import { computeCompositeUncertainty } from "./uncertainty/compositeUncertainty.js";
import { computeGravity } from "./gravity/gravityCalc.js";
import { evaluateTrigger } from "./trigger/interventionTrigger.js";
import { evaluateAtThreshold, sweepThreshold } from "./optimizer/thresholdOptimizer.js";
import { initSprintPanel } from "./ui/sprintPanel.js";
import { initRLPanel } from "./ui/rlPanel.js";
import "./styles/main.css";

Chart.register(...registerables);

// ─── State ───
let scenarios = [...PRESET_SCENARIOS];
let weights = [0.4, 0.35, 0.25];
let threshold = 0.15;
let lambda = 0.5;
let charts = {};
let currentEvalResults = null;

// ─── Tab Switching ───
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabId = btn.dataset.tab;

      // Toggle button states
      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Toggle content
      document.querySelectorAll(".tab-content").forEach((tc) => tc.classList.remove("active"));
      document.getElementById(`${tabId}-tab`)?.classList.add("active");
    });
  });
}

// ─── DOM References (Research Dashboard — prefixed with "r-") ───
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Research Dashboard sliders
const sliderSemantic = $("#r-weight-semantic");
const sliderEntropy = $("#r-weight-entropy");
const sliderReflection = $("#r-weight-reflection");
const sliderThreshold = $("#r-threshold-slider");
const sliderLambda = $("#lambda-slider");

const valSemantic = $("#r-weight-semantic-val");
const valEntropy = $("#r-weight-entropy-val");
const valReflection = $("#r-weight-reflection-val");
const valThreshold = $("#r-threshold-val");
const valLambda = $("#lambda-val");

// Buttons
const btnPreset = $("#btn-preset");
const btnRandom = $("#btn-random");
const btnMixed = $("#btn-mixed");

// Header Stats
const statScenarios = $("#stat-scenarios .stat-value");
const statOptimal = $("#stat-optimal .stat-value");
const statUtility = $("#stat-utility .stat-value");

// Confusion matrix
const cmTp = $("#cm-tp");
const cmTn = $("#cm-tn");
const cmFp = $("#cm-fp");
const cmFn = $("#cm-fn");
const metricPrecision = $("#metric-precision");
const metricRecall = $("#metric-recall");
const metricF1 = $("#metric-f1");

// Table
const scenarioTbody = $("#scenario-tbody");

// Modal
const modalOverlay = $("#modal-overlay");
const modalClose = $("#modal-close");

// ─── Chart Config ───
const chartColors = {
  interrupt: { border: "#6366f1", bg: "rgba(99, 102, 241, 0.15)" },
  error: { border: "#ef4444", bg: "rgba(239, 68, 68, 0.15)" },
  utility: { border: "#10b981", bg: "rgba(16, 185, 129, 0.15)" },
};

const commonScales = {
  x: {
    title: { display: true, text: "Threshold (τ)", color: "#64748b", font: { family: "Inter", size: 11 } },
    grid: { color: "rgba(255,255,255,0.04)" },
    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } },
    min: 0,
    max: 1,
  },
  y: {
    grid: { color: "rgba(255,255,255,0.04)" },
    ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } },
    min: 0,
    max: 1,
  },
};

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400, easing: "easeInOutQuart" },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "rgba(17, 24, 39, 0.95)",
      titleColor: "#f1f5f9",
      bodyColor: "#94a3b8",
      titleFont: { family: "Inter", weight: "600" },
      bodyFont: { family: "JetBrains Mono", size: 11 },
      borderColor: "rgba(99, 102, 241, 0.2)",
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
    },
  },
};

// ─── Initialize Charts ───
function initCharts() {
  charts.interrupt = new Chart($("#chart-interrupt"), {
    type: "line",
    data: {
      datasets: [
        {
          data: [],
          borderColor: chartColors.interrupt.border,
          backgroundColor: chartColors.interrupt.bg,
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          data: [],
          borderColor: "rgba(255,255,255,0.15)",
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: { ...commonScales.x },
        y: {
          ...commonScales.y,
          title: { display: true, text: "Interrupt Freq.", color: "#64748b", font: { family: "Inter", size: 11 } },
        },
      },
    },
  });

  charts.error = new Chart($("#chart-error"), {
    type: "line",
    data: {
      datasets: [
        {
          data: [],
          borderColor: chartColors.error.border,
          backgroundColor: chartColors.error.bg,
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          data: [],
          borderColor: "rgba(255,255,255,0.15)",
          borderWidth: 1,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: { ...commonScales.x },
        y: {
          ...commonScales.y,
          title: { display: true, text: "Error Rate", color: "#64748b", font: { family: "Inter", size: 11 } },
        },
      },
    },
  });

  charts.utility = new Chart($("#chart-utility"), {
    type: "line",
    data: {
      datasets: [
        {
          data: [],
          borderColor: chartColors.utility.border,
          backgroundColor: chartColors.utility.bg,
          borderWidth: 2.5,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
        {
          data: [],
          borderColor: "#f59e0b",
          backgroundColor: "#f59e0b",
          borderWidth: 0,
          pointRadius: 7,
          pointHoverRadius: 9,
          pointStyle: "star",
          showLine: false,
        },
        {
          data: [],
          borderColor: "rgba(99, 102, 241, 0.5)",
          borderWidth: 2,
          borderDash: [8, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: { ...commonScales.x },
        y: {
          ...commonScales.y,
          title: { display: true, text: "Utility", color: "#64748b", font: { family: "Inter", size: 11 } },
          min: -0.5,
          max: 1.1,
        },
      },
    },
  });
}

// ─── Update Dashboard ───
function updateDashboard() {
  const sweep = sweepThreshold(scenarios, weights, lambda, 100);
  const evalResult = evaluateAtThreshold(scenarios, threshold, weights);
  currentEvalResults = evalResult.results;

  updateCharts(sweep);

  statScenarios.textContent = scenarios.length;
  statOptimal.textContent = sweep.optimalThreshold.toFixed(2);
  statUtility.textContent = sweep.optimalUtility.toFixed(3);
  const optimalBadge = $("#optimal-badge");
  if (optimalBadge) {
      optimalBadge.textContent = `Optimal τ = ${sweep.optimalThreshold.toFixed(2)}`;
  }

  const { tp, tn, fp, fn } = evalResult.confusionMatrix;
  cmTp.textContent = tp;
  cmTn.textContent = tn;
  cmFp.textContent = fp;
  cmFn.textContent = fn;

  [cmTp, cmTn, cmFp, cmFn].forEach((el) => {
    el.style.transform = "scale(1.1)";
    setTimeout(() => (el.style.transform = "scale(1)"), 200);
  });

  metricPrecision.textContent = evalResult.precision.toFixed(3);
  metricRecall.textContent = evalResult.recall.toFixed(3);
  metricF1.textContent = evalResult.f1.toFixed(3);

  updateTable(evalResult.results);
}

function updateCharts(sweep) {
  const { curveData, optimalThreshold, optimalUtility } = sweep;

  charts.interrupt.data.datasets[0].data = curveData.map((d) => ({
    x: d.threshold,
    y: d.interruptFreq,
  }));
  charts.interrupt.data.datasets[1].data = [
    { x: threshold, y: 0 },
    { x: threshold, y: 1 },
  ];
  charts.interrupt.update("none");

  charts.error.data.datasets[0].data = curveData.map((d) => ({
    x: d.threshold,
    y: d.errorRate,
  }));
  charts.error.data.datasets[1].data = [
    { x: threshold, y: 0 },
    { x: threshold, y: 1 },
  ];
  charts.error.update("none");

  charts.utility.data.datasets[0].data = curveData.map((d) => ({
    x: d.threshold,
    y: d.utility,
  }));
  charts.utility.data.datasets[1].data = [
    { x: optimalThreshold, y: optimalUtility },
  ];
  charts.utility.data.datasets[2].data = [
    { x: threshold, y: -0.5 },
    { x: threshold, y: 1.1 },
  ];
  charts.utility.update("none");
}

function updateTable(results) {
  scenarioTbody.innerHTML = "";

  for (const r of results) {
    const tr = document.createElement("tr");
    tr.className = `row-${r.classification.toLowerCase()}`;
    tr.dataset.scenarioId = r.scenario.id;

    const U = r.uncertainty;
    const G = r.gravity;

    tr.innerHTML = `
      <td>${r.scenario.label}</td>
      <td><span style="opacity:0.6">${r.scenario.category}</span></td>
      <td>${U.components.semantic.toFixed(3)}</td>
      <td>${U.components.entropy.toFixed(3)}</td>
      <td>${U.components.reflection.toFixed(3)}</td>
      <td style="font-weight:600;color:${U.U > 0.5 ? "#f59e0b" : "#94a3b8"}">${U.U.toFixed(3)}</td>
      <td style="font-weight:600;color:${G.G > 0.5 ? "#ef4444" : "#94a3b8"}">${G.G.toFixed(3)}</td>
      <td style="font-weight:700;color:${r.trigger.score > threshold ? "#a78bfa" : "#64748b"}">${r.trigger.score.toFixed(3)}</td>
      <td>${r.predicted ? '<span class="tag-yes">YES</span>' : '<span class="tag-no">NO</span>'}</td>
      <td>${r.actual ? '<span class="tag-yes">YES</span>' : '<span class="tag-no">NO</span>'}</td>
    `;

    tr.addEventListener("click", () => openModal(r));
    scenarioTbody.appendChild(tr);
  }
}

// ─── Modal: Deep Dive ───
let modalCharts = {};

function openModal(result) {
  const { scenario, uncertainty, gravity, trigger } = result;

  Object.values(modalCharts).forEach((c) => c.destroy());
  modalCharts = {};

  $("#modal-title").textContent = scenario.label;
  $("#modal-category").textContent = scenario.category;

  const optionsContainer = $("#modal-options");
  const maxOption = Object.entries(scenario.confidenceDistribution).sort(
    (a, b) => b[1] - a[1]
  )[0][0];
  optionsContainer.innerHTML = scenario.options
    .map(
      (opt) =>
        `<span class="option-chip ${opt === maxOption ? "leading" : ""}">${opt}${opt === maxOption ? " ★" : ""}</span>`
    )
    .join("");

  const conflictsContainer = $("#modal-conflicts");
  if (scenario.reflectionConflicts.length === 0) {
    conflictsContainer.innerHTML = '<div class="no-conflicts">No conflicts identified</div>';
  } else {
    conflictsContainer.innerHTML = scenario.reflectionConflicts
      .map(
        (c, i) =>
          `<div class="conflict-item">
            <span class="conflict-name">${c}</span>
            <span class="conflict-severity">${(scenario.conflictSeverities[i] || 0.5).toFixed(2)}</span>
          </div>`
      )
      .join("");
  }

  const verdict = $("#modal-verdict");
  const score = trigger.score;
  if (trigger.interrupt) {
    verdict.className = "modal-verdict verdict-interrupt";
    verdict.innerHTML = `⚠️ INTERRUPT TRIGGERED — Score ${score.toFixed(3)} exceeds threshold ${threshold.toFixed(2)} (margin: +${trigger.margin.toFixed(3)})`;
  } else {
    verdict.className = "modal-verdict verdict-autonomous";
    verdict.innerHTML = `✅ AUTONOMOUS — Score ${score.toFixed(3)} below threshold ${threshold.toFixed(2)} (margin: ${trigger.margin.toFixed(3)})`;
  }

  modalCharts.radar = new Chart($("#chart-radar"), {
    type: "radar",
    data: {
      labels: ["Semantic\nDensity", "Decision\nEntropy", "Reflection\nDisagreement"],
      datasets: [
        {
          data: [
            uncertainty.components.semantic,
            uncertainty.components.entropy,
            uncertainty.components.reflection,
          ],
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.2)",
          borderWidth: 2,
          pointBackgroundColor: "#6366f1",
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: { stepSize: 0.25, color: "#64748b", backdropColor: "transparent", font: { family: "JetBrains Mono", size: 9 } },
          grid: { color: "rgba(255,255,255,0.06)" },
          angleLines: { color: "rgba(255,255,255,0.06)" },
          pointLabels: { color: "#94a3b8", font: { family: "Inter", size: 10 } },
        },
      },
    },
  });

  modalCharts.gravity = new Chart($("#chart-gravity-bar"), {
    type: "bar",
    data: {
      labels: ["Impact", "Irreversibility", "Propagation"],
      datasets: [
        {
          data: [gravity.factors.impact, gravity.factors.irreversibility, gravity.factors.propagation],
          backgroundColor: ["rgba(239, 68, 68, 0.6)", "rgba(245, 158, 11, 0.6)", "rgba(99, 102, 241, 0.6)"],
          borderColor: ["#ef4444", "#f59e0b", "#6366f1"],
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { family: "Inter", size: 10 } } },
        y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } } },
      },
    },
  });

  const confKeys = Object.keys(scenario.confidenceDistribution);
  const confVals = confKeys.map((k) => scenario.confidenceDistribution[k]);
  const confColors = confKeys.map((_, i) => {
    const hue = (i * 360) / confKeys.length;
    return `hsla(${hue + 230}, 70%, 60%, 0.7)`;
  });

  modalCharts.confidence = new Chart($("#chart-confidence"), {
    type: "bar",
    data: {
      labels: confKeys,
      datasets: [
        {
          data: confVals,
          backgroundColor: confColors,
          borderColor: confColors.map((c) => c.replace("0.7", "1")),
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 } } },
        y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { family: "Inter", size: 10 } } },
      },
    },
  });

  if (scenario.embeddingVectors && scenario.embeddingVectors.length > 0) {
    const points = scenario.embeddingVectors.map((v) => ({ x: v[0], y: v[1] }));
    const centroid = {
      x: points.reduce((s, p) => s + p.x, 0) / points.length,
      y: points.reduce((s, p) => s + p.y, 0) / points.length,
    };

    modalCharts.scatter = new Chart($("#chart-embedding-scatter"), {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Plans",
            data: points,
            backgroundColor: "rgba(99, 102, 241, 0.7)",
            borderColor: "#6366f1",
            borderWidth: 1,
            pointRadius: 6,
          },
          {
            label: "Centroid",
            data: [centroid],
            backgroundColor: "#f59e0b",
            borderColor: "#f59e0b",
            borderWidth: 2,
            pointRadius: 8,
            pointStyle: "crossRot",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            display: true,
            labels: { color: "#94a3b8", font: { family: "Inter", size: 10 }, usePointStyle: true, boxWidth: 8 },
          },
        },
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } }, title: { display: true, text: "Dim 1", color: "#64748b", font: { size: 10 } } },
          y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } }, title: { display: true, text: "Dim 2", color: "#64748b", font: { size: 10 } } },
        },
      },
    });
  }

  modalOverlay.classList.remove("hidden");
}

function closeModal() {
  modalOverlay.classList.add("hidden");
  Object.values(modalCharts).forEach((c) => c.destroy());
  modalCharts = {};
}

// ─── Event Listeners ───
function setupListeners() {
  // Research Dashboard weight sliders
  const updateWeights = () => {
    const raw = [
      parseInt(sliderSemantic.value),
      parseInt(sliderEntropy.value),
      parseInt(sliderReflection.value),
    ];
    const sum = raw.reduce((a, b) => a + b, 0) || 1;
    weights = raw.map((r) => r / sum);
    valSemantic.textContent = weights[0].toFixed(2);
    valEntropy.textContent = weights[1].toFixed(2);
    valReflection.textContent = weights[2].toFixed(2);
    updateDashboard();
  };

  sliderSemantic.addEventListener("input", updateWeights);
  sliderEntropy.addEventListener("input", updateWeights);
  sliderReflection.addEventListener("input", updateWeights);

  sliderThreshold.addEventListener("input", () => {
    threshold = parseInt(sliderThreshold.value) / 100;
    valThreshold.textContent = threshold.toFixed(2);
    updateDashboard();
  });

  sliderLambda.addEventListener("input", () => {
    lambda = parseInt(sliderLambda.value) / 100;
    valLambda.textContent = lambda.toFixed(2);
    updateDashboard();
  });

  btnPreset.addEventListener("click", () => {
    scenarios = [...PRESET_SCENARIOS];
    updateDashboard();
  });

  btnRandom.addEventListener("click", () => {
    scenarios = generateRandomBatch(30, Date.now());
    updateDashboard();
  });

  btnMixed.addEventListener("click", () => {
    scenarios = [
      ...PRESET_SCENARIOS,
      ...generateRandomBatch(30, Date.now()),
    ];
    updateDashboard();
  });

  modalClose.addEventListener("click", closeModal);
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

// ─── Init ───
function init() {
  initTabs();
  initCharts();
  setupListeners();
  updateDashboard();
  initSprintPanel();
  
  // Provide callback for "Apply to System" button
  initRLPanel((optimizedParams) => {
    threshold = optimizedParams.learnedThreshold;
    weights = [...optimizedParams.optimizedWeights];
    lambda = optimizedParams.optimizedLambda || 0.5;
    
    // Update sliders on Research Tab
    sliderThreshold.value = Math.round(threshold * 100);
    valThreshold.textContent = threshold.toFixed(2);
    
    sliderSemantic.value = Math.round(weights[0] * 100);
    valSemantic.textContent = weights[0].toFixed(2);
    
    sliderEntropy.value = Math.round(weights[1] * 100);
    valEntropy.textContent = weights[1].toFixed(2);
    
    sliderReflection.value = Math.round(weights[2] * 100);
    valReflection.textContent = weights[2].toFixed(2);
    
    sliderLambda.value = Math.round(lambda * 100);
    valLambda.textContent = lambda.toFixed(2);
    
    updateDashboard();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
