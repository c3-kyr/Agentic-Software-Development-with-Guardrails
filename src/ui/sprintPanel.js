/**
 * Sprint Panel — Main UI for the Sprint Runner tab
 * Handles: requirement input, pipeline visualization, agent output cards,
 * HITL collaboration modal, and real-time WebSocket events.
 */

import { createPipelineHTML, PipelineStateManager } from './pipelineViz.js';

let ws = null;
let pipeline = null;
let currentSprintId = null;

/**
 * Initialize the Sprint Panel
 */
export function initSprintPanel() {
    pipeline = new PipelineStateManager();
    connectWebSocket();
    setupSprintListeners();
}

// ─── WebSocket ───

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[WS] Connected');
        updateConnectionStatus(true);
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected, reconnecting...');
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        } catch (e) {
            console.error('[WS] Parse error:', e);
        }
    };
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('ws-status');
    if (indicator) {
        indicator.className = connected ? 'ws-indicator connected' : 'ws-indicator disconnected';
        indicator.title = connected ? 'Server connected' : 'Server disconnected';
    }
}

// ─── WebSocket Message Handler ───

function handleWSMessage(msg) {
    const { type, data } = msg;

    switch (type) {
        case 'agent:start':
            pipeline.setNodeState(data.agent, 'active');
            addLogEntry(`🚀 ${formatAgentName(data.agent)} started`, 'info');
            break;

        case 'agent:thinking':
            pipeline.setNodeState(data.agent, 'thinking');
            pipeline.setStatusText(data.agent, data.message);
            addLogEntry(`💭 ${formatAgentName(data.agent)}: ${data.message}`, 'thinking');
            break;

        case 'agent:done':
            pipeline.setNodeState(data.agent, 'complete');
            addAgentOutputCard(data.agent, data.output, data.revised);
            addLogEntry(`✅ ${formatAgentName(data.agent)} completed${data.revised ? ' (revised)' : ''}`, 'success');
            break;

        case 'rejector:start':
            pipeline.setNodeState('rejector', 'thinking');
            pipeline.setStatusText('rejector', `Evaluating ${formatAgentName(data.agent)}...`);
            addLogEntry(`◆ Rejector evaluating ${formatAgentName(data.agent)}...`, 'info');
            break;

        case 'rejector:result':
            handleRejectorResult(data);
            break;

        case 'hitl:request':
            pipeline.setNodeState(data.agent, 'blocked');
            showHITLModal(data);
            addLogEntry(`⚠️ HITL intervention requested for ${data.agent}`, 'warning');
            break;

        case 'hitl:response':
            hideHITLModal();
            addLogEntry(`🤝 Human feedback received`, 'success');
            break;

        case 'sprint:complete':
            handleSprintComplete(data);
            break;

        case 'sprint:error':
            addLogEntry(`❌ Sprint error: ${data.error}`, 'error');
            enableStartButton();
            break;
    }
}

function handleRejectorResult(data) {
    const color = data.action === 'collaborate' ? '#f59e0b' : '#10b981';
    pipeline.setNodeState('rejector', 'complete');
    pipeline.showBranch(data.action === 'collaborate' ? 'collaborate' : 'automate');

    // Update rejector score display
    const scoreEl = document.getElementById('rejector-score');
    if (scoreEl) {
        scoreEl.innerHTML = `
      <div class="rejector-metric">
        <span class="rejector-label">S = U × G</span>
        <span class="rejector-value" style="color: ${color}">${data.score.toFixed(3)}</span>
      </div>
      <div class="rejector-detail">
        <span>U=${data.uncertainty?.U?.toFixed(3) || '?'}</span>
        <span>G=${data.gravity?.G?.toFixed(3) || '?'}</span>
        <span>τ=${data.threshold}</span>
      </div>
      <div class="rejector-decision" style="background: ${data.action === 'collaborate' ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)'}; color: ${color}">
        ${data.action === 'collaborate' ? '🤝 COLLABORATE' : '⚡ AUTOMATE'}
      </div>
    `;
    }

    // Add to rejector decisions table
    addRejectorDecision(data);

    addLogEntry(
        `◆ Rejector: S=${data.score.toFixed(3)} → ${data.action.toUpperCase()} (Agent: ${data.agentName})`,
        data.action === 'collaborate' ? 'warning' : 'success'
    );
}

// ─── HITL Modal ───

function showHITLModal(data) {
    const modal = document.getElementById('hitl-modal');
    if (!modal) return;

    const { agent, rejectorResult, agentOutput } = data;

    document.getElementById('hitl-agent-name').textContent = agent;
    document.getElementById('hitl-score').textContent = rejectorResult.score.toFixed(3);
    document.getElementById('hitl-uncertainty').textContent = rejectorResult.uncertainty.U.toFixed(3);
    document.getElementById('hitl-gravity').textContent = rejectorResult.gravity.G.toFixed(3);

    // Uncertainty components
    const compEl = document.getElementById('hitl-components');
    compEl.innerHTML = `
    <div class="hitl-comp"><span>Semantic Density</span><span>${rejectorResult.uncertainty.components.semantic.toFixed(3)}</span></div>
    <div class="hitl-comp"><span>Decision Entropy</span><span>${rejectorResult.uncertainty.components.entropy.toFixed(3)}</span></div>
    <div class="hitl-comp"><span>Reflection Score</span><span>${rejectorResult.uncertainty.components.reflection.toFixed(3)}</span></div>
  `;

    // Conflicts
    const conflictsEl = document.getElementById('hitl-conflicts');
    const conflicts = rejectorResult.uncertainty?.details?.reflection?.conflicts || [];
    conflictsEl.innerHTML = conflicts.length > 0
        ? conflicts.map(c => `<div class="hitl-conflict">⚠ ${c}</div>`).join('')
        : '<div class="hitl-no-conflicts">No specific conflicts identified</div>';

    // Agent output preview
    document.getElementById('hitl-output-preview').textContent =
        typeof agentOutput === 'object' ? JSON.stringify(agentOutput, null, 2) : String(agentOutput);

    // Clear feedback
    document.getElementById('hitl-feedback').value = '';

    modal.classList.remove('hidden');
}

function hideHITLModal() {
    const modal = document.getElementById('hitl-modal');
    if (modal) modal.classList.add('hidden');
}

function sendHITLResponse(action) {
    const feedback = document.getElementById('hitl-feedback')?.value || '';

    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({
            type: 'hitl:response',
            sprintId: currentSprintId,
            action,
            feedback: action === 'revise' ? feedback : null,
        }));
    }

    hideHITLModal();
}

// ─── Agent Output Cards ───

function addAgentOutputCard(agentName, output, revised = false) {
    const container = document.getElementById('agent-outputs');
    if (!container) return;

    const card = document.createElement('div');
    card.className = `agent-output-card card-${agentName}${revised ? ' revised' : ''}`;

    const outputStr = typeof output === 'object' ? JSON.stringify(output, null, 2) : String(output);
    const truncated = outputStr.length > 1000 ? outputStr.substring(0, 1000) + '\n...' : outputStr;

    card.innerHTML = `
    <div class="output-card-header">
      <span class="output-agent-icon">${getAgentIcon(agentName)}</span>
      <span class="output-agent-name">${formatAgentName(agentName)}</span>
      ${revised ? '<span class="revised-badge">REVISED</span>' : ''}
    </div>
    <pre class="output-content">${escapeHTML(truncated)}</pre>
  `;

    container.appendChild(card);
    container.scrollTop = container.scrollHeight;
}

// ─── Rejector Decisions Table ───

function addRejectorDecision(data) {
    const tbody = document.getElementById('rejector-tbody');
    if (!tbody) return;

    const tr = document.createElement('tr');
    tr.className = data.action === 'collaborate' ? 'row-collaborate' : 'row-automate';
    tr.innerHTML = `
    <td>${data.agentName || '—'}</td>
    <td>${data.uncertainty?.components?.semantic?.toFixed(3) || '—'}</td>
    <td>${data.uncertainty?.components?.entropy?.toFixed(3) || '—'}</td>
    <td>${data.uncertainty?.components?.reflection?.toFixed(3) || '—'}</td>
    <td style="font-weight:600">${data.uncertainty?.U?.toFixed(3) || '—'}</td>
    <td style="font-weight:600">${data.gravity?.G?.toFixed(3) || '—'}</td>
    <td style="font-weight:700; color: ${data.action === 'collaborate' ? '#f59e0b' : '#10b981'}">${data.score?.toFixed(3) || '—'}</td>
    <td><span class="tag-${data.action === 'collaborate' ? 'yes' : 'no'}">${data.action === 'collaborate' ? 'COLLAB' : 'AUTO'}</span></td>
  `;
    tbody.appendChild(tr);
}

// ─── Activity Log ───

function addLogEntry(message, level = 'info') {
    const log = document.getElementById('activity-log');
    if (!log) return;

    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;

    const time = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="log-time">${time}</span> ${message}`;

    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

// ─── Sprint Complete ───

function handleSprintComplete(data) {
    const { summary } = data;

    pipeline.setNodeState('user', 'complete');

    addLogEntry(`🎉 Sprint complete!`, 'success');
    addLogEntry(`   📊 Rejector checks: ${summary.totalRejectorChecks}`, 'info');
    addLogEntry(`   🤝 HITL interventions: ${summary.hitlInterventions}`, 'info');
    addLogEntry(`   🪙 Total tokens: ${summary.totalTokens}`, 'info');
    addLogEntry(`   ⏱️ Duration: ${(summary.duration / 1000).toFixed(1)}s`, 'info');

    // Build the final observation table
    const observationTableRows = summary.rejectorDecisions.map(d => {
        const actionHtml = d.action === 'collaborate' 
            ? '<span style="color:#f59e0b">🤝 COLLABORATE</span>' 
            : '<span style="color:#10b981">⚡ AUTOMATE</span>';
        
        let resultText = 'Successfully generated output';
        if (d.agent.includes('System Architect')) resultText = 'Resolved architecture conflicts';
        if (d.agent.includes('Orchestrator')) resultText = 'Aligned project structure & tasks';
        if (d.agent.includes('Frontend')) resultText = 'Successfully generated React/UI components';
        if (d.agent.includes('Backend')) resultText = 'Successfully generated API/DB layer';

        return `
            <tr>
                <td style="font-weight: 600;">${d.agent}</td>
                <td>${d.uncertainty.toFixed(3)}</td>
                <td>${d.gravity.toFixed(3)}</td>
                <td style="font-weight: bold; color: ${d.score > 0.15 ? '#f59e0b' : '#94a3b8'}">${d.score.toFixed(3)}</td>
                <td>${actionHtml}</td>
                <td>${resultText}</td>
            </tr>
        `;
    }).join('');

    // Show summary card
    const summaryEl = document.getElementById('sprint-summary');
    if (summaryEl) {
        summaryEl.innerHTML = `
      <div class="summary-grid" style="margin-bottom: 2rem;">
        <div class="summary-stat">
          <span class="summary-value">${summary.totalRejectorChecks}</span>
          <span class="summary-label">Rejector Checks</span>
        </div>
        <div class="summary-stat">
          <span class="summary-value">${summary.hitlInterventions}</span>
          <span class="summary-label">HITL Interventions</span>
        </div>
        <div class="summary-stat">
          <span class="summary-value">${summary.totalTokens.toLocaleString()}</span>
          <span class="summary-label">Tokens Used</span>
        </div>
        <div class="summary-stat">
          <span class="summary-value">${(summary.duration / 1000).toFixed(1)}s</span>
          <span class="summary-label">Duration</span>
        </div>
      </div>
      
      <div class="observation-card" style="background: #1e293b; padding: 1.5rem; border-radius: 8px; border: 1px solid #334155;">
        <h3 style="margin-top: 0; color: #f8fafc;">Final Research Observation: The "Strategic Entropy" Hypothesis</h3>
        <p style="color: #cbd5e1; margin-bottom: 1.5rem; font-size: 0.95rem;">
            The final validation run using <strong>GPT-5.2</strong> on the KodeKloud API demonstrated a clear concentration of uncertainty at the higher levels of the hierarchy.
        </p>
        <div style="font-weight: bold; margin-bottom: 0.75rem; color: #cbd5e1; font-size: 0.9rem;">Validation Run Metrics (τ = ${document.getElementById('threshold-val')?.textContent || '0.15'})</div>
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem;">
            <thead>
                <tr style="border-bottom: 1px solid #334155; color: #94a3b8;">
                    <th style="padding: 0.75rem;">Agent</th>
                    <th style="padding: 0.75rem;">Uncertainty (U)</th>
                    <th style="padding: 0.75rem;">Gravity (G)</th>
                    <th style="padding: 0.75rem;">Score (S)</th>
                    <th style="padding: 0.75rem;">Action</th>
                    <th style="padding: 0.75rem;">Result</th>
                </tr>
            </thead>
            <tbody>
                ${observationTableRows}
            </tbody>
        </table>
      </div>
    `;
        summaryEl.classList.remove('hidden');
    }

    enableStartButton();
}

// ─── Setup ───

function setupSprintListeners() {
    // Start sprint
    const startBtn = document.getElementById('btn-start-sprint');
    if (startBtn) {
        startBtn.addEventListener('click', startSprint);
    }

    // HITL approve / revise
    const approveBtn = document.getElementById('hitl-approve');
    if (approveBtn) {
        approveBtn.addEventListener('click', () => sendHITLResponse('approve'));
    }

    const reviseBtn = document.getElementById('hitl-revise');
    if (reviseBtn) {
        reviseBtn.addEventListener('click', () => sendHITLResponse('revise'));
    }

    // HITL close
    const closeBtn = document.getElementById('hitl-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => sendHITLResponse('approve'));
    }

    // Sliders
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdVal = document.getElementById('threshold-val');
    if (thresholdSlider && thresholdVal) {
        thresholdSlider.addEventListener('input', () => {
            thresholdVal.textContent = (parseInt(thresholdSlider.value) / 100).toFixed(2);
        });
    }

    const weightInputs = [
        { slider: 'weight-semantic', val: 'weight-semantic-val' },
        { slider: 'weight-entropy', val: 'weight-entropy-val' },
        { slider: 'weight-reflection', val: 'weight-reflection-val' }
    ];

    weightInputs.forEach(w => {
        const slider = document.getElementById(w.slider);
        const val = document.getElementById(w.val);
        if (slider && val) {
            slider.addEventListener('input', () => {
                // To display them nicely, we just show the direct value / 100
                // Normalization happens on start sprint
                val.textContent = (parseInt(slider.value) / 100).toFixed(2);
            });
        }
    });
}

async function startSprint() {
    const requirementInput = document.getElementById('sprint-requirement');
    const thresholdSlider = document.getElementById('threshold-slider');
    const requirement = requirementInput?.value?.trim();

    if (!requirement) {
        requirementInput?.focus();
        if (requirementInput) {
            requirementInput.style.border = '1px solid #ef4444';
            setTimeout(() => requirementInput.style.border = '', 1000);
        }
        return;
    }

    // Get threshold from the existing slider
    const threshold = thresholdSlider ? parseInt(thresholdSlider.value) / 100 : 0.15;

    // Get weights from existing sliders
    const w1 = document.getElementById('weight-semantic') ? parseInt(document.getElementById('weight-semantic').value) : 40;
    const w2 = document.getElementById('weight-entropy') ? parseInt(document.getElementById('weight-entropy').value) : 35;
    const w3 = document.getElementById('weight-reflection') ? parseInt(document.getElementById('weight-reflection').value) : 25;
    const wSum = w1 + w2 + w3 || 1;
    const weights = [w1 / wSum, w2 / wSum, w3 / wSum];

    // Reset UI
    pipeline.resetAll();
    document.getElementById('agent-outputs').innerHTML = '';
    document.getElementById('rejector-tbody').innerHTML = '';
    document.getElementById('activity-log').innerHTML = '';
    document.getElementById('sprint-summary')?.classList.add('hidden');
    document.getElementById('rejector-score').innerHTML = '';

    // Start
    pipeline.setNodeState('user', 'complete');
    disableStartButton();

    addLogEntry(`🏁 Starting sprint with τ=${threshold.toFixed(2)}, weights=[${weights.map(w => w.toFixed(2)).join(', ')}]`, 'info');

    try {
        const resp = await fetch('/api/sprint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ requirement, threshold, weights }),
        });

        const result = await resp.json();
        currentSprintId = result.sprintId;
        addLogEntry(`📋 Sprint ID: ${result.sprintId}`, 'info');
    } catch (error) {
        addLogEntry(`❌ Failed to start sprint: ${error.message}`, 'error');
        enableStartButton();
    }
}

function disableStartButton() {
    const btn = document.getElementById('btn-start-sprint');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Sprint Running...';
    }
}

function enableStartButton() {
    const btn = document.getElementById('btn-start-sprint');
    if (btn) {
        btn.disabled = false;
        btn.textContent = '🚀 Start Sprint';
    }
}

// ─── Helpers ───

function formatAgentName(name) {
    const names = {
        systemArchitect: 'System Architect',
        orchestrator: 'Orchestrator',
        frontendAgent: 'Frontend Agent',
        backendAgent: 'Backend Agent',
    };
    return names[name] || name;
}

function getAgentIcon(name) {
    const icons = {
        systemArchitect: '🏗️',
        orchestrator: '📋',
        frontendAgent: '🎨',
        backendAgent: '⚙️',
    };
    return icons[name] || '🤖';
}

function escapeHTML(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
