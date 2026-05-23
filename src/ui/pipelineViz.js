/**
 * Pipeline Visualization — Animated CSS flowchart
 * Matches the architecture diagram: User → System Architect → Rejector → Orchestrator → FE/BE Agents
 */

/**
 * Create the pipeline visualization HTML
 * @returns {string} HTML string
 */
export function createPipelineHTML() {
    return `
    <div class="pipeline-container" id="pipeline-viz">
      <div class="pipeline-row pipeline-row-1">
        <div class="pipeline-node node-user" id="node-user" data-label="User">
          <div class="node-icon">👤</div>
          <div class="node-label">User</div>
        </div>
        <div class="pipeline-arrow arrow-right" id="arrow-user-arch">→</div>
        <div class="pipeline-node node-agent" id="node-systemArchitect" data-label="System Architect">
          <div class="node-icon">🏗️</div>
          <div class="node-label">System Architect</div>
          <div class="node-status" id="status-systemArchitect"></div>
        </div>
      </div>

      <div class="pipeline-connector connector-down" id="connector-arch-rejector">↓</div>

      <div class="pipeline-row pipeline-row-2">
        <div class="pipeline-node node-rejector" id="node-rejector" data-label="Rejector Gate">
          <div class="node-icon">◆</div>
          <div class="node-label">Rejector Gate</div>
          <div class="node-status" id="status-rejector"></div>
        </div>
      </div>

      <div class="pipeline-branch" id="branch-container">
        <div class="branch-left">
          <div class="pipeline-connector connector-branch-left">↙</div>
          <div class="pipeline-node node-action" id="node-automate" data-label="Automation">
            <div class="node-icon">⚡</div>
            <div class="node-label">Automation</div>
          </div>
        </div>
        <div class="branch-right">
          <div class="pipeline-connector connector-branch-right">↘</div>
          <div class="pipeline-node node-action node-collab" id="node-collaborate" data-label="Collaboration">
            <div class="node-icon">🤝</div>
            <div class="node-label">Collaboration</div>
          </div>
        </div>
      </div>

      <div class="pipeline-connector connector-down" id="connector-to-orchestrator">↓</div>

      <div class="pipeline-row pipeline-row-3">
        <div class="pipeline-node node-agent" id="node-orchestrator" data-label="Orchestrator">
          <div class="node-icon">📋</div>
          <div class="node-label">Orchestrator</div>
          <div class="node-status" id="status-orchestrator"></div>
        </div>
      </div>

      <div class="pipeline-connector connector-down" id="connector-to-agents">↓</div>

      <div class="pipeline-row pipeline-row-4">
        <div class="pipeline-node node-agent node-fe" id="node-frontendAgent" data-label="Frontend Agent">
          <div class="node-icon">🎨</div>
          <div class="node-label">Frontend Agent</div>
          <div class="node-status" id="status-frontendAgent"></div>
        </div>
        <div class="pipeline-node node-agent node-be" id="node-backendAgent" data-label="Backend Agent">
          <div class="node-icon">⚙️</div>
          <div class="node-label">Backend Agent</div>
          <div class="node-status" id="status-backendAgent"></div>
        </div>
      </div>
    </div>
  `;
}

/**
 * Pipeline state manager
 */
export class PipelineStateManager {
    constructor() {
        this.states = {
            user: 'idle',
            systemArchitect: 'idle',
            rejector: 'idle',
            automate: 'idle',
            collaborate: 'idle',
            orchestrator: 'idle',
            frontendAgent: 'idle',
            backendAgent: 'idle',
        };
    }

    /**
     * Set a node's state: idle, active, thinking, complete, blocked, error
     */
    setNodeState(nodeId, state) {
        this.states[nodeId] = state;
        const el = document.getElementById(`node-${nodeId}`);
        if (!el) return;

        // Remove all state classes
        el.classList.remove('state-idle', 'state-active', 'state-thinking', 'state-complete', 'state-blocked', 'state-error');
        el.classList.add(`state-${state}`);

        // Update status text
        const statusEl = document.getElementById(`status-${nodeId}`);
        if (statusEl) {
            const statusTexts = {
                idle: '',
                active: 'Starting...',
                thinking: 'Processing...',
                complete: '✓ Done',
                blocked: '⏸ Waiting for human',
                error: '✗ Error',
            };
            statusEl.textContent = statusTexts[state] || '';
        }
    }

    setStatusText(nodeId, text) {
        const statusEl = document.getElementById(`status-${nodeId}`);
        if (statusEl) statusEl.textContent = text;
    }

    /**
     * Show which branch path (automate/collaborate) is active
     */
    showBranch(branch) {
        const autoEl = document.getElementById('node-automate');
        const collabEl = document.getElementById('node-collaborate');

        if (autoEl) {
            autoEl.classList.remove('state-active', 'state-complete');
            if (branch === 'automate') autoEl.classList.add('state-active');
        }
        if (collabEl) {
            collabEl.classList.remove('state-active', 'state-complete');
            if (branch === 'collaborate') collabEl.classList.add('state-active');
        }
    }

    /**
     * Reset all nodes to idle
     */
    resetAll() {
        for (const nodeId of Object.keys(this.states)) {
            this.setNodeState(nodeId, 'idle');
        }
    }
}
