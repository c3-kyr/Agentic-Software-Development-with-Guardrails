/**
 * Sprint Runner — Full Pipeline Orchestrator
 * 
 * Runs the complete agentic sprint:
 * 1. System Architect → Rejector → (approve/collaborate)
 * 2. Orchestrator → Rejector → (approve/collaborate)
 * 3. Frontend Agent(s) → Rejector → (approve/collaborate)
 * 4. Backend Agent(s) → Rejector → (approve/collaborate)
 * 
 * Emits WebSocket events at each stage for real-time dashboard updates.
 */

import { initLLM } from './llmClient.js';
import { runSystemArchitect, getSystemArchitectPrompt, getUserPrompt } from './systemArchitect.js';
import { runOrchestrator, getOrchestratorPrompt, getOrchestratorUserPrompt } from './orchestrator.js';
import { runFrontendAgent, getFrontendAgentPrompt, getFrontendUserPrompt } from './frontendAgent.js';
import { runBackendAgent, getBackendAgentPrompt, getBackendUserPrompt } from './backendAgent.js';
import { runRejector } from './rejector.js';

/**
 * Run a full sprint
 * 
 * @param {Object} params
 * @param {string} params.requirement - User's project requirement
 * @param {number} params.threshold - Intervention threshold τ
 * @param {number[]} params.weights - Uncertainty weights [w1, w2, w3]
 * @param {string} params.apiKey - Gemini API key
 * @param {Function} params.emit - Function to emit events: (event, data) => void
 * @param {Function} params.waitForHuman - Async function that waits for human input: (rejectorResult) => Promise<{ action, feedback? }>
 * @returns {Promise<Object>} Sprint results
 */
export async function runSprint({ requirement, threshold, weights, apiKey, apiBaseUrl, emit, waitForHuman }) {
    // Initialize LLM
    initLLM(apiKey, apiBaseUrl);

    const sprintLog = {
        id: Date.now().toString(36),
        requirement,
        threshold,
        weights,
        startTime: new Date().toISOString(),
        stages: [],
        rejectorDecisions: [],
        totalTokens: 0,
        hitlInterventions: 0,
    };

    try {
        // ═══════════════════════════════════════════
        // STAGE 1: System Architect
        // ═══════════════════════════════════════════
        emit('agent:start', { agent: 'systemArchitect', stage: 1 });
        emit('agent:thinking', { agent: 'systemArchitect', message: 'Designing system architecture...' });

        let architectResult = await runSystemArchitect(requirement);
        sprintLog.totalTokens += architectResult.tokensUsed;

        emit('agent:done', { agent: 'systemArchitect', output: architectResult.plan });

        // Rejector Gate for System Architect
        emit('rejector:start', { agent: 'systemArchitect' });

        let rejectorResult = await runRejector({
            agentName: 'System Architect',
            systemPrompt: getSystemArchitectPrompt(),
            userPrompt: getUserPrompt(requirement),
            primaryOutput: architectResult.rawText,
            threshold,
            weights,
            context: requirement,
        });

        sprintLog.rejectorDecisions.push(rejectorResult);
        emit('rejector:result', rejectorResult);

        // Handle Collaboration
        if (rejectorResult.action === 'collaborate') {
            sprintLog.hitlInterventions++;
            emit('hitl:request', {
                agent: 'System Architect',
                rejectorResult,
                agentOutput: architectResult.plan,
            });

            const humanResponse = await waitForHuman(rejectorResult);
            emit('hitl:response', humanResponse);

            if (humanResponse.feedback) {
                // Re-run architect with human feedback
                emit('agent:thinking', { agent: 'systemArchitect', message: 'Revising based on human feedback...' });
                const revisedRequirement = `${requirement}\n\nHuman Feedback: ${humanResponse.feedback}`;
                architectResult = await runSystemArchitect(revisedRequirement);
                sprintLog.totalTokens += architectResult.tokensUsed;
                emit('agent:done', { agent: 'systemArchitect', output: architectResult.plan, revised: true });
            }
        }

        sprintLog.stages.push({
            agent: 'systemArchitect',
            output: architectResult.plan,
            rejector: rejectorResult,
        });

        // ═══════════════════════════════════════════
        // STAGE 2: Orchestrator
        // ═══════════════════════════════════════════
        emit('agent:start', { agent: 'orchestrator', stage: 2 });
        emit('agent:thinking', { agent: 'orchestrator', message: 'Breaking down into tasks...' });

        let orchResult = await runOrchestrator(architectResult.plan);
        sprintLog.totalTokens += orchResult.tokensUsed;

        emit('agent:done', { agent: 'orchestrator', output: orchResult.tasks });

        // Rejector Gate for Orchestrator
        emit('rejector:start', { agent: 'orchestrator' });

        rejectorResult = await runRejector({
            agentName: 'Orchestrator',
            systemPrompt: getOrchestratorPrompt(),
            userPrompt: getOrchestratorUserPrompt(architectResult.plan),
            primaryOutput: orchResult.rawText,
            threshold,
            weights,
            context: JSON.stringify(architectResult.plan).substring(0, 1000),
        });

        sprintLog.rejectorDecisions.push(rejectorResult);
        emit('rejector:result', rejectorResult);

        if (rejectorResult.action === 'collaborate') {
            sprintLog.hitlInterventions++;
            emit('hitl:request', {
                agent: 'Orchestrator',
                rejectorResult,
                agentOutput: orchResult.tasks,
            });

            const humanResponse = await waitForHuman(rejectorResult);
            emit('hitl:response', humanResponse);

            if (humanResponse.feedback) {
                emit('agent:thinking', { agent: 'orchestrator', message: 'Revising tasks based on feedback...' });
                const revisedPlan = { ...architectResult.plan, humanFeedback: humanResponse.feedback };
                orchResult = await runOrchestrator(revisedPlan);
                sprintLog.totalTokens += orchResult.tokensUsed;
                emit('agent:done', { agent: 'orchestrator', output: orchResult.tasks, revised: true });
            }
        }

        sprintLog.stages.push({
            agent: 'orchestrator',
            output: orchResult.tasks,
            rejector: rejectorResult,
        });

        // ═══════════════════════════════════════════
        // STAGE 3: Frontend Agent (for each task)
        // ═══════════════════════════════════════════
        const frontendTasks = orchResult.tasks.frontendTasks || [];
        const frontendResults = [];

        for (let i = 0; i < Math.min(frontendTasks.length, 2); i++) { // Limit to 2 tasks for demo
            const task = frontendTasks[i];
            emit('agent:start', { agent: 'frontendAgent', stage: 3, taskIndex: i, taskId: task.id });
            emit('agent:thinking', { agent: 'frontendAgent', message: `Implementing ${task.title}...` });

            let feResult = await runFrontendAgent(task, architectResult.plan);
            sprintLog.totalTokens += feResult.tokensUsed;

            emit('agent:done', { agent: 'frontendAgent', output: feResult.implementation, taskId: task.id });

            // Rejector Gate
            emit('rejector:start', { agent: 'frontendAgent', taskId: task.id });

            rejectorResult = await runRejector({
                agentName: `Frontend Agent (${task.id})`,
                systemPrompt: getFrontendAgentPrompt(),
                userPrompt: getFrontendUserPrompt(task, architectResult.plan),
                primaryOutput: feResult.rawText,
                threshold,
                weights,
                context: `Task: ${task.title}`,
            });

            sprintLog.rejectorDecisions.push(rejectorResult);
            emit('rejector:result', rejectorResult);

            if (rejectorResult.action === 'collaborate') {
                sprintLog.hitlInterventions++;
                emit('hitl:request', {
                    agent: `Frontend Agent (${task.id})`,
                    rejectorResult,
                    agentOutput: feResult.implementation,
                });

                const humanResponse = await waitForHuman(rejectorResult);
                emit('hitl:response', humanResponse);

                if (humanResponse.feedback) {
                    emit('agent:thinking', { agent: 'frontendAgent', message: 'Revising implementation...' });
                    const revisedTask = { ...task, humanFeedback: humanResponse.feedback };
                    feResult = await runFrontendAgent(revisedTask, architectResult.plan);
                    sprintLog.totalTokens += feResult.tokensUsed;
                    emit('agent:done', { agent: 'frontendAgent', output: feResult.implementation, taskId: task.id, revised: true });
                }
            }

            frontendResults.push({ task, result: feResult, rejector: rejectorResult });
        }

        sprintLog.stages.push({
            agent: 'frontendAgent',
            tasks: frontendResults,
        });

        // ═══════════════════════════════════════════
        // STAGE 4: Backend Agent (for each task)
        // ═══════════════════════════════════════════
        const backendTasks = orchResult.tasks.backendTasks || [];
        const backendResults = [];

        for (let i = 0; i < Math.min(backendTasks.length, 2); i++) { // Limit to 2 tasks for demo
            const task = backendTasks[i];
            emit('agent:start', { agent: 'backendAgent', stage: 4, taskIndex: i, taskId: task.id });
            emit('agent:thinking', { agent: 'backendAgent', message: `Implementing ${task.title}...` });

            let beResult = await runBackendAgent(task, architectResult.plan);
            sprintLog.totalTokens += beResult.tokensUsed;

            emit('agent:done', { agent: 'backendAgent', output: beResult.implementation, taskId: task.id });

            // Rejector Gate
            emit('rejector:start', { agent: 'backendAgent', taskId: task.id });

            rejectorResult = await runRejector({
                agentName: `Backend Agent (${task.id})`,
                systemPrompt: getBackendAgentPrompt(),
                userPrompt: getBackendUserPrompt(task, architectResult.plan),
                primaryOutput: beResult.rawText,
                threshold,
                weights,
                context: `Task: ${task.title}`,
            });

            sprintLog.rejectorDecisions.push(rejectorResult);
            emit('rejector:result', rejectorResult);

            if (rejectorResult.action === 'collaborate') {
                sprintLog.hitlInterventions++;
                emit('hitl:request', {
                    agent: `Backend Agent (${task.id})`,
                    rejectorResult,
                    agentOutput: beResult.implementation,
                });

                const humanResponse = await waitForHuman(rejectorResult);
                emit('hitl:response', humanResponse);

                if (humanResponse.feedback) {
                    emit('agent:thinking', { agent: 'backendAgent', message: 'Revising implementation...' });
                    const revisedTask = { ...task, humanFeedback: humanResponse.feedback };
                    beResult = await runBackendAgent(revisedTask, architectResult.plan);
                    sprintLog.totalTokens += beResult.tokensUsed;
                    emit('agent:done', { agent: 'backendAgent', output: beResult.implementation, taskId: task.id, revised: true });
                }
            }

            backendResults.push({ task, result: beResult, rejector: rejectorResult });
        }

        sprintLog.stages.push({
            agent: 'backendAgent',
            tasks: backendResults,
        });

        // ═══════════════════════════════════════════
        // COMPLETE
        // ═══════════════════════════════════════════
        sprintLog.endTime = new Date().toISOString();
        sprintLog.duration = new Date(sprintLog.endTime) - new Date(sprintLog.startTime);

        emit('sprint:complete', {
            summary: {
                totalStages: sprintLog.stages.length,
                totalRejectorChecks: sprintLog.rejectorDecisions.length,
                hitlInterventions: sprintLog.hitlInterventions,
                totalTokens: sprintLog.totalTokens,
                duration: sprintLog.duration,
                rejectorDecisions: sprintLog.rejectorDecisions.map(d => ({
                    agent: d.agentName,
                    action: d.action,
                    score: d.score,
                    uncertainty: d.uncertainty.U,
                    gravity: d.gravity.G,
                })),
            },
            fullLog: sprintLog,
        });

        return sprintLog;

    } catch (error) {
        console.error('[Sprint] Error:', error);
        emit('sprint:error', { error: error.message, log: sprintLog });
        throw error;
    }
}
