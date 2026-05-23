/**
 * Orchestrator Agent
 * Takes an architecture plan → produces frontend + backend task lists
 */

import { callLLM, extractJSON } from './llmClient.js';

const SYSTEM_PROMPT = `You are an Orchestrator Agent in a software engineering team.
Your job is to take an approved architecture plan and break it down into concrete, 
implementable tasks for the Frontend Agent and Backend Agent.

You MUST respond with ONLY valid JSON in this exact format:
{
  "frontendTasks": [
    {
      "id": "FE-1",
      "title": "Task title",
      "description": "Detailed description of what to build",
      "files": ["src/components/Example.jsx"],
      "dependencies": [],
      "priority": "high|medium|low",
      "estimatedComplexity": "simple|moderate|complex"
    }
  ],
  "backendTasks": [
    {
      "id": "BE-1",
      "title": "Task title",
      "description": "Detailed description of what to build",
      "files": ["src/routes/example.js"],
      "dependencies": [],
      "priority": "high|medium|low",
      "estimatedComplexity": "simple|moderate|complex"
    }
  ],
  "integrationNotes": "Any notes about how frontend and backend connect"
}

Create 2-4 tasks per agent. Be specific about files and what each task should produce.`;

/**
 * Run the Orchestrator agent
 * @param {Object} architecturePlan - Approved architecture plan from System Architect
 * @returns {Promise<Object>} Task breakdown
 */
export async function runOrchestrator(architecturePlan) {
    const userPrompt = `Break down this architecture plan into Frontend and Backend tasks:\n\n${JSON.stringify(architecturePlan, null, 2)}\n\nRespond with ONLY the JSON task breakdown.`;

    const result = await callLLM(SYSTEM_PROMPT, userPrompt, { temperature: 0.5 });
    const tasks = extractJSON(result.text);

    return {
        tasks,
        rawText: result.text,
        tokensUsed: result.tokensUsed,
    };
}

export function getOrchestratorPrompt() {
    return SYSTEM_PROMPT;
}

export function getOrchestratorUserPrompt(architecturePlan) {
    return `Break down this architecture plan into Frontend and Backend tasks:\n\n${JSON.stringify(architecturePlan, null, 2)}\n\nRespond with ONLY the JSON task breakdown.`;
}
