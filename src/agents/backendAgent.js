/**
 * Backend Agent
 * Takes a backend task → generates code/specifications
 */

import { callLLM, extractJSON } from './llmClient.js';

const SYSTEM_PROMPT = `You are a Backend Agent in a software engineering team.
Your job is to take a specific backend task and generate the code to implement it.

You MUST respond with ONLY valid JSON in this exact format:
{
  "taskId": "BE-X",
  "implementation": {
    "files": [
      {
        "path": "src/routes/example.js",
        "language": "javascript",
        "code": "// actual code here",
        "explanation": "What this file does"
      }
    ]
  },
  "designDecisions": [
    {
      "decision": "What decision was made",
      "options": ["Option A", "Option B"],
      "chosen": "Option A",
      "reasoning": "Why"
    }
  ],
  "notes": "Any implementation notes"
}

Write clean, production-ready code. Include proper imports, error handling, and comments.`;

/**
 * Run the Backend Agent on a single task
 * @param {Object} task - Backend task from Orchestrator
 * @param {Object} architecturePlan - Overall architecture for context
 * @returns {Promise<Object>}
 */
export async function runBackendAgent(task, architecturePlan) {
  const userPrompt = `Implement the following backend task:

Task: ${JSON.stringify(task, null, 2)}

Architecture Context: ${JSON.stringify(architecturePlan, null, 2)}

Respond with ONLY the JSON implementation.`;

  const result = await callLLM(SYSTEM_PROMPT, userPrompt, { temperature: 0.5 });
  const implementation = extractJSON(result.text);

  return {
    implementation,
    rawText: result.text,
    tokensUsed: result.tokensUsed,
  };
}

export function getBackendAgentPrompt() {
  return SYSTEM_PROMPT;
}

export function getBackendUserPrompt(task, architecturePlan) {
  return `Implement the following backend task:\n\nTask: ${JSON.stringify(task, null, 2)}\n\nArchitecture Context: ${JSON.stringify(architecturePlan, null, 2)}\n\nRespond with ONLY the JSON implementation.`;
}
