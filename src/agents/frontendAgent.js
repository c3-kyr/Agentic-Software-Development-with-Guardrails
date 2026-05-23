/**
 * Frontend Agent
 * Takes a frontend task → generates code/specifications
 */

import { callLLM, extractJSON } from './llmClient.js';

const SYSTEM_PROMPT = `You are a Frontend Agent in a software engineering team.
Your job is to take a specific frontend task and generate the code to implement it.

You MUST respond with ONLY valid JSON in this exact format:
{
  "taskId": "FE-X",
  "implementation": {
    "files": [
      {
        "path": "src/components/Example.jsx",
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

IMPORTANT: When writing code inside the "code" field of the JSON, ensure all double quotes are properly escaped with a backslash (\"). The response must be a single, valid JSON object. Do not include any text before or after the JSON.`;

/**
 * Run the Frontend Agent on a single task
 * @param {Object} task - Frontend task from Orchestrator
 * @param {Object} architecturePlan - Overall architecture for context
 * @returns {Promise<Object>}
 */
export async function runFrontendAgent(task, architecturePlan) {
  const userPrompt = `Implement the following frontend task:

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

export function getFrontendAgentPrompt() {
  return SYSTEM_PROMPT;
}

export function getFrontendUserPrompt(task, architecturePlan) {
  return `Implement the following frontend task:\n\nTask: ${JSON.stringify(task, null, 2)}\n\nArchitecture Context: ${JSON.stringify(architecturePlan, null, 2)}\n\nRespond with ONLY the JSON implementation.`;
}
