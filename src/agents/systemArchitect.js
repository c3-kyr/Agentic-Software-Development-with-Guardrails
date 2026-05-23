/**
 * System Architect Agent
 * Takes user requirements → produces structured architecture plan JSON
 */

import { callLLM, extractJSON } from './llmClient.js';

const SYSTEM_PROMPT = `You are a System Architect Agent in a software engineering team. 
Your job is to take user requirements and produce a structured architecture plan.

You MUST respond with ONLY valid JSON (no markdown, no explanation outside JSON) in this exact format:
{
  "projectName": "string",
  "summary": "brief project summary",
  "techStack": {
    "frontend": "framework/library name",
    "backend": "framework/runtime",
    "database": "database name",
    "other": ["any other tools"]
  },
  "components": [
    {
      "name": "ComponentName",
      "type": "frontend|backend|shared",
      "description": "what it does",
      "dependencies": ["other component names"]
    }
  ],
  "apiEndpoints": [
    {
      "method": "GET|POST|PUT|DELETE",
      "path": "/api/...",
      "description": "what it does",
      "requestBody": "description or null",
      "responseBody": "description"
    }
  ],
  "dataModels": [
    {
      "name": "ModelName",
      "fields": [
        { "name": "fieldName", "type": "string|number|boolean|date|reference", "required": true }
      ]
    }
  ],
  "architecturalDecisions": [
    {
      "decision": "What decision was made",
      "options": ["Option A", "Option B"],
      "chosen": "Option A",
      "reasoning": "Why this was chosen",
      "category": "database|architecture|security|api|infrastructure|frontend"
    }
  ]
}

Be thorough but practical. Include at least 3 architectural decisions.`;

/**
 * Run the System Architect agent
 * @param {string} requirement - User's project requirement
 * @returns {Promise<Object>} Architecture plan
 */
export async function runSystemArchitect(requirement) {
    const userPrompt = `Design the architecture for the following project requirement:\n\n${requirement}\n\nRespond with ONLY the JSON architecture plan.`;

    const result = await callLLM(SYSTEM_PROMPT, userPrompt, { temperature: 0.7 });
    const plan = extractJSON(result.text);

    return {
        plan,
        rawText: result.text,
        tokensUsed: result.tokensUsed,
    };
}

/**
 * Get the system prompt (exposed for Rejector to use with reflection calls)
 */
export function getSystemArchitectPrompt() {
    return SYSTEM_PROMPT;
}

export function getUserPrompt(requirement) {
    return `Design the architecture for the following project requirement:\n\n${requirement}\n\nRespond with ONLY the JSON architecture plan.`;
}
