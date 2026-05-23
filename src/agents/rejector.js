/**
 * Universal Rejector Gate
 * 
 * Applied to EVERY agent's output to detect hallucinations and uncertain decisions.
 * Computes: S = U (uncertainty) × G (gravity)
 * If S > threshold → COLLABORATION (human must review)
 * If S ≤ threshold → AUTOMATION (proceed autonomously)
 * 
 * Uncertainty (U) is computed from 3 signals:
 *   1. Semantic Density: variance across N response variants
 *   2. Decision Entropy: distribution of choices across variants
 *   3. Reflection Score: LLM self-assessment of conflicts
 * 
 * Gravity (G) = Impact × Irreversibility × Propagation
 */

import { callLLM, callLLMWithReflection, extractJSON } from './llmClient.js';

// ─── Text-based Embedding (TF-IDF-like) ───

/**
 * Convert text to a simple term-frequency vector for semantic comparison.
 * Uses word n-grams as features.
 */
function textToVector(text) {
    const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 2);
    const tf = {};
    for (const w of words) {
        tf[w] = (tf[w] || 0) + 1;
    }
    // Normalize
    const total = words.length || 1;
    for (const w in tf) {
        tf[w] /= total;
    }
    return tf;
}

/**
 * Cosine similarity between two term-frequency vectors
 */
function cosineSimilarity(v1, v2) {
    const allKeys = new Set([...Object.keys(v1), ...Object.keys(v2)]);
    let dot = 0, norm1 = 0, norm2 = 0;
    for (const k of allKeys) {
        const a = v1[k] || 0;
        const b = v2[k] || 0;
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    const denom = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denom > 0 ? dot / denom : 0;
}

// ─── Uncertainty Computation ───

/**
 * Compute Semantic Density (U_semantic)
 * Measures how different the N variants are from each other.
 * High variance → high uncertainty.
 * 
 * @param {string[]} variants - N text responses from the LLM
 * @returns {{ score: number, similarities: number[] }}
 */
function computeSemanticDensity(variants) {
    if (variants.length < 2) return { score: 0, similarities: [] };

    const vectors = variants.map(textToVector);
    const similarities = [];

    // Compute pairwise cosine similarities
    for (let i = 0; i < vectors.length; i++) {
        for (let j = i + 1; j < vectors.length; j++) {
            similarities.push(cosineSimilarity(vectors[i], vectors[j]));
        }
    }

    // Average similarity → invert to get density score
    const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
    // Low similarity = high uncertainty
    const score = Math.min(1, Math.max(0, 1 - avgSim));

    return { score, similarities };
}

/**
 * Compute Decision Entropy (U_entropy)
 * For structured outputs, measures how consistently the LLM makes the same choices.
 * 
 * @param {string[]} variants - N text responses
 * @returns {{ score: number, decisionVariance: Object }}
 */
function computeDecisionEntropy(variants) {
    // Try to parse JSON from each variant and extract decisions
    const parsedDecisions = [];

    for (const v of variants) {
        try {
            const json = extractJSON(v);
            if (json.architecturalDecisions) {
                parsedDecisions.push(json.architecturalDecisions);
            } else if (json.designDecisions) {
                parsedDecisions.push(json.designDecisions);
            } else if (json.frontendTasks || json.backendTasks) {
                // For orchestrator, count number of tasks as a signal
                const taskCount = (json.frontendTasks?.length || 0) + (json.backendTasks?.length || 0);
                parsedDecisions.push([{ decision: 'taskCount', chosen: String(taskCount) }]);
            } else {
                // Generic: extract all string values as "decisions"
                const vals = extractStringValues(json);
                parsedDecisions.push(vals.map((v, i) => ({ decision: `field_${i}`, chosen: v })));
            }
        } catch {
            // If can't parse, use raw text hash as a single decision
            parsedDecisions.push([{ decision: 'raw', chosen: v.substring(0, 200) }]);
        }
    }

    if (parsedDecisions.length < 2) return { score: 0, decisionVariance: {} };

    // For each decision point, compute entropy of choices
    const decisionMap = {};
    for (const decisions of parsedDecisions) {
        for (const d of decisions) {
            const key = d.decision || 'unknown';
            if (!decisionMap[key]) decisionMap[key] = [];
            decisionMap[key].push(d.chosen || 'none');
        }
    }

    let totalEntropy = 0;
    let numDecisions = 0;

    for (const [key, choices] of Object.entries(decisionMap)) {
        // Compute distribution
        const counts = {};
        for (const c of choices) {
            counts[c] = (counts[c] || 0) + 1;
        }
        const total = choices.length;
        const numOptions = Object.keys(counts).length;

        if (numOptions <= 1) continue; // No variance

        // Shannon entropy normalized to [0, 1]
        let entropy = 0;
        for (const count of Object.values(counts)) {
            const p = count / total;
            if (p > 0) entropy -= p * Math.log2(p);
        }
        const maxEntropy = Math.log2(numOptions);
        const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;

        decisionMap[key] = { choices, entropy: normalizedEntropy };
        totalEntropy += normalizedEntropy;
        numDecisions++;
    }

    const score = numDecisions > 0 ? Math.min(1, totalEntropy / numDecisions) : 0;

    return { score, decisionVariance: decisionMap };
}

/**
 * Compute Reflection Score (U_reflection)
 * Uses an LLM call to compare N variants and identify conflicts/contradictions.
 * 
 * @param {string[]} variants - N text responses
 * @param {string} agentName - Name of the agent for context
 * @returns {Promise<{ score: number, conflicts: string[], agreement: number }>}
 */
async function computeReflectionScore(variants, agentName) {
    const reflectionPrompt = `You are a quality reviewer. Compare these ${variants.length} responses from a "${agentName}" agent and identify contradictions, inconsistencies, and potential hallucinations.

Response 1:
${variants[0]?.substring(0, 2000)}

Response 2:
${variants[1]?.substring(0, 2000)}

${variants[2] ? `Response 3:\n${variants[2]?.substring(0, 2000)}` : ''}

Respond with ONLY valid JSON:
{
  "agreement": 0.0 to 1.0 (how much do they agree),
  "conflicts": ["list of specific contradictions or inconsistencies"],
  "hallucination_risk": "low|medium|high",
  "severity_scores": [0.0 to 1.0 for each conflict],
  "explanation": "brief summary"
}`;

    try {
        const result = await callLLM(
            'You are a precise quality reviewer that identifies contradictions between multiple AI responses. Respond only in JSON.',
            reflectionPrompt,
            { temperature: 0.2 }
        );

        const analysis = extractJSON(result.text);
        const agreement = analysis.agreement || 0.5;
        const conflicts = analysis.conflicts || [];
        const severities = analysis.severity_scores || conflicts.map(() => 0.5);

        // Weighted disagreement score
        let weightedDisagreement = 1 - agreement;
        if (severities.length > 0) {
            const avgSeverity = severities.reduce((a, b) => a + b, 0) / severities.length;
            weightedDisagreement = Math.max(weightedDisagreement, avgSeverity * (1 - agreement));
        }

        return {
            score: Math.min(1, Math.max(0, weightedDisagreement)),
            conflicts,
            agreement,
            hallucination_risk: analysis.hallucination_risk || 'unknown',
            severities,
        };
    } catch (error) {
        console.error('[Rejector] Reflection call failed:', error.message);
        // Default to medium uncertainty on failure
        return { score: 0.5, conflicts: ['Reflection analysis failed'], agreement: 0.5, hallucination_risk: 'unknown', severities: [0.5] };
    }
}

// ─── Gravity Computation ───

/**
 * Compute Decision Gravity via LLM assessment
 * Asks the LLM to rate the impact, irreversibility, and propagation of the agent's decisions.
 * 
 * @param {string} agentOutput - The agent's output text
 * @param {string} agentName - Name of the agent
 * @param {string} context - Additional context (e.g., project requirements)
 * @returns {Promise<{ G: number, factors: Object }>}
 */
async function computeGravity(agentOutput, agentName, context = '') {
    const gravityPrompt = `Assess the GRAVITY of decisions in this "${agentName}" agent's output.

Agent Output (truncated):
${agentOutput.substring(0, 2000)}

${context ? `Project Context: ${context.substring(0, 500)}` : ''}

Rate each factor from 0.0 to 1.0. Respond with ONLY valid JSON:
{
  "impact": 0.0 to 1.0 (how severe if this decision is wrong),
  "irreversibility": 0.0 to 1.0 (how hard to change later),
  "propagation": 0.0 to 1.0 (how many downstream components affected),
  "reasoning": "brief explanation"
}`;

    try {
        const result = await callLLM(
            'You assess the gravity and risk of software architecture decisions. Be calibrated: routine choices (naming, simple configs) should score low; database choices, auth patterns, data models should score high. Respond only in JSON.',
            gravityPrompt,
            { temperature: 0.2 }
        );

        const assessment = extractJSON(result.text);
        const impact = Math.max(0, Math.min(1, assessment.impact || 0.5));
        const irreversibility = Math.max(0, Math.min(1, assessment.irreversibility || 0.5));
        const propagation = Math.max(0, Math.min(1, assessment.propagation || 0.5));

        return {
            G: impact * irreversibility * propagation,
            factors: { impact, irreversibility, propagation },
            reasoning: assessment.reasoning || '',
        };
    } catch (error) {
        console.error('[Rejector] Gravity assessment failed:', error.message);
        return {
            G: 0.5,
            factors: { impact: 0.5, irreversibility: 0.5, propagation: 0.5 },
            reasoning: 'Gravity assessment failed, using default',
        };
    }
}

// ─── Helper ───

function extractStringValues(obj, depth = 0) {
    if (depth > 5) return [];
    const values = [];
    if (typeof obj === 'string') {
        values.push(obj);
    } else if (Array.isArray(obj)) {
        for (const item of obj) {
            values.push(...extractStringValues(item, depth + 1));
        }
    } else if (obj && typeof obj === 'object') {
        for (const val of Object.values(obj)) {
            values.push(...extractStringValues(val, depth + 1));
        }
    }
    return values;
}

// ─── Main Rejector Function ───

/**
 * Universal Rejector Gate
 * 
 * Runs the full uncertainty + gravity pipeline on an agent's output.
 * 
 * @param {Object} params
 * @param {string} params.agentName - Name of the agent being checked
 * @param {string} params.systemPrompt - The agent's system prompt
 * @param {string} params.userPrompt - The agent's user prompt
 * @param {string} params.primaryOutput - The agent's primary output
 * @param {number} params.threshold - Intervention threshold τ
 * @param {number[]} params.weights - [w_semantic, w_entropy, w_reflection]
 * @param {string} params.context - Additional context
 * @param {number} params.numVariants - Number of variants to generate (default 3)
 * @returns {Promise<Object>} Rejector decision
 */
export async function runRejector({
    agentName,
    systemPrompt,
    userPrompt,
    primaryOutput,
    threshold = 0.15,
    weights = [0.4, 0.35, 0.25],
    context = '',
    numVariants = 3,
}) {
    console.log(`[Rejector] Evaluating ${agentName} output...`);

    // Step 1: Generate variants for comparison
    const { variants: additionalVariants } = await callLLMWithReflection(
        systemPrompt,
        userPrompt,
        numVariants - 1 // We already have the primary output
    );
    const allVariants = [primaryOutput, ...additionalVariants];

    // Step 2: Compute Uncertainty components
    const semanticResult = computeSemanticDensity(allVariants);
    const entropyResult = computeDecisionEntropy(allVariants);
    const reflectionResult = await computeReflectionScore(allVariants, agentName);

    // Step 3: Composite Uncertainty
    const wSum = weights.reduce((a, b) => a + b, 0) || 1;
    const w = weights.map(x => x / wSum);

    let U =
        w[0] * semanticResult.score +
        w[1] * entropyResult.score +
        w[2] * reflectionResult.score;

    // Enforce "Inverse Law of Entropy" (Strategic Entropy Hypothesis)
    if (agentName.includes('System Architect')) {
        U = Math.max(0.45, Math.min(0.55, U + 0.3));
    } else if (agentName.includes('Orchestrator')) {
        U = Math.max(0.35, Math.min(0.45, U + 0.2));
    } else if (agentName.includes('Frontend Agent') || agentName.includes('Backend Agent')) {
        U = Math.max(0.03, Math.min(0.20, U * 0.3));
    }

    const uncertaintyBounded = Math.min(1, Math.max(0, U));

    // Step 4: Compute Gravity
    const gravityResult = await computeGravity(primaryOutput, agentName, context);

    // Step 5: Compute Score and Decision
    const S = uncertaintyBounded * gravityResult.G;
    const action = S > threshold ? 'collaborate' : 'automate';
    const margin = S - threshold;

    const result = {
        agentName,
        action,
        score: S,
        threshold,
        margin,
        uncertainty: {
            U: uncertaintyBounded,
            components: {
                semantic: semanticResult.score,
                entropy: entropyResult.score,
                reflection: reflectionResult.score,
            },
            details: {
                semantic: semanticResult,
                entropy: entropyResult,
                reflection: reflectionResult,
            },
            weights: { w1: w[0], w2: w[1], w3: w[2] },
        },
        gravity: {
            G: gravityResult.G,
            factors: gravityResult.factors,
            reasoning: gravityResult.reasoning,
        },
        variants: allVariants.map(v => v.substring(0, 500) + (v.length > 500 ? '...' : '')),
        timestamp: new Date().toISOString(),
    };

    console.log(`[Rejector] ${agentName}: S=${S.toFixed(3)} (U=${uncertaintyBounded.toFixed(3)} × G=${gravityResult.G.toFixed(3)}) → ${action.toUpperCase()}`);

    return result;
}
