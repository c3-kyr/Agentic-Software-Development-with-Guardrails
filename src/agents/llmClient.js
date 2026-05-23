/**
 * LLM Client — KodeKloud (OpenAI SDK Wrapper)
 * Provides single-call and multi-variant (reflection) calls for uncertainty estimation.
 */

import OpenAI from 'openai';

let openai = null;
const MODEL = 'gpt-5.4';

/**
 * Initialize the OpenAI client
 */
export function initLLM(apiKey, apiBaseUrl) {
    openai = new OpenAI({
        apiKey: apiKey,
        baseURL: apiBaseUrl || 'https://api.openai.com/v1'
    });
}

/**
 * Single LLM call
 * @param {string} systemPrompt 
 * @param {string} userPrompt 
 * @param {Object} options - { temperature, maxTokens }
 * @returns {Promise<{ text: string, tokensUsed: number }>}
 */
export async function callLLM(systemPrompt, userPrompt, options = {}) {
    // KodeKloud gpt-5.2 strictly requires temperature = 1
    const { temperature = 1.0, maxTokens = 16384 } = options;

    try {
        const completion = await openai.chat.completions.create({
            model: MODEL,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userPrompt },
            ],
            temperature: 1.0, // Fixed at 1.0 for KodeKloud compatibility
            max_tokens: maxTokens,
        }, { timeout: 300000 }); // 5 minutes timeout

        const text = completion.choices[0]?.message?.content || '';
        const tokensUsed = completion.usage?.total_tokens || 0;

        console.log(`[LLM] Response received (${text.length} chars)`);
        if (text.length === 0) {
            console.warn('[LLM] Warning: Received empty response from model');
        }

        return { text, tokensUsed };
    } catch (error) {
        console.error('[LLM] Call failed:', error.message);
        throw new Error(`LLM call failed: ${error.message}`);
    }
}

/**
 * Multi-variant LLM call for uncertainty estimation
 */
export async function callLLMWithReflection(systemPrompt, userPrompt, n = 3) {
    const promises = [];

    for (let i = 0; i < n; i++) {
        // Since KodeKloud requires T=1.0, we rely on stochasticity at that level
        // instead of varying the temperature parameter itself.
        promises.push(
            callLLM(systemPrompt, userPrompt, { temperature: 1.0, maxTokens: 16384 })
        );
    }

    const results = await Promise.all(promises);
    const variants = results.map(r => r.text);
    const tokensUsed = results.reduce((sum, r) => sum + r.tokensUsed, 0);

    return { variants, tokensUsed };
}

/**
 * Extract JSON from an LLM response
 */
export function extractJSON(text) {
    let contentToParse = text.trim();

    if (contentToParse.startsWith('```')) {
        const lines = contentToParse.split('\n');
        lines.shift();
        if (lines[lines.length - 1].trim().startsWith('```')) {
            lines.pop();
        }
        contentToParse = lines.join('\n').trim();
    }

    const firstBrace = contentToParse.indexOf('{');
    const lastBrace = contentToParse.lastIndexOf('}');
    const firstBracket = contentToParse.indexOf('[');
    const lastBracket = contentToParse.lastIndexOf(']');

    let startIdx = firstBrace;
    let endIdx = lastBrace;

    if (firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace)) {
        startIdx = firstBracket;
        endIdx = lastBracket;
    }

    if (startIdx !== -1 && endIdx !== -1 && endIdx >= startIdx) {
        contentToParse = contentToParse.substring(startIdx, endIdx + 1);
    }

    try {
        return JSON.parse(contentToParse);
    } catch (originalError) {
        let inString = false;
        let isEscaped = false;
        let sanitized = '';

        for (let i = 0; i < contentToParse.length; i++) {
            const char = contentToParse[i];

            if (char === '\\' && !isEscaped) {
                isEscaped = true;
            } else if (isEscaped) {
                if (['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'].includes(char)) {
                    sanitized += '\\' + char;
                } else if (char === "'") {
                    sanitized += "'";
                } else {
                    sanitized += '\\\\' + char;
                }
                isEscaped = false;
            } else {
                if (char === '"') {
                    inString = !inString;
                    sanitized += char;
                } else if (inString) {
                    if (char === '\n') sanitized += '\\n';
                    else if (char === '\r') sanitized += '\\r';
                    else if (char === '\t') sanitized += '\\t';
                    else if (char.charCodeAt(0) < 32) { }
                    else sanitized += char;
                } else {
                    sanitized += char;
                }
                isEscaped = false;
            }
        }

        try {
            return JSON.parse(sanitized);
        } catch (fallbackError) {
            // Attempt to repair truncated JSON
            let repaired = sanitized;
            if (inString) {
                repaired += '"';
            }
            
            // Re-calculate stack based on repaired string (ignoring strings)
            let stack = [];
            let inStr2 = false;
            let esc = false;
            for (let i = 0; i < repaired.length; i++) {
                const c = repaired[i];
                if (c === '\\' && !esc) esc = true;
                else if (esc) esc = false;
                else if (c === '"') inStr2 = !inStr2;
                else if (!inStr2) {
                    if (c === '{') stack.push('}');
                    else if (c === '[') stack.push(']');
                    else if (c === '}' || c === ']') stack.pop();
                }
            }
            
            // Close unclosed objects/arrays
            while (stack.length > 0) {
                // If there's a trailing comma right before closing, it will fail JSON.parse.
                // It's safer to remove trailing commas before adding closing brackets.
                repaired = repaired.replace(/,\s*$/, '');
                repaired += stack.pop();
            }
            
            try {
                return JSON.parse(repaired);
            } catch (repairError) {
                const snippet = sanitized.substring(0, 200).replace(/\n/g, ' ') + '...';
                console.error('[LLM] Original parse error:', originalError.message);
                console.error('[LLM] Fallback parse error:', fallbackError.message);
                console.error('[LLM] Repair error:', repairError.message);
                console.error('[LLM] Full Sanitized Payload length:', sanitized.length);
                throw new Error(`Could not extract JSON from LLM response. Error: ${repairError.message}. Snippet: ${snippet}`);
            }
        }
    }
}
