/**
 * Express + WebSocket Server
 * Serves the HITL Agentic Sprint API
 */

import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs/promises';
import path from 'path';
import { runSprint } from './src/agents/sprintRunner.js';

const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;

if (!OPENAI_API_KEY) {
    console.error('❌ OPENAI_API_KEY not set in .env');
    process.exit(1);
}

// ─── Active Sprint State ───
const activeSprints = new Map();
const humanResponseCallbacks = new Map();

// ─── Telemetry Persistence ───
const TELEMETRY_FILE = path.join(process.cwd(), 'data', 'sprints.json');

async function ensureTelemetryDir() {
    try {
        await fs.mkdir(path.dirname(TELEMETRY_FILE), { recursive: true });
    } catch (e) {}
}

async function saveSprintLog(sprintLog) {
    await ensureTelemetryDir();
    let logs = [];
    try {
        const data = await fs.readFile(TELEMETRY_FILE, 'utf-8');
        logs = JSON.parse(data);
    } catch (e) {}
    logs.push(sprintLog);
    await fs.writeFile(TELEMETRY_FILE, JSON.stringify(logs, null, 2));
}

// ─── WebSocket ───
wss.on('connection', (ws) => {
    const clientId = uuidv4();
    console.log(`[WS] Client connected: ${clientId}`);

    ws.clientId = clientId;

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data);
            console.log(`[WS] Message from ${clientId}:`, msg.type);

            if (msg.type === 'hitl:response') {
                const callback = humanResponseCallbacks.get(msg.sprintId);
                if (callback) {
                    callback({ action: msg.action, feedback: msg.feedback });
                    humanResponseCallbacks.delete(msg.sprintId);
                }
            }
        } catch (e) {
            console.error('[WS] Invalid message:', e.message);
        }
    });

    ws.on('close', () => {
        console.log(`[WS] Client disconnected: ${clientId}`);
    });
});

function broadcast(event, data) {
    const msg = JSON.stringify({ type: event, data, timestamp: Date.now() });
    wss.clients.forEach((client) => {
        if (client.readyState === 1) { // WebSocket.OPEN
            client.send(msg);
        }
    });
}

// ─── API Routes ───

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', connections: wss.clients.size });
});

app.get('/api/telemetry/parameters', async (req, res) => {
    try {
        const data = await fs.readFile(TELEMETRY_FILE, 'utf-8');
        const logs = JSON.parse(data);

        if (logs.length < 5) {
            return res.json({ status: 'insufficient_data', message: 'Need more runs to compute empirical values.' });
        }

        const tokenCosts = logs.map(l => l.totalTokens).sort((a, b) => a - b);
        
        // Empirical T_min (10th percentile) and T_max (90th percentile)
        const p10 = tokenCosts[Math.floor(tokenCosts.length * 0.1)] || 500;
        const p90 = tokenCosts[Math.floor(tokenCosts.length * 0.9)] || 10000;
        
        // Empirical T_review: avg tokens spent in HITL feedback loop (approx. heuristic from duration differences)
        const hitlRuns = logs.filter(l => l.hitlInterventions > 0);
        let t_review_empirical = 800; 
        if (hitlRuns.length > 0) {
            const avgHitlTokens = hitlRuns.reduce((sum, l) => sum + (l.totalTokens / Math.max(1, l.hitlInterventions)), 0) / hitlRuns.length;
            // The extra overhead tokens caused by interventions ~ roughly 15-20% of a task
            t_review_empirical = Math.round(avgHitlTokens * 0.15);
        }

        res.json({
            status: 'success',
            params: {
                T_min: p10,
                T_max: p90,
                T_review: t_review_empirical,
                alpha: 2.5, // Typically derived from historical fix vs rewrite time
                eta: 0.6    // Derived from historical smooth-automation value vs intervention
            },
            sampleSize: logs.length
        });
    } catch (e) {
        res.json({ status: 'no_data', message: 'No telemetry data found.' });
    }
});

app.post('/api/sprint', async (req, res) => {
    const { requirement, threshold = 0.15, weights = [0.4, 0.35, 0.25] } = req.body;

    if (!requirement) {
        return res.status(400).json({ error: 'Requirement is required' });
    }

    const sprintId = uuidv4();
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`🚀 Starting Sprint: ${sprintId}`);
    console.log(`📋 Requirement: ${requirement.substring(0, 100)}...`);
    console.log(`📊 Threshold: ${threshold}, Weights: [${weights.join(', ')}]`);
    console.log(`${'═'.repeat(60)}\n`);

    res.json({ sprintId, status: 'started' });

    // Run sprint asynchronously
    try {
        const result = await runSprint({
            requirement,
            threshold,
            weights,
            apiKey: OPENAI_API_KEY,
            apiBaseUrl: OPENAI_BASE_URL,
            emit: (event, data) => {
                broadcast(event, { ...data, sprintId });
            },
            waitForHuman: (rejectorResult) => {
                return new Promise((resolve) => {
                    const timeoutMs = 5 * 60 * 1000; // 5 minute timeout
                    humanResponseCallbacks.set(sprintId, resolve);

                    // Auto-approve after timeout
                    setTimeout(() => {
                        if (humanResponseCallbacks.has(sprintId)) {
                            console.log(`[HITL] Timeout for ${sprintId}, auto-approving`);
                            humanResponseCallbacks.delete(sprintId);
                            resolve({ action: 'approve', feedback: null });
                        }
                    }, timeoutMs);
                });
            },
        });

        activeSprints.set(sprintId, result);
        
        // Save telemetry data for RL Model
        await saveSprintLog(result).catch(err => console.error('Failed to save sprint telemetry:', err));

    } catch (error) {
        console.error(`[Sprint] Failed:`, error);
        broadcast('sprint:error', { sprintId, error: error.message });
    }
});

app.get('/api/sprint/:id', (req, res) => {
    const sprint = activeSprints.get(req.params.id);
    if (sprint) {
        res.json(sprint);
    } else {
        res.status(404).json({ error: 'Sprint not found' });
    }
});

// ─── Start ───
server.listen(PORT, () => {
    console.log(`\n🔧 HITL Agentic Sprint Server`);
    console.log(`   HTTP:  http://localhost:${PORT}`);
    console.log(`   WS:    ws://localhost:${PORT}/ws`);
    console.log(`   API:   POST /api/sprint\n`);
});
