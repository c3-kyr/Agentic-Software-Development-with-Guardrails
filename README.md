# Agentic Software Development with Guardrails

A prototype full-stack Agentic AI framework demonstrating **Human-in-the-Loop (HITL)** orchestration for LLM-based autonomous AI agents. 

This project aims to solve the reliability problem inherent in autonomous LLM software engineering. By dynamically deciding when an agent acts autonomously versus when it requires human review, we maximize productivity while minimizing catastrophic failures and hallucinations.

## 🚀 Features

- **Multi-Agent Orchestration**: Powered by LLM agents acting as System Architects, Frontend Developers, and Backend Developers working collaboratively.
- **Dynamic Human-in-the-Loop (HITL)**: A "Rejector" mechanism flags risky decisions for human collaboration based on a mathematically sound threshold.
- **Deep Q-Network (DQN) Engine**: An integrated Reinforcement Learning agent that dynamically learns and optimizes the intervention threshold.
- **Custom Reward Modeling**: The DQN uses a token-cost reward system that penalizes critical failures at **3.5x** the rate of false positives to enforce risk-aversion.
- **Real-Time Dashboard**: A WebSocket-driven React frontend to visualize agent reliability, test scenarios, view confusion matrices, and monitor live F1-score convergence.

## 🧠 Architecture Overview

1. **Agent Pipeline (`src/agents/`)**: Individual LLM agents handle specialized development tasks. 
2. **Uncertainty & Gravity Modules (`src/uncertainty/`, `src/gravity/`)**: Evaluates every agent proposal by quantifying semantic uncertainty, decision entropy, and reflection disagreement, multiplied by the systemic impact of the change (Gravity).
3. **Reinforcement Learning (`src/rl/`)**: A pure JavaScript neural network and replay buffer train a DQN on a continuous parameter optimization loop (w₁, w₂, w₃, λ) to update the threshold.
4. **WebSocket Server (`server.js`)**: A Node.js/Express backend that streams telemetry and state changes to the UI in real-time.

## 🛠 Setup & Installation

### Prerequisites
- Node.js (v16+ recommended)
- An OpenAI API Key (or a compatible local LLM proxy)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/c3-kyr/Agentic-Software-Development-with-Guardrails.git
   cd hitl-research
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure Environment Variables**
   Rename or create a `.env` file in the root directory:
   ```env
   PORT=3001
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_BASE_URL= # (Optional) If you are using a proxy
   ```

4. **Run the Development Server**
   ```bash
   npm run dev
   ```
   *The Vite frontend dashboard will automatically open in your browser, connected to the Node.js WebSocket backend.*

## 📈 The Dashboard

The UI provides three main interactive tabs:
1. **Sprint Runner**: Kick off autonomous coding sprints, view the agent pipeline, and collaborate with the AI via HITL pop-up alerts.
2. **Research Dashboard**: View the mathematical impact of uncertainty weights, scenario matrices, and cost utility functions.
3. **RL Training**: Run a live DQN episode simulation to observe how the AI agent independently discovers the optimal human intervention threshold by maximizing the F1 score over time.

---

*Built for advanced agentic coding research.*
