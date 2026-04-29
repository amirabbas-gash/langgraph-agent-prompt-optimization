# langgraph-agent-prompt-optimization

This project demonstrates how to use **Agent-Lightning's Automatic Prompt Optimization (APO)** to systematically improve the performance of a **LangGraph** agent.

## Core Concepts
- **LangGraph:** Used to define the agent's workflow and state management.
- **Agent-Lightning:** A research-grade framework by Microsoft for training AI agents.
- **APO Algorithm:** Uses "textual gradients" (LLM critiques) to automatically rewrite system prompts to maximize a reward signal.

## How it Works
1. **Rollout:** The agent runs a task using a candidate prompt.
2. **Span Collection:** Agent-Lightning records the internal steps (Spans).
3. **Critique:** An LLM reviews the Spans of failed tasks to identify prompt ambiguities.
4. **Optimization:** The APO algorithm generates a new, improved prompt template.

## Setup
1. `pip install -r requirements.txt`
2. Add your `OPENAI_API_KEY` to a `.env` file.
3. Run `python train_apo.py`.