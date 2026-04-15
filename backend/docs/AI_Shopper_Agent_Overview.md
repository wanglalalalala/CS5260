# AI Shopper Agent Overview (Group 30)

## Team Information

Course: CS5260  
Group: 30

Members:
- Liu Fayang (A0229303M)
- Wang Longjun (A0332210A)
- Wu Di (A0326687M)
- Xiang Subing (A0332048L)

## 1) What this agent is

AI Shopper Agent is a multi-turn, tool-augmented shopping assistant for consumer electronics.
It combines:

- LLM-based dialogue orchestration (intent routing + clarification + recommendation writing)
- Deterministic retrieval/tool pipeline (RAG + filters + comparison + checkout simulation)
- Interactive Streamlit UI for real-time demo and user interaction

The system is designed to move users from vague requests to actionable product decisions through a "consult-and-decide" workflow.

## 2) High-level architecture

### Dialogue layer (`backend/agent/dialogue`)

- **Orchestrator**: `ShoppingAgent` keeps persistent dialogue state across turns.
- **LangGraph routing**:
  - `supervisor` decides next route
  - `clarify` asks focused follow-up questions and outputs clickable suggestions
  - `search` extracts constraints, rewrites query, retrieves products, and generates user-facing recommendations
  - deterministic nodes: `compare`, `detail`, `checkout`
- **State tracking**:
  - category, brand, price constraints, previous products, clarify count, whether search already happened

### Retrieval + tool layer (`backend/rag`, `backend/agent/tools.py`)

- **Data/index**:
  - local ChromaDB vector index + SQLite structured store
  - curated Amazon consumer electronics dataset
- **Search pipeline**:
  - semantic retrieval + deterministic filtering/ranking
  - constraints include category, brand, min/max price, rating, specs
- **Deterministic tool calls**:
  - comparison, product detail lookup, checkout simulation, brand stats, price stats, keyword fallback, etc.

### Frontend (`frontend`)

- Streamlit chat application with:
  - category quick-start buttons
  - assistant suggestion chips
  - reasoning/tool-call expander
  - applied filters + recommended items + token/cost sidebar
  - provider-aware backend config (OpenAI/Qwen)

## 3) Supported interaction patterns

- User provides vague need -> agent asks one focused clarification
- User provides constraints (budget/brand/product type) -> agent searches and recommends top picks
- User asks for side-by-side comparison (e.g., "compare #1 and #2")
- User asks for details of one item
- User confirms purchase -> simulated checkout response

## 4) Grounding and reliability strategy

- Retrieval outputs are based on local product data (not fabricated catalog entries).
- Critical operations use deterministic Python functions instead of free-form LLM reasoning:
  - strict filters
  - ranking
  - product comparison and checkout
- LLM is mainly used for:
  - intent/slot extraction
  - clarification wording
  - readable recommendation narration

## 5) Cost/latency strategy

- Supports lightweight models (`qwen-plus`, `gpt-4o-mini`, plus offline mock mode).
- Tracks usage tokens and estimated cost in backend/frontend telemetry.
- Retrieval and most deterministic operations run locally (fast and low-cost).

## 6) Current scope boundary

- Domain is intentionally constrained to consumer electronics categories in dataset.
- This is a demo assistant for recommendation + decision support, not a live e-commerce transaction system.
- Checkout is simulated for closed-loop demonstration.

