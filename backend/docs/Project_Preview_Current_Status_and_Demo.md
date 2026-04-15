# Project Preview: Current Status + Demo Guide (Group 30)

## 1) Current project status (preview snapshot)

This preview build is a working end-to-end AI shopping assistant with:

- multi-turn dialogue management
- clarification-first interaction
- semantic product retrieval with hard constraints
- deterministic compare/detail/checkout actions
- interactive Streamlit frontend
- token usage + cost telemetry

In short: the core "ask -> clarify -> retrieve -> explain -> compare/checkout" loop is functional and demo-ready.

## 2) Proposal alignment checklist

Reference: `CS5260_Proposal.pdf`

### Objective A: Develop an autonomous agent

**Status: Implemented (core)**

- Dialogue state is persisted across turns.
- Supervisor route selection is implemented (`clarify/search/compare/detail/checkout`).
- Clarify agent asks focused follow-ups when constraints are missing.
- Search agent rewrites query + applies structured constraints.
- Deterministic tool paths are integrated for compare/detail/checkout.

### Objective B: Interactive front-end

**Status: Implemented**

- Streamlit chat UI is integrated with real backend agent.
- Category starter cards + suggestion chips improve interaction speed.
- Reasoning/tool-call panel is available for explainability during demo.
- UX has been upgraded (chat bubble separation, styling system, clearer composition guidance).

### Objective C: Latency and cost efficiency

**Status: Implemented (initial), still being tuned**

- Lightweight model targets are supported (`qwen-plus`, `gpt-4o-mini`).
- Token/cost logging is implemented and visible in UI sidebar.
- Retrieval + filtering remain local and deterministic.
- Ongoing tuning: prompt compactness and routing precision for fewer unnecessary calls.

### Core features in proposal

1. **Dynamic intent recognition + state tracking** -> Implemented  
2. **Semantic retrieval via RAG** -> Implemented  
3. **Deterministic tool calling** -> Implemented  
4. **Explainable recommendations** -> Implemented (reasoning panel + grounded response format)

### Evaluation dimensions in proposal

- **Tool invocation accuracy**: basic flow implemented, currently validated via manual/integration testing.
- **Retrieval relevance & grounding**: retrieval and deterministic constraints are active; iterative quality tuning ongoing.
- **Dialogue coherence & completion**: multi-turn flow works; clarification and follow-up behavior improved.
- **Cost/token efficiency**: telemetry implemented; optimization is ongoing.

## 3) What is already beyond the original proposal

Additional improvements implemented during iteration:

- Suggestion chips for both clarify and search turns (not only text-only follow-up).
- Dynamic follow-up strategy in recommendation output (instead of fixed hardcoded endings).
- UI/UX redesign with clearer assistant/user separation and improved readability.
- Better post-search refinement UX (budget/brand/compare next actions).
- Missing-price handling now estimates a reasonable price instead of raw `N/A` in recommendation output.

## 4) Suggested demo flow for TAs (8–10 minutes)

## Demo goal

Show full-loop capability from vague request to comparison and action, while highlighting grounding and tool usage.

### Demo script

1. **Start broad query**
   - Example: "I need an action camera."
   - Expected: agent asks a clarifying question (budget/brand/type).

2. **Add hard constraints**
   - Example: "Under $300, Sony preferred."
   - Expected: search route triggers, returns ranked recommendations with explanations.

3. **Use quick interaction**
   - Click a suggestion chip (e.g., compare top two / tighten budget).
   - Expected: follow-up action executes without retyping.

4. **Compare products**
   - Example: "Compare #1 and #2."
   - Expected: side-by-side comparison output + concise recommendation.

5. **Inspect one product**
   - Example: "Show details of the first one."
   - Expected: detail route returns structured product details.

6. **Simulate final step**
   - Example: "Buy the first one."
   - Expected: checkout simulation returns confirmation/order id.

7. **Explainability + cost**
   - Open "Show reasoning" panel and sidebar token/cost section.
   - Expected: clear evidence of route/tool behavior and approximate usage.

## 5) Demo points to explicitly mention during presentation

- Why we separate LLM reasoning from deterministic tools.
- How constraints are enforced (price/brand/etc.) in retrieval pipeline.
- Why this design reduces hallucination risk compared to pure chatbots.
- How multi-turn state tracking improves task completion over one-shot search.
- How token/cost monitoring supports the "lean inference" objective.

## 6) Known limitations in this preview build

- Dataset is snapshot-based (not live marketplace inventory).
- Some products have sparse metadata (especially price/specs).
- Tool-accuracy and relevance evaluation are not yet in a fully automated benchmark dashboard.
- UI is functionally strong but still polishable for final presentation.

## 7) Next steps before final checkpoint

1. Add a small evaluation harness for route/tool precision and retrieval relevance.
2. Improve ranking and tie-break logic for mixed-quality catalog entries.
3. Finalize consistent prompt policy for recommendation tone and follow-up behavior.
4. Freeze demo scenarios and prepare deterministic backup examples.
5. Package a clean submission zip with group-number naming rule required by assignment.

