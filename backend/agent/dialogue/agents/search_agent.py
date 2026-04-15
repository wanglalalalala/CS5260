"""
Search Agent.
=============

Extracts remaining slots from the user's latest turn, rewrites the need into
a dense retrieval query, invokes the RAG pipeline, and asks the responder
to narrate the results. Products are ALWAYS grounded in retrieved data.
"""

from __future__ import annotations

from .base import BaseAgent, AgentOutput
from ..state import DialogueState, to_constraints
from ..llm import get_client
from ..prompts import (
    SEARCH_SYSTEM,
    RESPONDER_SYSTEM,
    search_user,
    responder_user,
)

# Import lazily inside methods to avoid a hard dependency on the RAG layer
# when this module is imported for unit tests.


_NO_RESULTS_MSG = (
    "I couldn't find products matching those constraints. "
    "Want to relax the budget, change the brand, or try a different category?"
)


class SearchAgent(BaseAgent):
    name = "search"

    def __init__(self):
        self.llm = get_client()

    def run(self, state: DialogueState, user_message: str) -> AgentOutput:
        from rag.pipeline import search as rag_search

        # Step 1 — LLM extracts new slots + rewrites the query.
        plan = self.llm.call_json(
            system=SEARCH_SYSTEM,
            user=search_user(state.snapshot(), user_message, state.history),
        )
        delta = plan.get("state_delta") or {}
        rewritten = plan.get("rewritten_query") or user_message

        # Step 2 — apply the delta to a provisional state snapshot so the
        # constraints passed to RAG reflect the freshest information.
        provisional = DialogueState(**state.to_dict())
        from ..state import merge_delta
        merge_delta(provisional, delta)

        constraints = to_constraints(provisional, top_k=20, limit=5)

        # Step 3 — run the deterministic RAG pipeline.
        result = rag_search(rewritten, constraints=constraints)
        products = result.products

        # Step 4 — narrate via the responder LLM (grounded in real products).
        if products:
            reply = self.llm.call_text(
                system=RESPONDER_SYSTEM,
                user=responder_user(result.constraints_applied, products),
            )
        else:
            reply = _NO_RESULTS_MSG

        return AgentOutput(
            state_delta={
                **delta,
                # last_products is stored separately by the orchestrator.
            },
            reply=reply,
            next_action="search",
            payload={
                "rewritten_query": rewritten,
                "retrieved_count": result.retrieved_count,
                "filtered_count": result.filtered_count,
                "products": products,
                "constraints_applied": result.constraints_applied,
            },
        )
