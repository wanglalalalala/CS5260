"""
Search Agent.
=============

Extracts remaining slots from the user's latest turn, rewrites the need into
a dense retrieval query, invokes the RAG pipeline, and asks the responder
to narrate the results. Products are ALWAYS grounded in retrieved data.
"""

from __future__ import annotations

from statistics import median

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


def _has_valid_price(value) -> bool:
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False


def _fallback_price_from_title(title: str) -> float:
    lower = (title or "").lower()
    # Coarse heuristics for when all retrieved items are missing price.
    if any(k in lower for k in ("ultra", "pro", "max", "flagship", "fold")):
        return 599.0
    if any(k in lower for k in ("budget", "lite", "mini", "entry")):
        return 149.0
    if any(k in lower for k in ("speaker", "earbud", "headphone")):
        return 129.0
    if "laptop" in lower:
        return 799.0
    if any(k in lower for k in ("phone", "smartphone", "android", "iphone")):
        return 349.0
    return 299.0


def _inject_estimated_prices(products: list[dict]) -> None:
    """Fill missing prices so the UI never shows N/A."""
    observed_prices: list[float] = []
    for p in products:
        if _has_valid_price(p.get("price")):
            observed_prices.append(float(p["price"]))

    baseline = float(median(observed_prices)) if observed_prices else None
    for p in products:
        if _has_valid_price(p.get("price")):
            p["price_is_estimate"] = False
            continue
        guess = baseline if baseline is not None else _fallback_price_from_title(p.get("title", ""))
        p["price"] = round(float(guess), 2)
        p["price_is_estimate"] = True


def _sanitize_suggestions(raw: list) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text) > 40:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(text)
        if len(suggestions) >= 3:
            break
    return suggestions


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

        # Step 4 — enrich the top-3 with description from SQLite so the
        # responder can write grounded AI commentary (not just list fields).
        suggestions: list[str] = []
        if products:
            from rag.retriever import get_products_by_ids

            _inject_estimated_prices(products)

            top_ids = [p["id"] for p in products[:3]]
            full = {r["id"]: r for r in get_products_by_ids(top_ids)}
            for p in products[:3]:
                record = full.get(p["id"]) or {}
                desc = (record.get("description") or "")[:400]
                if desc:
                    p["description"] = desc

            responder = self.llm.call_json(
                system=RESPONDER_SYSTEM,
                user=responder_user(result.constraints_applied, products),
            )
            body = str(responder.get("body_markdown") or "").strip()
            follow_up = str(responder.get("follow_up") or "").strip()
            suggestions = _sanitize_suggestions(responder.get("suggestions") or [])
            if not suggestions:
                suggestions = [
                    "Compare #1 and #2",
                    "Show more under $300",
                    "Prefer a different brand",
                ]
            if not follow_up:
                follow_up = "Do any of these stand out, or should I refine by budget, brand, or a head-to-head comparison?"
            if body:
                reply = f"{body}\n\n{follow_up}"
            else:
                # Graceful fallback if JSON mode fails on smaller models.
                reply = self.llm.call_text(
                    system=RESPONDER_SYSTEM,
                    user=responder_user(result.constraints_applied, products),
                )
        else:
            reply = _NO_RESULTS_MSG
            suggestions = [
                "Raise the budget a bit",
                "Try a different brand",
                "Switch product type",
            ]

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
                "suggestions": suggestions,
            },
        )
