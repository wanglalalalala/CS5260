"""Adapter from the frontend contract to the backend ShoppingAgent.

The Streamlit UI expects a dict shaped like `mock_agent.generate_reply`:
    {assistant_reply, applied_filters, recommended_items,
     reasoning_trace, tool_calls, usage}

`ShoppingAgent.chat()` returns `TurnResult(reply, route, debug)` plus mutates
its internal `DialogueState`. This module bridges the two and exposes the
same `generate_reply` / `stream_text` API as the mock backend so `app.py`
can swap implementations with a one-line import change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


_AGENT_KEY = "_real_shopping_agent"
_USAGE_BASELINE_KEY = "_real_usage_baseline"


def _get_agent():
    from agent.dialogue.orchestrator import ShoppingAgent

    agent = st.session_state.get(_AGENT_KEY)
    if agent is None:
        agent = ShoppingAgent()
        st.session_state[_AGENT_KEY] = agent
    return agent


def reset_agent() -> None:
    st.session_state.pop(_AGENT_KEY, None)
    st.session_state.pop(_USAGE_BASELINE_KEY, None)


def _slots_to_filters(state) -> List[str]:
    tags: List[str] = []
    if state.category:
        tags.append(f"Category = {state.category}")
    if state.subcategory:
        tags.append(f"Subcategory = {state.subcategory}")
    if state.brand:
        tags.append(f"Brand = {state.brand}")
    if state.max_price is not None:
        tags.append(f"Budget <= ${int(state.max_price)}")
    elif getattr(state, "budget_skipped", False):
        tags.append("Budget: open")
    if state.min_price is not None:
        tags.append(f"Budget >= ${int(state.min_price)}")
    if state.min_rating is not None:
        tags.append(f"Rating >= {state.min_rating}")
    if state.use_case:
        tags.append(f"Use case: {state.use_case}")
    for k, v in (state.required_specs or {}).items():
        tags.append(f"{k}: {v}")
    return tags


def _short_reason(p: Dict[str, Any]) -> str:
    """Prefer the first sentence of the product description as the sidebar
    blurb (real content), falling back to category/relevance metadata."""
    desc = (p.get("description") or "").strip()
    if desc:
        first = desc.split(". ")[0].strip().rstrip(".")
        if len(first) > 140:
            first = first[:137].rstrip() + "…"
        if len(first) >= 20:
            return first + "."
    return (
        p.get("subcategory")
        or p.get("main_category")
        or f"Relevance {p.get('relevance_score', 0):.2f}"
    )


def _products_to_cards(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards = []
    for p in products[:5]:
        price = p.get("price")
        rating = p.get("rating") or 0.0
        pid = p.get("id") or ""
        cards.append(
            {
                "id": pid,
                "name": p.get("title") or p.get("name") or "Unnamed product",
                "brand": p.get("brand") or "Unknown",
                "price": float(price) if price else 0.0,
                "rating": float(rating),
                "short_reason": _short_reason(p),
                "price_is_estimate": bool(p.get("price_is_estimate")),
                "url": f"https://www.amazon.com/dp/{pid}" if pid else "",
            }
        )
    return cards


def _usage_delta() -> Dict[str, int]:
    """Pull the token delta since the last turn from the backend USAGE ledger."""
    from agent.dialogue.llm import USAGE

    baseline = st.session_state.get(_USAGE_BASELINE_KEY) or {"in": 0, "out": 0}
    current_in = USAGE.input_tokens
    current_out = USAGE.output_tokens

    delta = {
        "prompt_tokens": max(0, current_in - baseline["in"]),
        "completion_tokens": max(0, current_out - baseline["out"]),
    }
    delta["total_tokens"] = delta["prompt_tokens"] + delta["completion_tokens"]

    st.session_state[_USAGE_BASELINE_KEY] = {"in": current_in, "out": current_out}
    return delta


def _escape_dollars(text: str) -> str:
    """Escape unescaped `$` so Streamlit's Markdown does not render price
    ranges as LaTeX math."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            out.append("\\$")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def generate_reply(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    agent = _get_agent()
    turn = agent.chat(user_message)

    state = agent.state
    products = list(state.last_products or [])
    filters = _slots_to_filters(state)

    reasoning_trace = [
        f"Supervisor routed to: {turn.route}",
    ]
    debug = turn.debug or {}
    if "rewritten_query" in debug:
        reasoning_trace.append(f"Rewritten query: {debug['rewritten_query']}")
    if "retrieved_count" in debug:
        reasoning_trace.append(
            f"Retrieved {debug['retrieved_count']} candidates, "
            f"kept {debug.get('filtered_count', len(products))} after filters."
        )
    if "constraints_applied" in debug:
        reasoning_trace.append(f"Constraints applied: {debug['constraints_applied']}")

    tool_calls = []
    if turn.route == "search":
        tool_calls.append(
            {
                "name": "rag.search",
                "args": {
                    "query": debug.get("rewritten_query", user_message),
                    "constraints": debug.get("constraints_applied", {}),
                },
                "result": {
                    "retrieved_count": debug.get("retrieved_count", 0),
                    "filtered_count": debug.get("filtered_count", len(products)),
                },
                "status": "success",
            }
        )
    elif turn.route in {"compare", "checkout", "detail"}:
        tool_calls.append(
            {
                "name": f"rag.{turn.route}",
                "args": {k: v for k, v in debug.items() if k != "agent"},
                "result": "see reply",
                "status": "success",
            }
        )

    raw = debug.get("suggestions") or []
    suggestions = [s for s in raw if isinstance(s, str) and s.strip()]

    return {
        "assistant_reply": _escape_dollars(turn.reply or ""),
        "applied_filters": filters,
        "recommended_items": _products_to_cards(products),
        "reasoning_trace": reasoning_trace,
        "tool_calls": tool_calls,
        "usage": _usage_delta(),
        "suggestions": suggestions,
    }


def stream_text(text: str) -> Iterable[str]:
    for token in text.split(" "):
        yield token + " "
