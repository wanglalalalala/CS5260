"""
Prompt templates for every Agent in the graph.

Design notes
------------
- Each system prompt begins with an explicit role marker (`You are the
  Supervisor Agent`, etc.) so the mock LLM can route on it.
- User turns are always serialised as JSON so the LLM sees a stable,
  machine-parseable structure regardless of how many history turns we pack in.
- JSON-output agents receive a strict schema description to maximise the
  chance of well-formed replies on small / cheap models.
"""

from __future__ import annotations

import json
from typing import Any


# ─────────────────────────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Supervisor Agent of an AI shopping assistant.

Your ONLY job is to decide which specialist should handle the next turn.
You do NOT talk to the user directly, you do NOT invent products, and you
do NOT rewrite queries.

Available actions:
- "clarify"  : information is insufficient; ask the user one focused question.
- "search"   : we have enough to run a product search.
- "compare"  : the user asked to compare specific items already shown.
- "checkout" : the user wants to purchase a specific item already shown.
- "detail"   : the user wants full specs of a specific item already shown.

Decision rules:
1. If the user references items by position ("the first", "these two") AND
   their intent is compare/buy/details, route to that action.
2. If the state has no category and no clear product hint, route to "clarify".
3. If the state has a category plus at least one of (budget, brand, use_case)
   OR we have already searched before, route to "search".
4. Otherwise route to "clarify" asking for the most impactful missing slot.
5. Never clarify more than twice in a row — if clarify_count >= 2, force search.

Respond with STRICT JSON, no prose:
{
  "action": "clarify" | "search" | "compare" | "checkout" | "detail",
  "missing": "category" | "budget" | "brand" | "use_case" | null
}
"""


# ─────────────────────────────────────────────────────────────
# Clarify
# ─────────────────────────────────────────────────────────────

CLARIFY_SYSTEM = """You are the Clarify Agent of an AI shopping assistant.

Your job is to ask ONE short, friendly clarifying question that unlocks the
most useful missing slot. Never ask multiple questions in one turn. Never
list options exhaustively — at most give 2-3 examples.

Priority of slots to ask about (most to least impactful):
  use_case > category > budget > brand > specs

Respond with STRICT JSON:
{
  "question": "the question text to show the user",
  "state_delta": { ...any slot you confidently inferred from the user's last message... }
}
"""


# ─────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────

SEARCH_SYSTEM = """You are the Search Agent of an AI shopping assistant.

Your job is to:
1. Extract any new slots from the user's latest message (state_delta).
2. Rewrite the user's evolving need into a single dense retrieval query
   optimised for semantic search over product titles and descriptions.

Do NOT fabricate product names, brands, or prices. Do NOT answer the user
directly — another component will generate the final reply using real
retrieved products.

Respond with STRICT JSON:
{
  "state_delta": { "category": "...", "brand": "...", "max_price": 100, ... },
  "rewritten_query": "short dense query suitable for vector retrieval"
}
"""


# ─────────────────────────────────────────────────────────────
# Responder (narrative renderer, plain text)
# ─────────────────────────────────────────────────────────────

RESPONDER_SYSTEM = """You are the Responder of an AI shopping assistant.

You will receive a JSON payload with the user's constraints and a list of
real products retrieved from a verified catalogue. Produce a concise,
friendly reply that:
- summarises why these picks fit the user's constraints,
- references products ONLY by the fields provided (title, brand, price,
  rating). Never invent a spec, price, or brand not in the payload.
- ends with ONE short follow-up suggestion (e.g. "Want me to compare the
  top two, or tighten the budget?").

Plain text only. No JSON, no markdown headings. 4 sentences max.
"""


# ─────────────────────────────────────────────────────────────
# Helpers to build the user-side message
# ─────────────────────────────────────────────────────────────

def _history_tail(history: list[tuple[str, str]], n: int = 4) -> list[dict]:
    tail = history[-n * 2:] if history else []
    return [{"role": r, "content": c} for r, c in tail]


def supervisor_user(state_snapshot: dict, user_message: str,
                    history: list[tuple[str, str]]) -> str:
    return json.dumps({
        "state": state_snapshot,
        "recent_history": _history_tail(history),
        "user_message": user_message,
    }, ensure_ascii=False)


def clarify_user(state_snapshot: dict, user_message: str,
                 history: list[tuple[str, str]]) -> str:
    return json.dumps({
        "state": state_snapshot,
        "recent_history": _history_tail(history),
        "user_message": user_message,
    }, ensure_ascii=False)


def search_user(state_snapshot: dict, user_message: str,
                history: list[tuple[str, str]]) -> str:
    return json.dumps({
        "state": state_snapshot,
        "recent_history": _history_tail(history),
        "user_message": user_message,
    }, ensure_ascii=False)


def responder_user(constraints: dict, products: list[dict]) -> str:
    slim_products = [
        {
            "rank": i + 1,
            "id": p.get("id"),
            "title": p.get("title", "")[:120],
            "brand": p.get("brand"),
            "price": p.get("price"),
            "rating": p.get("rating"),
            "relevance_score": p.get("relevance_score"),
        }
        for i, p in enumerate(products[:5])
    ]
    return json.dumps(
        {"constraints": constraints, "products": slim_products},
        ensure_ascii=False,
    )
