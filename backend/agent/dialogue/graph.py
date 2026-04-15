"""
LangGraph multi-agent orchestration.
====================================

Node layout:

    entry → supervisor ──► clarify  ──► END
                       │
                       ├► search   ──► END
                       │
                       ├► compare  ──► END   (deterministic tool node)
                       ├► checkout ──► END   (deterministic tool node)
                       └► detail   ──► END   (deterministic tool node)

The state object carried by the graph is a plain dict (LangGraph requirement)
that we project back to / from our `DialogueState` dataclass at the boundary.
"""

from __future__ import annotations

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from .state import DialogueState, merge_delta
from .agents import SupervisorAgent, ClarifyAgent, SearchAgent


# ─────────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    dialogue: DialogueState       # live dialogue state (mutated in place)
    user_message: str             # current-turn input
    route: str                    # action chosen by Supervisor
    referenced_ids: list[str]     # resolved IDs for direct-action routes
    reply: str                    # final user-facing reply
    debug: dict                   # per-turn diagnostics


# ─────────────────────────────────────────────────────────────
# Deterministic tool nodes (compare / checkout / detail)
# ─────────────────────────────────────────────────────────────

def _compare_node(gs: GraphState) -> GraphState:
    from rag.retriever import get_products_by_ids
    from .llm import get_client
    from .prompts import (
        COMPARE_SUMMARY_SYSTEM,
        compare_summary_user,
        _NOISY_SPEC_KEYS,
    )

    ids = gs.get("referenced_ids") or []
    if len(ids) < 2:
        gs["reply"] = (
            "I need at least two products to compare. "
            "Tell me which ones — e.g. 'compare the first two'."
        )
        return gs

    products = get_products_by_ids(ids)
    if len(products) < 2:
        gs["reply"] = "Couldn't load enough product details to compare."
        gs["debug"] = {"compared_ids": ids}
        return gs

    table_md = _render_compare_table_md(products, _NOISY_SPEC_KEYS)

    try:
        summary = get_client().call_text(
            system=COMPARE_SUMMARY_SYSTEM,
            user=compare_summary_user(products),
        )
    except Exception:
        summary = ""

    parts = [table_md]
    if summary.strip():
        parts.append(summary.strip())
    parts.append("_Reply with 'buy the first one' or ask me to tighten the search._")

    gs["reply"] = "\n\n".join(parts)
    gs["debug"] = {"compared_ids": ids}
    return gs


def _checkout_node(gs: GraphState) -> GraphState:
    from rag.pipeline import checkout

    ids = gs.get("referenced_ids") or []
    if not ids:
        gs["reply"] = "Which item would you like to order?"
        return gs

    receipt = checkout(ids[0])
    if receipt.get("success"):
        gs["reply"] = (
            f"Order confirmed — {receipt['product_title']}.\n"
            f"Order ID: {receipt['order_id']}. "
            f"Anything else you'd like to look at?"
        )
    else:
        gs["reply"] = receipt.get("message", "Could not place the order.")
    gs["debug"] = {"receipt": receipt}
    return gs


def _detail_node(gs: GraphState) -> GraphState:
    from rag.pipeline import detail

    ids = gs.get("referenced_ids") or []
    if not ids:
        gs["reply"] = "Which product would you like the full details for?"
        return gs

    record = detail(ids[0])
    gs["reply"] = _render_detail(record)
    gs["debug"] = {"product_id": ids[0]}
    return gs


# ─────────────────────────────────────────────────────────────
# LLM-backed nodes
# ─────────────────────────────────────────────────────────────

def _make_supervisor_node(agent: SupervisorAgent):
    def node(gs: GraphState) -> GraphState:
        out = agent.run(gs["dialogue"], gs["user_message"])
        gs["route"] = out.next_action or "clarify"
        gs["referenced_ids"] = out.payload.get("referenced_ids", [])
        return gs
    return node


def _make_clarify_node(agent: ClarifyAgent):
    def node(gs: GraphState) -> GraphState:
        state = gs["dialogue"]
        out = agent.run(state, gs["user_message"])
        merge_delta(state, out.state_delta)
        state.clarify_count += 1
        gs["reply"] = out.reply or ""
        gs["debug"] = {
            "agent": "clarify",
            "suggestions": out.payload.get("suggestions", []),
        }
        return gs
    return node


def _make_search_node(agent: SearchAgent):
    def node(gs: GraphState) -> GraphState:
        state = gs["dialogue"]
        out = agent.run(state, gs["user_message"])
        merge_delta(state, out.state_delta)
        state.clarify_count = 0
        state.has_searched = True
        state.last_products = out.payload.get("products", [])
        gs["reply"] = out.reply or ""
        gs["debug"] = {"agent": "search", **{
            k: v for k, v in out.payload.items() if k != "products"
        }}
        return gs
    return node


# ─────────────────────────────────────────────────────────────
# Routing function
# ─────────────────────────────────────────────────────────────

def _router(gs: GraphState) -> str:
    route = gs.get("route", "clarify")
    return route if route in {"clarify", "search", "compare", "checkout", "detail"} else "clarify"


# ─────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────

def build_graph():
    supervisor = SupervisorAgent()
    clarify = ClarifyAgent()
    searcher = SearchAgent()

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", _make_supervisor_node(supervisor))
    graph.add_node("clarify",    _make_clarify_node(clarify))
    graph.add_node("search",     _make_search_node(searcher))
    graph.add_node("compare",    _compare_node)
    graph.add_node("checkout",   _checkout_node)
    graph.add_node("detail",     _detail_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", _router, {
        "clarify":  "clarify",
        "search":   "search",
        "compare":  "compare",
        "checkout": "checkout",
        "detail":   "detail",
    })

    for terminal in ("clarify", "search", "compare", "checkout", "detail"):
        graph.add_edge(terminal, END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Lightweight renderers for tool nodes
# ─────────────────────────────────────────────────────────────

def _render_compare_table_md(products: list[dict], noisy_keys: set[str]) -> str:
    """Render a clean side-by-side Markdown table for 2-3 products."""
    if not products:
        return "No overlap found to compare."

    def _label(p: dict) -> str:
        brand = (p.get("brand") or "").strip() or "—"
        model = (p.get("model") or "").strip()
        return f"{brand} {model}".strip() or "Product"

    def _fmt_price(v) -> str:
        if v is None or v == "" or v == -1:
            return "N/A"
        try:
            return f"\\${float(v):.2f}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_rating(p: dict) -> str:
        r = p.get("rating")
        n = p.get("rating_count") or 0
        if not r:
            return "—"
        return f"⭐ {float(r):.1f} ({int(n)})"

    def _truncate(v, width: int = 42) -> str:
        s = str(v) if v not in (None, "") else "—"
        return s if len(s) <= width else s[: width - 1] + "…"

    headers = ["Field"] + [_label(p) for p in products]
    rows: list[list[str]] = []
    rows.append(["Title"] + [_truncate(p.get("title"), 60) for p in products])
    rows.append(["Price"] + [_fmt_price(p.get("price")) for p in products])
    rows.append(["Rating"] + [_fmt_rating(p) for p in products])

    # Spec rows: only include keys that appear in at least 2 products and
    # are not in the noise list.
    from collections import Counter
    counts: Counter = Counter()
    for p in products:
        specs = p.get("specifications") or {}
        for k in specs:
            if k.lower() in noisy_keys:
                continue
            counts[k] += 1
    shared_keys = [k for k, c in counts.items() if c >= 2][:6]
    for key in shared_keys:
        row = [key]
        for p in products:
            specs = p.get("specifications") or {}
            row.append(_truncate(specs.get(key, "—")))
        rows.append(row)

    # Build Markdown table.
    lines = ["**Side-by-side comparison:**", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _render_detail(record: dict) -> str:
    if "error" in record:
        return record["error"]

    title = record.get("title", "")
    brand = record.get("brand", "")
    price = record.get("price")
    rating = record.get("rating")
    specs = record.get("specifications") or {}

    head = f"{title}"
    meta = f"Brand: {brand or 'N/A'}  |  " \
           f"Price: {'$'+format(price,'.2f') if price else 'N/A'}  |  " \
           f"Rating: {rating if rating else 'N/A'}"
    lines = [head, meta, ""]

    if specs:
        lines.append("Key specs:")
        for k, v in list(specs.items())[:8]:
            lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Would you like to buy it, compare it with another, or keep browsing?")
    return "\n".join(lines)
