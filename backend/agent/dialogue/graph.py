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
    from rag.pipeline import compare

    ids = gs.get("referenced_ids") or []
    if len(ids) < 2:
        gs["reply"] = (
            "I need at least two products to compare. "
            "Tell me which ones — e.g. 'compare the first two'."
        )
        return gs

    table = compare(ids)
    gs["reply"] = _render_compare_table(table)
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
        gs["debug"] = {"agent": "clarify"}
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

def _render_compare_table(table: dict) -> str:
    cols = table.get("columns", [])
    rows = table.get("rows", [])
    if not cols or not rows:
        return "No overlap found to compare."

    lines = ["Here's a side-by-side look:"]
    for row in rows[:10]:
        field = row.get("field", "")
        values = " | ".join(str(row.get(c, ""))[:40] for c in cols)
        lines.append(f"- {field}: {values}")
    lines.append("Want me to recommend one, or narrow the specs further?")
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
