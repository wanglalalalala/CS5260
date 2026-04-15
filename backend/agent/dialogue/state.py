"""
Dialogue State Tracker (DST) data structures.
=============================================

The `DialogueState` is the single source of truth shared across every Agent
in the graph. Each Agent reads the current state, emits a `state_delta`
(a partial dict describing only what changed), and the orchestrator merges
the delta back into the state.

All slots are aligned one-to-one with `rag.pipeline.SearchConstraints`, so
`to_constraints()` is a near-trivial projection that lets the RAG layer
stay fully decoupled from the dialogue layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# ─────────────────────────────────────────────────────────────
# Dialogue state
# ─────────────────────────────────────────────────────────────

@dataclass
class DialogueState:
    """Accumulated user preferences and dialogue context across turns."""

    # --- shopping slots (mirror SearchConstraints) ---
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    use_case: Optional[str] = None           # e.g. "gift for elderly"
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_rating: Optional[float] = None
    required_specs: dict[str, str] = field(default_factory=dict)

    # --- dialogue control ---
    history: list[tuple[str, str]] = field(default_factory=list)   # [(role, text), ...]
    last_products: list[dict] = field(default_factory=list)        # most recent search result
    referenced_ids: list[str] = field(default_factory=list)        # resolved from "the first one" etc.
    clarify_count: int = 0                                         # consecutive clarifications
    has_searched: bool = False
    turn_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def snapshot(self) -> dict:
        """Compact view suitable for LLM context injection (no raw history)."""
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "brand": self.brand,
            "use_case": self.use_case,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "min_rating": self.min_rating,
            "required_specs": self.required_specs,
            "has_searched": self.has_searched,
            "clarify_count": self.clarify_count,
            "num_last_products": len(self.last_products),
        }


# ─────────────────────────────────────────────────────────────
# Delta merging
# ─────────────────────────────────────────────────────────────

_SLOT_KEYS = {
    "category", "subcategory", "brand", "use_case",
    "max_price", "min_price", "min_rating", "required_specs",
}


def merge_delta(state: DialogueState, delta: dict[str, Any] | None) -> DialogueState:
    """
    Merge an LLM-produced partial state into the live DialogueState.

    Only whitelisted slot keys are accepted; unknown keys are silently ignored
    so a hallucinated field cannot corrupt the state.
    """
    if not delta:
        return state

    for key, value in delta.items():
        if key not in _SLOT_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        if key == "required_specs" and isinstance(value, dict):
            state.required_specs.update(value)
        else:
            setattr(state, key, value)

    return state


# ─────────────────────────────────────────────────────────────
# Projection: DialogueState → RAG SearchConstraints (as dict)
# ─────────────────────────────────────────────────────────────

def to_constraints(state: DialogueState, top_k: int = 20, limit: int = 5) -> dict:
    """
    Project the dialogue state into a kwargs dict consumable by
    `rag.pipeline.search(query, constraints=...)`.
    """
    c: dict = {"top_k": top_k, "limit": limit}

    if state.category:
        c["category"] = state.category
    if state.subcategory:
        c["subcategory"] = state.subcategory
    if state.brand:
        c["brand"] = state.brand
    if state.max_price is not None:
        c["max_price"] = float(state.max_price)
    if state.min_price is not None:
        c["min_price"] = float(state.min_price)
    if state.min_rating is not None:
        c["min_rating"] = float(state.min_rating)
    if state.required_specs:
        c["required_specs"] = dict(state.required_specs)

    return c


# ─────────────────────────────────────────────────────────────
# Reference resolution: "the first one" / "the second" / "these two"
# ─────────────────────────────────────────────────────────────

_ORDINAL_MAP = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
}


def resolve_references(state: DialogueState, text: str) -> list[str]:
    """
    Best-effort resolver mapping phrases like 'the first one' or 'these two'
    to product IDs from `state.last_products`.

    Returns an empty list when no reference can be grounded.
    """
    if not state.last_products:
        return []

    lower = text.lower()
    positions: list[int] = []

    # ordinal words / digits
    for word, idx in _ORDINAL_MAP.items():
        if word in lower:
            positions.append(idx)

    # explicit "#1", "#2"
    for i in range(1, len(state.last_products) + 1):
        if f"#{i}" in lower or f"number {i}" in lower:
            positions.append(i)

    # bulk phrases
    if "these two" in lower or "first two" in lower or "both" in lower:
        positions = [1, 2]
    if "all of them" in lower or "all" in lower and "these" in lower:
        positions = list(range(1, len(state.last_products) + 1))

    # deduplicate while preserving order
    seen: set[int] = set()
    positions = [p for p in positions if not (p in seen or seen.add(p))]

    ids: list[str] = []
    for pos in positions:
        if 1 <= pos <= len(state.last_products):
            ids.append(state.last_products[pos - 1]["id"])
    return ids
