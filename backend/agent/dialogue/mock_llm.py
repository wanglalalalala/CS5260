"""
Deterministic mock LLM for offline development & CI.
====================================================

Pattern-matches on the user/system content to return plausible JSON or text
so the whole multi-agent graph can be exercised without an API key.
"""

from __future__ import annotations

import json
import re


_CATEGORY_KEYWORDS = {
    "headphone": "All Electronics",
    "earbud": "All Electronics",
    "phone": "Cell Phones & Accessories",
    "smartphone": "Cell Phones & Accessories",
    "case": "Cell Phones & Accessories",
    "charger": "Cell Phones & Accessories",
    "cable": "Cell Phones & Accessories",
    "laptop": "Computers",
    "keyboard": "Computers",
    "mouse": "Computers",
    "camera": "Camera & Photo",
    "tablet": "Computers",
}


def _extract_category(text: str) -> str | None:
    lower = text.lower()
    for kw, cat in _CATEGORY_KEYWORDS.items():
        if kw in lower:
            return cat
    return None


def _extract_price(text: str) -> tuple[float | None, float | None]:
    lower = text.lower()
    max_p, min_p = None, None

    m = re.search(r"under\s*\$?(\d+)", lower) or re.search(r"below\s*\$?(\d+)", lower)
    if m:
        max_p = float(m.group(1))

    m = re.search(r"over\s*\$?(\d+)", lower) or re.search(r"above\s*\$?(\d+)", lower)
    if m:
        min_p = float(m.group(1))

    m = re.search(r"\$?(\d+)\s*-\s*\$?(\d+)", lower)
    if m:
        min_p = float(m.group(1))
        max_p = float(m.group(2))

    m = re.search(r"budget[^0-9]*\$?(\d+)", lower)
    if m and max_p is None:
        max_p = float(m.group(1))

    return max_p, min_p


_KNOWN_BRANDS = [
    "apple", "samsung", "sony", "anker", "bose", "logitech",
    "dell", "hp", "asus", "lenovo", "google", "microsoft",
]


def _extract_brand(text: str) -> str | None:
    lower = text.lower()
    for b in _KNOWN_BRANDS:
        if b in lower:
            return b.capitalize()
    return None


def _detect_action(text: str) -> str | None:
    lower = text.lower()
    if any(w in lower for w in ("compare", "vs", "versus", "difference")):
        return "compare"
    if any(w in lower for w in ("buy", "purchase", "order", "check out", "checkout")):
        return "checkout"
    if any(w in lower for w in ("detail", "spec", "tell me more", "more info")):
        return "detail"
    return None


# ─────────────────────────────────────────────────────────────
# Main entry points consumed by LLMClient
# ─────────────────────────────────────────────────────────────

def mock_json_response(system: str, user: str) -> dict:
    """Route on system-prompt identity markers embedded by prompts.py."""
    s_lower = system.lower()

    if "supervisor" in s_lower:
        return _mock_supervisor(user)
    if "clarify" in s_lower or "ask a clarifying" in s_lower:
        return _mock_clarify(user)
    if "search" in s_lower and "rewrite" in s_lower:
        return _mock_search(user)
    return {}


def mock_text_response(system: str, user: str) -> str:
    """Used by the responder narrative renderer."""
    return (
        "Based on your needs, here are the top picks ranked by relevance and rating. "
        "Let me know if you'd like to compare them, see more details, or refine the search."
    )


# ─────────────────────────────────────────────────────────────
# Per-agent mock logic
# ─────────────────────────────────────────────────────────────

def _mock_supervisor(user: str) -> dict:
    """Decide routing: clarify vs search vs direct tool actions."""
    payload = _parse_user_block(user)
    msg = payload.get("user_message", "")
    state = payload.get("state", {})

    # Direct-action keywords only fire if we actually have products to act on.
    has_products = state.get("num_last_products", 0) > 0
    direct = _detect_action(msg)
    if direct in ("compare", "checkout", "detail") and has_products:
        return {"action": direct}

    # If the state already carries a category (or user just mentioned one)
    # and we have at least a price OR a brand OR a use case → search.
    cat = state.get("category") or _extract_category(msg)
    brand = state.get("brand") or _extract_brand(msg)
    max_p_new, min_p_new = _extract_price(msg)
    max_p = state.get("max_price") or max_p_new
    use_case = state.get("use_case")

    if cat and (max_p is not None or brand or use_case or state.get("has_searched")):
        return {"action": "search"}

    if not cat:
        return {"action": "clarify", "missing": "category"}

    return {"action": "clarify", "missing": "budget"}


def _mock_clarify(user: str) -> dict:
    payload = _parse_user_block(user)
    msg = payload.get("user_message", "")
    state = payload.get("state", {})

    # Absorb any slot the user just supplied, even while clarifying.
    delta: dict = {}
    cat = _extract_category(msg)
    if cat and not state.get("category"):
        delta["category"] = cat
    brand = _extract_brand(msg)
    if brand and not state.get("brand"):
        delta["brand"] = brand
    max_p, min_p = _extract_price(msg)
    if max_p is not None and state.get("max_price") is None:
        delta["max_price"] = max_p
    if min_p is not None and state.get("min_price") is None:
        delta["min_price"] = min_p

    merged = {**state, **delta}

    if not merged.get("category"):
        q = "What kind of product are you shopping for today? (e.g. headphones, laptop, phone accessories)"
    elif merged.get("max_price") is None:
        q = "Do you have a budget in mind? Even a rough ceiling helps me narrow things down."
    elif not merged.get("brand"):
        q = "Any brand preference, or should I show options across brands?"
    else:
        q = "Anything specific about features or use case I should factor in?"

    return {"question": q, "state_delta": delta}


def _mock_search(user: str) -> dict:
    payload = _parse_user_block(user)
    msg = payload.get("user_message", "")
    state = payload.get("state", {})

    delta: dict = {}

    cat = _extract_category(msg)
    if cat and not state.get("category"):
        delta["category"] = cat

    brand = _extract_brand(msg)
    if brand and not state.get("brand"):
        delta["brand"] = brand

    max_p, min_p = _extract_price(msg)
    if max_p is not None and state.get("max_price") is None:
        delta["max_price"] = max_p
    if min_p is not None and state.get("min_price") is None:
        delta["min_price"] = min_p

    # Build a simple rewritten query out of whatever we know.
    pieces = []
    if state.get("brand") or brand:
        pieces.append(str(state.get("brand") or brand))
    if state.get("category") or cat:
        pieces.append(str(state.get("category") or cat))
    pieces.append(msg.strip())
    rewritten = " ".join(p for p in pieces if p).strip()

    return {"state_delta": delta, "rewritten_query": rewritten or msg}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse_user_block(user: str) -> dict:
    """
    prompts.py always serialises the user turn as a JSON object so both
    real LLMs and the mock can read the same structure.
    """
    try:
        return json.loads(user)
    except (json.JSONDecodeError, TypeError):
        return {"user_message": user, "state": {}}
