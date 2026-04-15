from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass
class Product:
    name: str
    brand: str
    price: float
    rating: float
    short_reason: str


CATALOG: List[Product] = [
    Product(
        name="Sony WH-1000XM5",
        brand="Sony",
        price=349.0,
        rating=4.7,
        short_reason="Excellent noise cancellation for office and travel.",
    ),
    Product(
        name="Anker Soundcore Space One",
        brand="Anker",
        price=99.0,
        rating=4.4,
        short_reason="High value pick with strong battery life.",
    ),
    Product(
        name="Apple AirPods Pro 2",
        brand="Apple",
        price=249.0,
        rating=4.6,
        short_reason="Best choice for Apple ecosystem users.",
    ),
    Product(
        name="JBL Tune 770NC",
        brand="JBL",
        price=129.0,
        rating=4.3,
        short_reason="Balanced sound and lightweight comfort.",
    ),
]


def _extract_budget(text: str) -> float | None:
    match = re.search(r"(?:under|below|<=?)\s*\$?(\d+)", text.lower())
    if match:
        return float(match.group(1))
    return None


def _extract_brand(text: str) -> str | None:
    lowered = text.lower()
    for brand in {"sony", "anker", "apple", "jbl"}:
        if brand in lowered:
            return brand.title()
    return None


def generate_reply(
    user_message: str, history: List[Dict[str, str]] | None = None
) -> Dict[str, object]:
    """Return a mock agent response that mimics final backend shape."""
    history = history or []
    budget = _extract_budget(user_message)
    brand = _extract_brand(user_message)

    filtered = CATALOG
    filters = []

    if budget is not None:
        filtered = [item for item in filtered if item.price <= budget]
        filters.append(f"Budget <= ${int(budget)}")
    if brand:
        filtered = [item for item in filtered if item.brand == brand]
        filters.append(f"Brand = {brand}")

    if not filtered:
        filtered = sorted(CATALOG, key=lambda x: x.rating, reverse=True)[:2]
        filters.append("Fallback recommendation used")

    filtered = sorted(filtered, key=lambda x: (x.rating, -x.price), reverse=True)[:3]
    retrieval_candidates = len(filtered)

    items_text = "\n".join(
        [f"- {item.name} (${item.price:.0f}, rating {item.rating:.1f})" for item in filtered]
    )
    assistant_reply = (
        "I found options that match your request.\n\n"
        "Top recommendations:\n"
        f"{items_text}\n\n"
        "Tell me if you want stricter filters (budget, brand, usage, feature)."
    )

    prompt_tokens = 120 + min(len(user_message), 140)
    completion_tokens = 180 + len(filtered) * 30
    reasoning_trace = [
        "Parse user intent and extract hard constraints (budget/brand).",
        "Run semantic retrieval over catalog candidates.",
        "Apply deterministic filters to remove non-matching products.",
        "Rank by rating and value, then return top recommendations.",
    ]

    return {
        "assistant_reply": assistant_reply,
        "applied_filters": filters,
        "recommended_items": [asdict(item) for item in filtered],
        "reasoning_trace": reasoning_trace,
        "tool_calls": [
            {
                "name": "semantic_retrieval",
                "args": {"query": user_message, "top_k": 5},
                "result": {"candidate_count": retrieval_candidates},
                "status": "success",
            },
            {
                "name": "apply_filters",
                "args": {"filters": filters},
                "result": {"remaining_count": len(filtered)},
                "status": "success",
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_text(text: str) -> Iterable[str]:
    for token in text.split(" "):
        yield token + " "
