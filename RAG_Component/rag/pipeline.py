"""
RAG Pipeline — the single entry-point consumed by the Agent orchestrator.
=========================================================================

Workflow:
    1. Receive **rewritten query** + structured constraints from dialogue agent
    2. Hierarchical semantic retrieval (category → brand → cosine)
    3. Deterministic tool-chain filtering (price / brand / rating / specs)
    4. Rank & limit
    5. Return PipelineResult (ready for LLM to generate recommendation text)

Also exposes convenience wrappers for browsing, comparison, checkout, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from rag.retriever import (
    retrieve_products,
    browse_categories,
    browse_brands,
    browse_subcategories,
    get_product_detail,
    get_products_by_ids,
)
from agent.tools import (
    apply_price_filter,
    filter_by_brand,
    filter_by_rating,
    filter_by_specs,
    rank_candidates,
    compare_products,
    simulate_purchase,
    find_similar_products,
    brand_summary,
    price_statistics,
    keyword_search,
    product_detail,
    spec_lookup,
)


@dataclass
class SearchConstraints:
    """Structured constraints extracted by the dialogue agent."""
    category: Optional[str] = None
    subcategory: Optional[str] = None
    brand: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    allowed_brands: Optional[list[str]] = None
    excluded_brands: Optional[list[str]] = None
    min_rating: Optional[float] = None
    min_rating_count: int = 0
    required_specs: Optional[dict[str, str]] = None
    sort_by: str = "relevance_score"
    top_k: int = 10
    limit: int = 5


@dataclass
class PipelineResult:
    """Structured output returned to the Agent / frontend."""
    query: str = ""
    constraints_applied: dict = field(default_factory=dict)
    retrieved_count: int = 0
    filtered_count: int = 0
    products: list[dict] = field(default_factory=list)
    comparison: Optional[dict] = None
    checkout: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────────────────────
# Core search pipeline
# ─────────────────────────────────────────────────────────────

def search(
    query: str,
    constraints: SearchConstraints | dict | None = None,
) -> PipelineResult:
    """
    Full RAG search pipeline: retrieve → filter → rank → return.
    """
    if constraints is None:
        constraints = SearchConstraints()
    elif isinstance(constraints, dict):
        constraints = SearchConstraints(**{
            k: v for k, v in constraints.items()
            if k in SearchConstraints.__dataclass_fields__
        })

    # Step 1 — hierarchical semantic retrieval
    candidates = retrieve_products(
        query=query,
        category=constraints.category,
        subcategory=constraints.subcategory,
        brand=constraints.brand,
        top_k=constraints.top_k,
        min_price=constraints.min_price,
        max_price=constraints.max_price,
    )
    retrieved_count = len(candidates)

    # Step 2 — deterministic filtering chain
    filtered = candidates

    if constraints.max_price is not None or constraints.min_price is not None:
        filtered = apply_price_filter(
            filtered,
            max_price=constraints.max_price,
            min_price=constraints.min_price,
        )

    if constraints.allowed_brands or constraints.excluded_brands:
        filtered = filter_by_brand(
            filtered,
            allowed_brands=constraints.allowed_brands,
            excluded_brands=constraints.excluded_brands,
        )

    if constraints.min_rating is not None or constraints.min_rating_count > 0:
        filtered = filter_by_rating(
            filtered,
            min_rating=constraints.min_rating or 0.0,
            min_rating_count=constraints.min_rating_count,
        )

    if constraints.required_specs:
        filtered = filter_by_specs(filtered, required_specs=constraints.required_specs)

    # Step 3 — rank and limit
    ranked = rank_candidates(
        filtered,
        sort_by=constraints.sort_by,
        limit=constraints.limit,
    )

    return PipelineResult(
        query=query,
        constraints_applied=_constraints_summary(constraints),
        retrieved_count=retrieved_count,
        filtered_count=len(filtered),
        products=ranked,
    )


# ─────────────────────────────────────────────────────────────
# Convenience wrappers for Agent tool calls
# ─────────────────────────────────────────────────────────────

def explore_categories() -> list[dict]:
    return browse_categories()


def explore_brands(category: Optional[str] = None, min_products: int = 2) -> list[dict]:
    return browse_brands(category, min_products)


def explore_subcategories(category: Optional[str] = None, brand: Optional[str] = None) -> list[dict]:
    return browse_subcategories(category, brand)


def compare(product_ids: list[str]) -> dict:
    return compare_products(product_ids)


def checkout(product_id: str) -> dict:
    return simulate_purchase(product_id)


def similar(product_id: str, top_k: int = 5) -> list[dict]:
    return find_similar_products(product_id, top_k=top_k)


def detail(product_id: str) -> dict:
    return product_detail(product_id)


def specs(product_id: str) -> dict:
    return spec_lookup(product_id)


def brand_info(brand: str, category: Optional[str] = None) -> dict:
    return brand_summary(brand, category)


def price_stats(category: Optional[str] = None, brand: Optional[str] = None) -> dict:
    return price_statistics(category, brand)


def text_search(keyword: str, category: Optional[str] = None, brand: Optional[str] = None, limit: int = 10) -> list[dict]:
    return keyword_search(keyword, category, brand, limit)


# ─────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────

def _constraints_summary(c: SearchConstraints) -> dict:
    s: dict = {}
    if c.category:
        s["category"] = c.category
    if c.subcategory:
        s["subcategory"] = c.subcategory
    if c.brand:
        s["brand"] = c.brand
    if c.max_price is not None:
        s["max_price"] = c.max_price
    if c.min_price is not None:
        s["min_price"] = c.min_price
    if c.allowed_brands:
        s["allowed_brands"] = c.allowed_brands
    if c.excluded_brands:
        s["excluded_brands"] = c.excluded_brands
    if c.min_rating is not None:
        s["min_rating"] = c.min_rating
    if c.min_rating_count > 0:
        s["min_rating_count"] = c.min_rating_count
    if c.required_specs:
        s["required_specs"] = c.required_specs
    s["sort_by"] = c.sort_by
    return s
