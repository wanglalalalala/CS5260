"""
Deterministic Tool functions for the AI Shopper Agent.
======================================================

13 tools covering filtering, comparison, analytics, and actions.
Each tool operates on structured data (SQLite / retriever output)
and never hallucinates — the LLM decides *when* to call them, but
the results are always grounded in real product data.
"""

from __future__ import annotations

import json
import sqlite3
import pathlib
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
SQLITE_PATH = DATA_DIR / "products.db"


def _get_db():
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows) -> list[dict]:
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("specifications"), str):
            try:
                d["specifications"] = json.loads(d["specifications"])
            except json.JSONDecodeError:
                d["specifications"] = {}
        result.append(d)
    return result


# ════════════════════════════════════════════════════════════
# 1. PRICE FILTER
# ════════════════════════════════════════════════════════════

def apply_price_filter(
    products: list[dict],
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
) -> list[dict]:
    """Filter products by price range. Products with unknown price are kept."""
    result = products
    if min_price is not None:
        result = [p for p in result
                  if p.get("price") is None or p["price"] >= min_price]
    if max_price is not None:
        result = [p for p in result
                  if p.get("price") is None or p["price"] <= max_price]
    return result


# ════════════════════════════════════════════════════════════
# 2. BRAND FILTER
# ════════════════════════════════════════════════════════════

def filter_by_brand(
    products: list[dict],
    allowed_brands: list[str] | None = None,
    excluded_brands: list[str] | None = None,
) -> list[dict]:
    """Include / exclude products by brand (case-insensitive)."""
    result = products
    if allowed_brands:
        lower = {b.lower() for b in allowed_brands}
        result = [p for p in result if p.get("brand", "").lower() in lower]
    if excluded_brands:
        lower_ex = {b.lower() for b in excluded_brands}
        result = [p for p in result if p.get("brand", "").lower() not in lower_ex]
    return result


# ════════════════════════════════════════════════════════════
# 3. RATING FILTER
# ════════════════════════════════════════════════════════════

def filter_by_rating(
    products: list[dict],
    min_rating: float = 0.0,
    min_rating_count: int = 0,
) -> list[dict]:
    """Keep products with rating >= threshold and enough reviews."""
    return [
        p for p in products
        if p.get("rating", 0) >= min_rating
        and p.get("rating_count", 0) >= min_rating_count
    ]


# ════════════════════════════════════════════════════════════
# 4. SPEC FILTER (keyword-based)
# ════════════════════════════════════════════════════════════

def filter_by_specs(
    products: list[dict],
    required_specs: dict[str, str] | None = None,
) -> list[dict]:
    """
    Keep products whose specs match ALL required key→value pairs
    (case-insensitive substring match).
    Falls back to SQLite for full spec data.
    """
    if not required_specs:
        return products

    conn = _get_db()
    ph = ",".join("?" for _ in products)
    ids = [p["id"] for p in products]
    rows = conn.execute(
        f"SELECT id, specifications FROM products WHERE id IN ({ph})", ids
    ).fetchall()
    conn.close()

    spec_map: dict[str, dict] = {}
    for r in rows:
        specs = r["specifications"]
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except json.JSONDecodeError:
                specs = {}
        spec_map[r["id"]] = specs

    result = []
    for p in products:
        specs = spec_map.get(p["id"], {})
        if all(
            required_val.lower() in str(specs.get(k, "")).lower()
            for k, required_val in required_specs.items()
        ):
            result.append(p)
    return result


# ════════════════════════════════════════════════════════════
# 5. SPEC LOOKUP — get full specs for one product
# ════════════════════════════════════════════════════════════

def spec_lookup(product_id: str) -> dict:
    """Return detailed specifications for a single product."""
    conn = _get_db()
    row = conn.execute(
        "SELECT id, title, brand, model, specifications FROM products WHERE id = ?",
        (product_id,),
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"Product {product_id} not found"}
    d = dict(row)
    if isinstance(d.get("specifications"), str):
        try:
            d["specifications"] = json.loads(d["specifications"])
        except json.JSONDecodeError:
            d["specifications"] = {}
    return d


# ════════════════════════════════════════════════════════════
# 6. COMPARE PRODUCTS (side-by-side)
# ════════════════════════════════════════════════════════════

def compare_products(product_ids: list[str]) -> dict:
    """
    Generate a structured side-by-side comparison table.

    Returns {"columns": [id1, id2, ...], "rows": [{field, id1_val, id2_val}, ...]}
    """
    conn = _get_db()
    ph = ",".join("?" for _ in product_ids)
    rows = conn.execute(
        f"SELECT * FROM products WHERE id IN ({ph})", product_ids
    ).fetchall()
    conn.close()

    full = _rows_to_dicts(rows)
    if not full:
        return {"columns": [], "rows": []}

    all_spec_keys: list[str] = []
    for fp in full:
        for k in fp.get("specifications", {}):
            if k not in all_spec_keys:
                all_spec_keys.append(k)

    basic_fields = ["title", "brand", "model", "price", "main_category",
                    "subcategory", "rating", "rating_count"]
    table_rows: list[dict] = []

    for field in basic_fields:
        row = {"field": field}
        for fp in full:
            row[fp["id"]] = fp.get(field, "")
        table_rows.append(row)

    for spec_key in all_spec_keys:
        row = {"field": f"spec:{spec_key}"}
        for fp in full:
            row[fp["id"]] = fp.get("specifications", {}).get(spec_key, "N/A")
        table_rows.append(row)

    return {"columns": [fp["id"] for fp in full], "rows": table_rows}


# ════════════════════════════════════════════════════════════
# 7. RANK CANDIDATES
# ════════════════════════════════════════════════════════════

def rank_candidates(
    products: list[dict],
    sort_by: str = "relevance_score",
    ascending: bool = False,
    limit: int = 5,
) -> list[dict]:
    """Sort candidates by a field and return top-N."""
    valid = {"relevance_score", "price", "rating", "rating_count"}
    if sort_by not in valid:
        sort_by = "relevance_score"

    def key_fn(p):
        v = p.get(sort_by)
        if v is None:
            return float("-inf") if not ascending else float("inf")
        return v

    return sorted(products, key=key_fn, reverse=(not ascending))[:limit]


# ════════════════════════════════════════════════════════════
# 8. SIMILAR PRODUCTS — find products similar to a given one
# ════════════════════════════════════════════════════════════

def find_similar_products(
    product_id: str,
    top_k: int = 5,
    same_category: bool = True,
) -> list[dict]:
    """
    Use the product's own text as a query to find similar items.
    Excludes the source product from results.
    """
    from rag.retriever import retrieve_products, get_product_detail

    detail = get_product_detail(product_id)
    if not detail:
        return []

    query = f"{detail['title']} {detail.get('brand', '')} {detail.get('description', '')[:300]}"
    category = detail["main_category"] if same_category else None

    results = retrieve_products(query, category=category, top_k=top_k + 1)
    return [r for r in results if r["id"] != product_id][:top_k]


# ════════════════════════════════════════════════════════════
# 9. BRAND SUMMARY — aggregated brand analytics
# ════════════════════════════════════════════════════════════

def brand_summary(
    brand: str,
    category: Optional[str] = None,
) -> dict:
    """
    Return aggregated stats for a brand: product count, price range,
    avg rating, top-rated products, subcategories covered.
    """
    conn = _get_db()
    where = "WHERE LOWER(brand) = LOWER(?)"
    params: list = [brand]
    if category:
        where += " AND main_category = ?"
        params.append(category)

    stats = conn.execute(
        f"""
        SELECT COUNT(*) AS product_count,
               MIN(CASE WHEN price > 0 THEN price END) AS min_price,
               MAX(price) AS max_price,
               ROUND(AVG(CASE WHEN price > 0 THEN price END), 2) AS avg_price,
               ROUND(AVG(rating), 2) AS avg_rating,
               GROUP_CONCAT(DISTINCT subcategory) AS subcategories
        FROM products {where}
        """,
        params,
    ).fetchone()

    top_rated = conn.execute(
        f"""
        SELECT id, title, rating, price FROM products {where}
        ORDER BY rating DESC, rating_count DESC LIMIT 5
        """,
        params,
    ).fetchall()

    conn.close()

    return {
        "brand": brand,
        "stats": dict(stats) if stats else {},
        "top_rated": [dict(r) for r in top_rated],
    }


# ════════════════════════════════════════════════════════════
# 10. PRICE STATISTICS — price distribution in a segment
# ════════════════════════════════════════════════════════════

def price_statistics(
    category: Optional[str] = None,
    brand: Optional[str] = None,
) -> dict:
    """
    Return price distribution stats (min, max, avg, median, percentiles)
    for a given segment.
    """
    conn = _get_db()
    where_parts = ["price > 0"]
    params: list = []
    if category:
        where_parts.append("main_category = ?")
        params.append(category)
    if brand:
        where_parts.append("LOWER(brand) = LOWER(?)")
        params.append(brand)

    where = " AND ".join(where_parts)

    row = conn.execute(
        f"""
        SELECT COUNT(*) AS count,
               MIN(price) AS min_price,
               MAX(price) AS max_price,
               ROUND(AVG(price), 2) AS avg_price
        FROM products WHERE {where}
        """,
        params,
    ).fetchone()

    prices = conn.execute(
        f"SELECT price FROM products WHERE {where} ORDER BY price",
        params,
    ).fetchall()
    conn.close()

    price_list = [p["price"] for p in prices]
    result = dict(row) if row else {}
    if price_list:
        n = len(price_list)
        result["median_price"] = round(price_list[n // 2], 2)
        result["p25"] = round(price_list[n // 4], 2)
        result["p75"] = round(price_list[3 * n // 4], 2)
    return result


# ════════════════════════════════════════════════════════════
# 11. KEYWORD SEARCH — full-text search on title + description
# ════════════════════════════════════════════════════════════

def keyword_search(
    keyword: str,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Simple SQL LIKE search on title and description.
    Useful as a fallback when semantic search returns too few results.
    """
    conn = _get_db()
    where_parts = ["(LOWER(title) LIKE ? OR LOWER(description) LIKE ?)"]
    pattern = f"%{keyword.lower()}%"
    params: list = [pattern, pattern]

    if category:
        where_parts.append("main_category = ?")
        params.append(category)
    if brand:
        where_parts.append("LOWER(brand) = LOWER(?)")
        params.append(brand)

    where = " AND ".join(where_parts)

    rows = conn.execute(
        f"""
        SELECT id, title, brand, model, price, main_category, subcategory,
               rating, rating_count
        FROM products WHERE {where}
        ORDER BY rating_count DESC
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
# 12. PRODUCT DETAIL — full info for a single product
# ════════════════════════════════════════════════════════════

def product_detail(product_id: str) -> dict:
    """Return the complete product record."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"Product {product_id} not found"}
    d = dict(row)
    if isinstance(d.get("specifications"), str):
        try:
            d["specifications"] = json.loads(d["specifications"])
        except json.JSONDecodeError:
            d["specifications"] = {}
    return d


# ════════════════════════════════════════════════════════════
# 13. MOCK CHECKOUT
# ════════════════════════════════════════════════════════════

def simulate_purchase(product_id: str) -> dict:
    """Simulate a purchase — returns a mock order confirmation."""
    conn = _get_db()
    row = conn.execute(
        "SELECT id, title, price FROM products WHERE id = ?", (product_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"success": False, "message": f"Product {product_id} not found."}

    product = dict(row)
    price_str = f"${product['price']:.2f}" if product["price"] else "Price TBD"
    return {
        "success": True,
        "order_id": f"ORD-{product_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "product_id": product["id"],
        "product_title": product["title"],
        "price": product["price"],
        "message": f"Order placed for '{product['title']}' at {price_str}.",
    }


# ════════════════════════════════════════════════════════════
# TOOL REGISTRY — for Agent integration
# ════════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, callable] = {
    "apply_price_filter": apply_price_filter,
    "filter_by_brand": filter_by_brand,
    "filter_by_rating": filter_by_rating,
    "filter_by_specs": filter_by_specs,
    "spec_lookup": spec_lookup,
    "compare_products": compare_products,
    "rank_candidates": rank_candidates,
    "find_similar_products": find_similar_products,
    "brand_summary": brand_summary,
    "price_statistics": price_statistics,
    "keyword_search": keyword_search,
    "product_detail": product_detail,
    "simulate_purchase": simulate_purchase,
}
