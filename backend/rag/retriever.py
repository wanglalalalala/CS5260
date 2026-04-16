"""
3-Level Hierarchical RAG Retriever
===================================

Level 1 — Category drilldown   (main_category filter)
Level 2 — Brand drilldown      (brand filter)
Level 3 — Semantic search      (cosine similarity on product text embeddings)

The Agent can call any level independently or combine them:
    - browse_categories()          → discover Level-1
    - browse_brands(category)      → discover Level-2
    - retrieve_products(query, ...) → Level-3 semantic + optional L1/L2 filters

All results carry hierarchical metadata so the Agent knows where in the
taxonomy each product sits.
"""

from __future__ import annotations

import json
import os
import sqlite3
import pathlib
import threading
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_PATH = DATA_DIR / "products.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_client: chromadb.ClientAPI | None = None
_collection = None
_index_ready = False
_index_lock = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    value = (os.getenv(name) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _collection_exists(client: chromadb.ClientAPI, name: str) -> bool:
    try:
        client.get_collection(name=name)
        return True
    except Exception:
        return False


def _index_artifacts_ready() -> bool:
    if not SQLITE_PATH.exists():
        return False
    if not CHROMA_DIR.exists():
        return False
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _collection_exists(client, "products")
    except Exception:
        return False


def _ensure_index_ready() -> None:
    """
    Optional startup hook for cloud deployment:
    build the index automatically when artifacts are missing.

    Controlled by environment variables:
      AUTO_BUILD_INDEX_ON_START=1   enable auto-build
      AUTO_BUILD_INDEX_USE_CACHE=1  use local cache only (no HF download)
    """
    global _index_ready

    if _index_ready:
        return
    if _index_artifacts_ready():
        _index_ready = True
        return
    if not _env_flag("AUTO_BUILD_INDEX_ON_START", "0"):
        return

    with _index_lock:
        if _index_ready:
            return
        if _index_artifacts_ready():
            _index_ready = True
            return

        use_cache = _env_flag("AUTO_BUILD_INDEX_USE_CACHE", "0")
        print(
            "RAG index artifacts missing. Auto-building now "
            f"(use_cache={use_cache}) ..."
        )
        from data.build_index import main as build_index_main

        build_index_main(use_cache=use_cache)
        if not _index_artifacts_ready():
            raise RuntimeError(
                "Auto-build finished but RAG index is still unavailable. "
                "Check backend/data/chroma_db and backend/data/products.db."
            )
        _index_ready = True


def _get_collection():
    global _client, _collection
    _ensure_index_ready()
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_collection(
            name="products", embedding_function=ef
        )
    return _collection


def _get_db():
    _ensure_index_ready()
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────
# Level 1 — Category browsing
# ─────────────────────────────────────────────────────────────

def browse_categories() -> list[dict]:
    """
    Return all main categories with product counts.

    Returns
    -------
    [{"category": "Cell Phones & Accessories", "product_count": 3471,
      "brands": 245, "avg_rating": 4.1}, ...]
    """
    conn = _get_db()
    rows = conn.execute(
        """
        SELECT main_category AS category,
               COUNT(*) AS product_count,
               COUNT(DISTINCT brand) AS brands,
               ROUND(AVG(rating), 2) AS avg_rating
        FROM products
        GROUP BY main_category
        ORDER BY product_count DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# Level 2 — Brand browsing (within a category)
# ─────────────────────────────────────────────────────────────

def browse_brands(
    category: Optional[str] = None,
    min_products: int = 2,
) -> list[dict]:
    """
    Return brands with aggregated stats, optionally filtered by category.

    Returns
    -------
    [{"brand": "Apple", "product_count": 87, "avg_price": 299.5,
      "avg_rating": 4.3, "main_category": "..."}, ...]
    """
    conn = _get_db()
    if category:
        rows = conn.execute(
            """
            SELECT brand, main_category, product_count, avg_price,
                   ROUND(avg_rating, 2) AS avg_rating
            FROM brand_stats
            WHERE main_category = ? AND product_count >= ?
            ORDER BY product_count DESC
            """,
            (category, min_products),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT brand,
                   GROUP_CONCAT(DISTINCT main_category) AS main_category,
                   SUM(product_count) AS product_count,
                   ROUND(AVG(avg_price), 2) AS avg_price,
                   ROUND(AVG(avg_rating), 2) AS avg_rating
            FROM brand_stats
            GROUP BY brand
            HAVING SUM(product_count) >= ?
            ORDER BY product_count DESC
            """,
            (min_products,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def browse_subcategories(
    category: Optional[str] = None,
    brand: Optional[str] = None,
) -> list[dict]:
    """
    Explore the category tree at any granularity.
    """
    conn = _get_db()
    query = "SELECT main_category, subcategory, SUM(product_count) AS count FROM category_tree WHERE 1=1"
    params: list = []
    if category:
        query += " AND main_category = ?"
        params.append(category)
    if brand:
        query += " AND brand = ?"
        params.append(brand)
    query += " GROUP BY main_category, subcategory ORDER BY count DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# Level 3 — Semantic retrieval (with optional L1/L2 filters)
# ─────────────────────────────────────────────────────────────

def retrieve_products(
    query: str,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    brand: Optional[str] = None,
    top_k: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> list[dict]:
    """
    Hierarchical semantic retrieval.

    Filters are pushed down into ChromaDB's metadata WHERE clause so that
    the ANN search only considers products in the target segment.

    Parameters
    ----------
    query        : rewritten query from dialogue agent
    category     : Level-1 filter (main_category)
    subcategory  : finer category filter
    brand        : Level-2 filter
    top_k        : max results
    min_price    : price floor (products with unknown price are kept)
    max_price    : price ceiling

    Returns
    -------
    list[dict] sorted by relevance (highest first).
    """
    collection = _get_collection()
    where = _build_where(category, subcategory, brand, min_price, max_price)

    kwargs: dict = {"query_texts": [query], "n_results": top_k}
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception:
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    products: list[dict] = []
    if not results["ids"] or not results["ids"][0]:
        return products

    for idx, pid in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][idx]
        distance = results["distances"][0][idx] if results.get("distances") else 0.0
        relevance = round(1.0 - distance, 4)

        price = meta.get("price", -1.0)
        products.append({
            "id": pid,
            "title": meta.get("title", ""),
            "brand": meta.get("brand", ""),
            "model": meta.get("model", ""),
            "price": price if price > 0 else None,
            "main_category": meta.get("main_category", ""),
            "subcategory": meta.get("subcategory", ""),
            "rating": meta.get("rating", 0.0),
            "rating_count": meta.get("rating_count", 0),
            "relevance_score": relevance,
        })

    return products


def get_product_detail(product_id: str) -> dict | None:
    """Fetch full product details from SQLite."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if isinstance(d.get("specifications"), str):
        try:
            d["specifications"] = json.loads(d["specifications"])
        except json.JSONDecodeError:
            d["specifications"] = {}
    return d


def get_products_by_ids(product_ids: list[str]) -> list[dict]:
    """Fetch full product details for a list of IDs."""
    if not product_ids:
        return []
    conn = _get_db()
    ph = ",".join("?" for _ in product_ids)
    rows = conn.execute(
        f"SELECT * FROM products WHERE id IN ({ph})", product_ids
    ).fetchall()
    conn.close()
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


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _build_where(
    category: Optional[str],
    subcategory: Optional[str],
    brand: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
) -> dict | None:
    conditions = []
    if category:
        conditions.append({"main_category": {"$eq": category}})
    if subcategory:
        conditions.append({"subcategory": {"$eq": subcategory}})
    if brand:
        conditions.append({"brand": {"$eq": brand}})
    if min_price is not None:
        conditions.append({"price": {"$gte": min_price}})
    if max_price is not None:
        conditions.append({"price": {"$lte": max_price}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
