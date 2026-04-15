"""
Build a 3-level hierarchical index for the Consumer-Electronics RAG.

Level 1 — main_category  (e.g. "Cell Phones & Accessories", "Computers")
Level 2 — brand          (e.g. "Apple", "Samsung", "Sony")
Level 3 — individual product (semantic embedding for fine-grained retrieval)

Storage:
    ChromaDB  — one collection with rich metadata for hierarchical filtering
    SQLite    — full structured data for deterministic tool queries

Usage:
    python -m data.build_index          # first time (downloads from HF)
    python -m data.build_index --cache  # rebuild from local cache only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from data.load_dataset import get_products

DATA_DIR = pathlib.Path(__file__).resolve().parent
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_PATH = DATA_DIR / "products.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 256


def _product_to_text(p: dict) -> str:
    """Combine fields into a rich text chunk for embedding."""
    specs_str = ", ".join(
        f"{k}: {v}" for k, v in p.get("specifications", {}).items()
    )
    model_str = f" Model: {p['model']}." if p.get("model") else ""
    return (
        f"{p['title']}.{model_str} Brand: {p['brand']}. "
        f"Category: {p['main_category']} > {p.get('subcategory', '')}. "
        f"{p.get('description', '')[:800]} "
        f"Specs: {specs_str}."
    )


def build_chroma(products: list[dict]):
    """Embed products and store in ChromaDB with hierarchical metadata."""
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection("products")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="products",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start : start + BATCH_SIZE]
        ids = [p["id"] for p in batch]
        documents = [_product_to_text(p) for p in batch]
        metadatas = [
            {
                "title": p["title"][:200],
                "brand": p["brand"],
                "model": p.get("model", ""),
                "price": p["price"] if p["price"] is not None else -1.0,
                "main_category": p["main_category"],
                "subcategory": p.get("subcategory", ""),
                "rating": p["rating"],
                "rating_count": p["rating_count"],
                "has_price": 1 if p["price"] is not None else 0,
            }
            for p in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  ChromaDB batch {start//BATCH_SIZE + 1}: "
              f"upserted {len(batch)} (total {min(start+BATCH_SIZE, len(products))})")

    print(f"ChromaDB: {collection.count()} products indexed -> {CHROMA_DIR}")


def build_sqlite(products: list[dict]):
    """Create SQLite tables for structured queries."""
    conn = sqlite3.connect(str(SQLITE_PATH))
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute(
        """
        CREATE TABLE products (
            id              TEXT PRIMARY KEY,
            title           TEXT,
            brand           TEXT,
            model           TEXT,
            price           REAL,
            main_category   TEXT,
            subcategory     TEXT,
            description     TEXT,
            specifications  TEXT,
            rating          REAL,
            rating_count    INTEGER,
            image_url       TEXT
        )
        """
    )

    cur.execute("DROP TABLE IF EXISTS brand_stats")
    cur.execute(
        """
        CREATE TABLE brand_stats (
            brand           TEXT,
            main_category   TEXT,
            product_count   INTEGER,
            avg_price       REAL,
            avg_rating      REAL,
            PRIMARY KEY (brand, main_category)
        )
        """
    )

    cur.execute("DROP TABLE IF EXISTS category_tree")
    cur.execute(
        """
        CREATE TABLE category_tree (
            main_category   TEXT,
            subcategory     TEXT,
            brand           TEXT,
            product_count   INTEGER,
            PRIMARY KEY (main_category, subcategory, brand)
        )
        """
    )

    for p in products:
        cur.execute(
            """
            INSERT OR REPLACE INTO products
            (id, title, brand, model, price, main_category, subcategory,
             description, specifications, rating, rating_count, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["id"], p["title"], p["brand"], p.get("model", ""),
                p["price"], p["main_category"], p.get("subcategory", ""),
                p.get("description", ""),
                json.dumps(p.get("specifications", {})),
                p["rating"], p["rating_count"],
                p.get("image_url", ""),
            ),
        )

    cur.execute(
        """
        INSERT OR REPLACE INTO brand_stats (brand, main_category, product_count, avg_price, avg_rating)
        SELECT brand, main_category, COUNT(*), AVG(CASE WHEN price > 0 THEN price END), AVG(rating)
        FROM products
        GROUP BY brand, main_category
        """
    )

    cur.execute(
        """
        INSERT OR REPLACE INTO category_tree (main_category, subcategory, brand, product_count)
        SELECT main_category, subcategory, brand, COUNT(*)
        FROM products
        GROUP BY main_category, subcategory, brand
        """
    )

    conn.commit()
    conn.close()
    print(f"SQLite: {len(products)} products + aggregation tables -> {SQLITE_PATH}")


def main(use_cache: bool = True):
    products = get_products(use_cache=use_cache)
    build_chroma(products)
    build_sqlite(products)
    print("Index build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False,
                        help="Use local cache only (skip HF download)")
    args = parser.parse_args()
    main(use_cache=args.cache)
