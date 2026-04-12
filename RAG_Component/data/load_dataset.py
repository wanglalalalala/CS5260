"""
Load and clean the Amazon Products 2023 dataset from HuggingFace.

Source: iguzelofficial/AMAZON-Products-2023
Focus categories for our Consumer-Electronics RAG:
    - Cell Phones & Accessories
    - All Electronics
    - Computers
    - Camera & Photo
    - Home Audio & Theater

Each product is normalised into a flat dict with:
    id, title, brand, price, main_category, subcategory, model,
    description, features, specifications (dict), rating, rating_count,
    image_url
"""

from __future__ import annotations

import ast
import json
import re
import pathlib
from typing import Optional

CACHE_PATH = pathlib.Path(__file__).resolve().parent / "products_clean.json"

TARGET_CATEGORIES = {
    "Cell Phones & Accessories",
    "All Electronics",
    "Computers",
    "Camera & Photo",
    "Home Audio & Theater",
}


def _parse_details(raw: Optional[str]) -> dict:
    """Parse the stringified dict in the 'details' column."""
    if not raw:
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def _extract_brand(row: dict, details: dict) -> str:
    """Try multiple sources to find the brand."""
    if row.get("store"):
        store = row["store"]
        if "Visit the" in store:
            return store.replace("Visit the", "").replace("Store", "").strip()
        if "Brand:" in store:
            return store.split("Brand:")[-1].strip()
        cleaned = store.split("Format:")[0].strip()
        if cleaned and len(cleaned) < 60:
            return cleaned

    for key in ("Brand", "brand", "Manufacturer", "manufacturer"):
        if details.get(key):
            return str(details[key]).strip()

    return "Unknown"


def _extract_model(title: str, details: dict) -> str:
    """Try to extract a model name/number."""
    for key in ("Item model number", "Model Number", "Model Name", "model"):
        if details.get(key):
            return str(details[key]).strip()
    return ""


def _extract_subcategory(row: dict) -> str:
    """Pick the most specific category from the categories list."""
    cats = row.get("categories") or []
    if isinstance(cats, str):
        try:
            cats = ast.literal_eval(cats)
        except Exception:
            cats = []
    if cats and len(cats) > 0:
        return cats[-1] if isinstance(cats[-1], str) else str(cats[-1])
    return ""


def _build_specs(details: dict) -> dict:
    """Extract hardware-like spec fields from the details dict."""
    spec_keys = {
        "Battery", "battery", "Battery Capacity", "Screen Size",
        "Display Size", "RAM", "ram", "Storage", "storage",
        "Processor", "processor", "CPU", "Weight", "weight",
        "Connectivity", "connectivity", "Bluetooth", "bluetooth",
        "OS", "Operating System", "Color", "color", "Colour",
        "Resolution", "resolution", "Megapixels",
        "Item Weight", "Product Dimensions", "Wireless Type",
        "Included Components", "Hardware Platform",
        "Memory Storage Capacity", "Hard Disk Size",
        "Batteries", "Number of Items",
    }
    specs = {}
    for k, v in details.items():
        if k in spec_keys and v:
            specs[k] = str(v).strip()
    return specs


def load_from_huggingface(
    max_per_category: int = 2000,
    require_price: bool = False,
) -> list[dict]:
    """
    Download, filter, and clean the Amazon dataset.

    Parameters
    ----------
    max_per_category : int
        Cap per main_category to keep the dataset manageable.
    require_price : bool
        If True, skip products with no price.
    """
    from datasets import load_dataset as hf_load

    ds = hf_load("iguzelofficial/AMAZON-Products-2023", split="train")

    count_by_cat: dict[str, int] = {}
    products: list[dict] = []

    for row in ds:
        cat = row.get("main_category")
        if cat not in TARGET_CATEGORIES:
            continue
        count_by_cat.setdefault(cat, 0)
        if count_by_cat[cat] >= max_per_category:
            continue

        title = (row.get("title") or "").strip()
        if not title or len(title) < 5:
            continue

        price = row.get("price")
        if price is None and require_price:
            continue
        if price is not None:
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = None

        details = _parse_details(row.get("details"))
        brand = _extract_brand(row, details)
        model = _extract_model(title, details)
        subcategory = _extract_subcategory(row)
        specs = _build_specs(details)

        description_parts = []
        desc = row.get("description")
        if desc:
            description_parts.append(
                desc if isinstance(desc, str)
                else " ".join(desc) if isinstance(desc, list)
                else str(desc)
            )
        feats = row.get("features") or []
        if feats:
            description_parts.append(
                " | ".join(feats) if isinstance(feats, list) else str(feats)
            )

        product = {
            "id": row.get("parent_asin", f"PROD_{len(products):05d}"),
            "title": title,
            "brand": brand,
            "model": model,
            "price": price,
            "main_category": cat,
            "subcategory": subcategory,
            "description": " ".join(description_parts)[:2000],
            "specifications": specs,
            "rating": float(row.get("average_rating") or 0),
            "rating_count": int(row.get("rating_number") or 0),
            "image_url": row.get("image") or "",
        }
        products.append(product)
        count_by_cat[cat] += 1

    print(f"Loaded {len(products)} products from HuggingFace")
    for c, n in sorted(count_by_cat.items()):
        print(f"  {c}: {n}")
    return products


def save_cache(products: list[dict]):
    """Save cleaned products to a local JSON cache."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2, default=str)
    print(f"Cached {len(products)} products -> {CACHE_PATH}")


def load_cache() -> list[dict]:
    """Load from local JSON cache if available."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_PATH}. "
            "Run `python -m data.load_dataset` first."
        )
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_products(use_cache: bool = True, **kwargs) -> list[dict]:
    """
    Main entry point: return cleaned products, using cache if available.
    """
    if use_cache and CACHE_PATH.exists():
        return load_cache()
    products = load_from_huggingface(**kwargs)
    save_cache(products)
    return products


if __name__ == "__main__":
    products = load_from_huggingface()
    save_cache(products)

    brands = {}
    for p in products:
        brands[p["brand"]] = brands.get(p["brand"], 0) + 1
    print(f"\nTop 20 brands:")
    for b, c in sorted(brands.items(), key=lambda x: -x[1])[:20]:
        print(f"  {b}: {c}")
