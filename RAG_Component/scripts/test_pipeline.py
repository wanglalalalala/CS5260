"""
End-to-end test for the 3-level hierarchical RAG + 13 Tools pipeline.

Usage (from project root):
    1. python -m data.load_dataset      # download & cache
    2. python -m data.build_index       # build ChromaDB + SQLite
    3. python -m scripts.test_pipeline  # run all tests
"""

import json
from rag.pipeline import (
    search, explore_categories, explore_brands, explore_subcategories,
    compare, checkout, similar, detail, specs, brand_info, price_stats,
    text_search,
)


def div(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def show(products: list[dict], max_show: int = 5):
    if not products:
        print("  (no products)")
        return
    for i, p in enumerate(products[:max_show], 1):
        price = f"${p['price']:.2f}" if p.get("price") else "N/A"
        print(
            f"  {i}. [{p.get('id','')}] {p.get('title','')[:70]}\n"
            f"     Brand: {p.get('brand','')}  |  Price: {price}  |  "
            f"Rating: {p.get('rating','')}  |  Rel: {p.get('relevance_score','')}"
        )


# ── Level 1: Category browsing ─────────────────────────────

def test_browse_categories():
    div("Test 1: Browse categories (Level 1)")
    cats = explore_categories()
    for c in cats:
        print(f"  {c['category']}: {c['product_count']} products, "
              f"{c['brands']} brands, avg rating {c['avg_rating']}")
    assert len(cats) > 0


# ── Level 2: Brand browsing ────────────────────────────────

def test_browse_brands():
    div("Test 2: Browse brands within Cell Phones (Level 2)")
    brands = explore_brands("Cell Phones & Accessories", min_products=3)
    for b in brands[:10]:
        print(f"  {b['brand']}: {b['product_count']} products, "
              f"avg ${b.get('avg_price','N/A')}, rating {b['avg_rating']}")
    assert len(brands) > 0


def test_browse_subcategories():
    div("Test 3: Browse subcategories")
    subs = explore_subcategories(category="All Electronics")
    for s in subs[:10]:
        print(f"  {s['main_category']} > {s['subcategory']}: {s['count']}")


# ── Level 3: Semantic search ──────────────────────────────

def test_basic_search():
    div("Test 4: Basic semantic search")
    result = search("wireless bluetooth headphones noise cancelling")
    print(f"  Retrieved: {result.retrieved_count}  |  Filtered: {result.filtered_count}")
    show(result.products)
    assert result.retrieved_count > 0


def test_category_filtered_search():
    div("Test 5: Search within Cell Phones category")
    result = search(
        "smartphone case with screen protector",
        constraints={"category": "Cell Phones & Accessories"},
    )
    print(f"  Retrieved: {result.retrieved_count}  |  Filtered: {result.filtered_count}")
    show(result.products)


def test_brand_filtered_search():
    div("Test 6: Search within a specific brand")
    result = search(
        "laptop charger USB-C fast charging",
        constraints={"brand": "Anker"},
    )
    print(f"  Retrieved: {result.retrieved_count}  |  Filtered: {result.filtered_count}")
    show(result.products)


def test_price_filtered_search():
    div("Test 7: Search with price constraint (under $30)")
    result = search(
        "phone accessories",
        constraints={
            "category": "Cell Phones & Accessories",
            "max_price": 30.0,
        },
    )
    print(f"  Retrieved: {result.retrieved_count}  |  Filtered: {result.filtered_count}")
    show(result.products)


def test_combined_constraints():
    div("Test 8: Combined constraints (Computers, $50-$200, rating >= 4.0)")
    result = search(
        "keyboard mechanical RGB",
        constraints={
            "category": "Computers",
            "min_price": 50.0,
            "max_price": 200.0,
            "min_rating": 4.0,
            "sort_by": "rating",
        },
    )
    print(f"  Retrieved: {result.retrieved_count}  |  Filtered: {result.filtered_count}")
    print(f"  Constraints: {json.dumps(result.constraints_applied, indent=2)}")
    show(result.products)


# ── Tool tests ────────────────────────────────────────────

def test_compare():
    div("Test 9: Compare products")
    result = search("wireless earbuds", constraints={"top_k": 3, "limit": 3})
    if len(result.products) >= 2:
        ids = [p["id"] for p in result.products[:2]]
        comp = compare(ids)
        print(f"  Comparing: {comp['columns']}")
        for row in comp["rows"][:8]:
            f = row["field"]
            vals = " | ".join(f"{c}: {row.get(c,'')}" for c in comp["columns"])
            print(f"    {f:>25s}  →  {vals}")
    else:
        print("  (not enough products to compare)")


def test_checkout():
    div("Test 10: Mock checkout")
    result = search("USB cable", constraints={"limit": 1})
    if result.products:
        receipt = checkout(result.products[0]["id"])
        print(f"  {json.dumps(receipt, indent=2)}")


def test_similar():
    div("Test 11: Find similar products")
    result = search("tablet", constraints={"limit": 1})
    if result.products:
        pid = result.products[0]["id"]
        print(f"  Similar to: {result.products[0]['title'][:60]}")
        sims = similar(pid, top_k=3)
        show(sims)


def test_brand_info():
    div("Test 12: Brand summary")
    info = brand_info("Apple", "Cell Phones & Accessories")
    print(f"  Stats: {json.dumps(info.get('stats', {}), indent=2)}")
    print(f"  Top rated:")
    for p in info.get("top_rated", [])[:3]:
        print(f"    - {p.get('title','')[:60]} (rating: {p.get('rating','')})")


def test_price_stats():
    div("Test 13: Price statistics")
    stats = price_stats("All Electronics")
    print(f"  {json.dumps(stats, indent=2)}")


def test_keyword_search():
    div("Test 14: Keyword search (fallback)")
    results = text_search("charger", category="Cell Phones & Accessories", limit=5)
    show(results)


def test_spec_lookup():
    div("Test 15: Spec lookup")
    result = search("camera", constraints={"category": "Camera & Photo", "limit": 1})
    if result.products:
        pid = result.products[0]["id"]
        sp = specs(pid)
        print(f"  Product: {sp.get('title','')[:60]}")
        print(f"  Specs: {json.dumps(sp.get('specifications', {}), indent=2)}")


def main():
    tests = [
        test_browse_categories,
        test_browse_brands,
        test_browse_subcategories,
        test_basic_search,
        test_category_filtered_search,
        test_brand_filtered_search,
        test_price_filtered_search,
        test_combined_constraints,
        test_compare,
        test_checkout,
        test_similar,
        test_brand_info,
        test_price_stats,
        test_keyword_search,
        test_spec_lookup,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\n  *** FAILED: {e}")
            import traceback
            traceback.print_exc()

    div(f"Results: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
