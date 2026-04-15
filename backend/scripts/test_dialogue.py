"""
End-to-end tests for the multi-agent dialogue layer.

Runs on top of the mock LLM by default (no API key required), so these
tests double as a smoke test for the routing graph.

Usage (from RAG_Component/ as CWD):
    LLM_PROVIDER=mock python -m scripts.test_dialogue
"""

from __future__ import annotations

import os
import sys

# Load `.env` first so LLM_PROVIDER from file wins over the hard-coded default.
try:
    import pathlib
    from dotenv import load_dotenv
    _env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

# Fall back to mock only if nothing else was configured.
os.environ.setdefault("LLM_PROVIDER", "mock")

from agent.dialogue.orchestrator import ShoppingAgent
from agent.dialogue.llm import USAGE


def div(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def show_turn(user: str, result):
    print(f"USER : {user}")
    print(f"BOT  [{result.route:>8s}]: {result.reply[:220]}")


# ─────────────────────────────────────────────────────────────
# Individual scenarios
# ─────────────────────────────────────────────────────────────

def test_greeting_then_category():
    div("Test 1: greeting + initial vague input → clarify")
    a = ShoppingAgent()
    r = a.chat("I want to buy something")
    show_turn("I want to buy something", r)
    assert r.route == "clarify", r


def test_category_then_budget_clarify():
    div("Test 2: category given but no budget → clarify budget")
    a = ShoppingAgent()
    r = a.chat("I'm looking for headphones")
    show_turn("I'm looking for headphones", r)
    # Either clarify (asking budget) or go straight to search.
    assert r.route in ("clarify", "search"), r


def test_full_search():
    div("Test 3: full constraints → search")
    a = ShoppingAgent()
    r = a.chat("Sony headphones under $200 for daily commute")
    show_turn("Sony headphones under $200 for daily commute", r)
    assert r.route == "search", r


def test_multi_turn_refine():
    div("Test 4: multi-turn progressive refinement")
    a = ShoppingAgent()
    r = a.chat("I need headphones")
    show_turn("I need headphones", r)
    r = a.chat("budget 150")
    show_turn("budget 150", r)
    r = a.chat("Sony only")
    show_turn("Sony only", r)
    assert a.state.brand and a.state.brand.lower() == "sony"
    assert a.state.max_price == 150


def test_compare_reference():
    div("Test 5: 'compare the first two' after a search")
    a = ShoppingAgent()
    a.chat("Sony headphones under $300")
    if not a.state.last_products:
        print("  (skip — no catalogue results in this run)")
        return
    r = a.chat("Compare the first two")
    show_turn("Compare the first two", r)
    assert r.route == "compare", r


def test_checkout_reference():
    div("Test 6: 'buy the first one' after a search")
    a = ShoppingAgent()
    a.chat("Anker USB-C charger under $50")
    if not a.state.last_products:
        print("  (skip — no catalogue results in this run)")
        return
    r = a.chat("I'll take the first one")
    show_turn("I'll take the first one", r)
    assert r.route == "checkout", r


def test_detail_reference():
    div("Test 7: 'details of the second one'")
    a = ShoppingAgent()
    a.chat("Bluetooth earbuds under $100")
    if len(a.state.last_products) < 2:
        print("  (skip — not enough catalogue results)")
        return
    r = a.chat("Tell me more about the second one")
    show_turn("Tell me more about the second one", r)
    assert r.route == "detail", r


def test_force_search_after_two_clarifies():
    div("Test 8: clarify_count cap forces search")
    a = ShoppingAgent()
    a.state.clarify_count = 2
    a.state.category = "All Electronics"
    r = a.chat("anything will do")
    show_turn("anything will do", r)
    assert r.route == "search", r


def test_reset():
    div("Test 9: reset clears slots")
    a = ShoppingAgent()
    a.chat("Sony headphones under $200")
    a.reset()
    assert a.state.brand is None
    assert a.state.max_price is None
    print("  state cleared OK")


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

def main():
    tests = [
        test_greeting_then_category,
        test_category_then_budget_clarify,
        test_full_search,
        test_multi_turn_refine,
        test_compare_reference,
        test_checkout_reference,
        test_detail_reference,
        test_force_search_after_two_clarifies,
        test_reset,
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
    div("LLM usage summary")
    print(f"  calls        : {USAGE.calls}")
    print(f"  input_tokens : {USAGE.input_tokens}")
    print(f"  output_tokens: {USAGE.output_tokens}")
    print(f"  est. cost USD: {USAGE.estimated_usd:.6f}")

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
