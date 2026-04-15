"""
Public entry point for the dialogue layer.
==========================================

Usage
-----
    from agent.dialogue.orchestrator import ShoppingAgent

    agent = ShoppingAgent()
    reply = agent.chat("I'm looking for noise cancelling headphones under $200")
    print(reply)

The `ShoppingAgent` class owns the persistent `DialogueState` so callers
(CLI, Streamlit, tests) only have to feed user messages and consume replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .state import DialogueState
from .graph import build_graph


_GREETING = "Hi! What are you shopping for today?"


@dataclass
class TurnResult:
    reply: str
    route: str                      # which specialist / tool handled the turn
    debug: dict = field(default_factory=dict)


class ShoppingAgent:
    """Stateful multi-agent shopping assistant."""

    def __init__(self):
        self._graph = build_graph()
        self.state = DialogueState()

    # ── Public API ──────────────────────────────────────────

    def greet(self) -> str:
        return _GREETING

    def chat(self, user_message: str) -> TurnResult:
        user_message = (user_message or "").strip()
        if not user_message:
            return TurnResult(reply=_GREETING, route="noop")

        self.state.turn_count += 1
        self.state.history.append(("user", user_message))

        gs = {
            "dialogue": self.state,
            "user_message": user_message,
            "route": "",
            "referenced_ids": [],
            "reply": "",
            "debug": {},
        }
        gs = self._graph.invoke(gs)

        reply = gs.get("reply") or "Sorry, I didn't catch that. Could you rephrase?"
        self.state.history.append(("assistant", reply))

        return TurnResult(
            reply=reply,
            route=gs.get("route", ""),
            debug=gs.get("debug", {}),
        )

    def reset(self) -> None:
        self.state = DialogueState()


# ─────────────────────────────────────────────────────────────
# Convenience functional API (stateless, for quick experiments)
# ─────────────────────────────────────────────────────────────

def run_turn(state: DialogueState, user_message: str) -> tuple[str, DialogueState, dict]:
    """Single-turn functional wrapper used by tests."""
    graph = build_graph()
    state.turn_count += 1
    state.history.append(("user", user_message))

    gs = {
        "dialogue": state,
        "user_message": user_message,
        "route": "",
        "referenced_ids": [],
        "reply": "",
        "debug": {},
    }
    gs = graph.invoke(gs)

    reply = gs.get("reply") or ""
    state.history.append(("assistant", reply))
    return reply, state, gs.get("debug", {})
