"""
Supervisor Agent.
=================

Routes the current turn to the right downstream node. Uses cheap rules first
(keyword/reference detection) and only falls back to the LLM when the rules
are inconclusive. This keeps average per-turn token spend low.
"""

from __future__ import annotations

from .base import BaseAgent, AgentOutput
from ..state import DialogueState, resolve_references
from ..llm import get_client
from ..prompts import SUPERVISOR_SYSTEM, supervisor_user


_DIRECT_ACTION_KEYWORDS = {
    "compare":  ("compare", " vs ", "versus", "difference between"),
    "checkout": ("buy", "purchase", "order", "check out", "checkout", "i'll take"),
    "detail":   ("details", "spec", "specs", "specifications", "tell me more"),
}


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self):
        self.llm = get_client()

    # ── Rule fast-path ──────────────────────────────────────

    def _rule_route(self, state: DialogueState, msg: str) -> tuple[str | None, list[str]]:
        lower = msg.lower()

        # Direct-action keywords only fire if we have something to act on.
        for action, keywords in _DIRECT_ACTION_KEYWORDS.items():
            if any(k in lower for k in keywords) and state.last_products:
                ids = resolve_references(state, msg)
                if not ids and action in ("checkout", "detail"):
                    # Default to the first product if the user says "buy it"
                    # without naming which one.
                    ids = [state.last_products[0]["id"]]
                if not ids and action == "compare" and len(state.last_products) >= 2:
                    ids = [state.last_products[0]["id"], state.last_products[1]["id"]]
                return action, ids

        # Force-search if we've been clarifying too long.
        if state.clarify_count >= 2 and state.category:
            return "search", []

        return None, []

    # ── Main ────────────────────────────────────────────────

    def run(self, state: DialogueState, user_message: str) -> AgentOutput:
        action, ids = self._rule_route(state, user_message)

        if action is None:
            decision = self.llm.call_json(
                system=SUPERVISOR_SYSTEM,
                user=supervisor_user(state.snapshot(), user_message, state.history),
            )
            action = decision.get("action", "clarify")

        if action not in {"clarify", "search", "compare", "checkout", "detail"}:
            action = "clarify"

        return AgentOutput(
            next_action=action,
            payload={"referenced_ids": ids},
        )
