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

# Low-information replies — the user agreed with something but didn't tell
# us what to change. Route to clarify so we ask them to be specific.
_LOW_INFO_REPLIES = {
    "yes", "yeah", "yep", "ok", "okay", "sure", "please", "go ahead",
    "no", "nope", "not really",
}


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self):
        self.llm = get_client()

    # ── Rule fast-path ──────────────────────────────────────

    def _rule_route(self, state: DialogueState, msg: str) -> tuple[str | None, list[str]]:
        lower = msg.lower().strip(" .?!")

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

        # Low-information reply after a search ("yes", "ok", ...) — the user
        # agreed in principle but gave us no new constraint. Ask them to be
        # specific instead of re-running the exact same search.
        if state.has_searched and lower in _LOW_INFO_REPLIES:
            return "clarify", []

        # Force-search if we've been clarifying too long — but only once we
        # have at least the category to anchor retrieval.
        if state.clarify_count >= 3 and state.category:
            return "search", []

        # Fast-path to search when we have a category plus at least one
        # HARD constraint — or the user explicitly opted out of a budget.
        # Only on the first search; once we have results, require NEW info
        # before re-searching, otherwise we'd loop on the same query.
        if not state.has_searched and state.category and (
            state.brand or state.max_price is not None or state.budget_skipped
        ):
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
