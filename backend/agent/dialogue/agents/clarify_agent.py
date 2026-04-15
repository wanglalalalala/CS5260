"""
Clarify Agent.
==============

Asks ONE focused question to unlock the most impactful missing slot.
Also absorbs any slot value the user may have just supplied, so we never
throw away information gained during the clarification loop.
"""

from __future__ import annotations

from .base import BaseAgent, AgentOutput
from ..state import DialogueState
from ..llm import get_client
from ..prompts import CLARIFY_SYSTEM, clarify_user


_DEFAULT_QUESTION = "Could you tell me a bit more about what you're shopping for?"


class ClarifyAgent(BaseAgent):
    name = "clarify"

    def __init__(self):
        self.llm = get_client()

    def run(self, state: DialogueState, user_message: str) -> AgentOutput:
        decision = self.llm.call_json(
            system=CLARIFY_SYSTEM,
            user=clarify_user(state.snapshot(), user_message, state.history),
        )

        question = decision.get("question") or _DEFAULT_QUESTION
        delta = decision.get("state_delta") or {}

        return AgentOutput(
            state_delta=delta,
            reply=question,
            next_action="clarify",
        )
