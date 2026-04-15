"""Base class shared by every specialist Agent in the dialogue graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..state import DialogueState


@dataclass
class AgentOutput:
    """Standard envelope every Agent returns to the orchestrator."""
    state_delta: dict = field(default_factory=dict)
    reply: Optional[str] = None         # user-visible text, if any
    next_action: Optional[str] = None   # optional hand-off hint
    payload: dict = field(default_factory=dict)  # diagnostics / debug data


class BaseAgent(ABC):
    """Every Agent exposes one method: `run(state, user_message)`."""

    name: str = "base"

    @abstractmethod
    def run(self, state: DialogueState, user_message: str) -> AgentOutput:
        ...
