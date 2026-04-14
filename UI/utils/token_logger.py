from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PRICING_SNAPSHOT = {
    "snapshot_version": "2026-04-14",
    "currency": "USD",
    "unit": "per_1k_tokens",
    "models": {
        "gpt-4o": {
            "provider": "openai",
            "prompt_per_1k": 0.005,
            "completion_per_1k": 0.015,
            "source_url": "https://openai.com/pricing",
        },
        "claude-sonnet-4-6": {
            "provider": "anthropic",
            "prompt_per_1k": 0.003,
            "completion_per_1k": 0.015,
            "source_url": "https://www.anthropic.com/pricing#api",
        },
        "gemini-2.5-flash": {
            "provider": "google",
            "prompt_per_1k": 0.00035,
            "completion_per_1k": 0.00105,
            "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
        },
        "qwen-plus": {
            "provider": "qwen",
            "prompt_per_1k": 0.0008,
            "completion_per_1k": 0.002,
            "source_url": "https://www.alibabacloud.com/help/en/model-studio/getting-started/models",
        },
    },
}
DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "pricing" / "pricing_snapshot.json"
FALLBACK_UNKNOWN_MODEL_RATE = {"prompt_per_1k": 0.001, "completion_per_1k": 0.002}


@dataclass
class TurnUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenLogger:
    """Session-level token usage tracker and cost estimator."""

    def __init__(
        self,
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        self.snapshot_path = snapshot_path or DEFAULT_SNAPSHOT_PATH
        self.snapshot = self._load_pricing_snapshot()
        self.pricing = pricing or self._extract_rate_table(self.snapshot)
        self.average_rate = self._compute_average_rate(self.pricing)
        self.turns: List[TurnUsage] = []

    def _load_pricing_snapshot(self) -> Dict[str, Any]:
        if self.snapshot_path.exists():
            with self.snapshot_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return DEFAULT_PRICING_SNAPSHOT

    def _extract_rate_table(self, snapshot: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        models = snapshot.get("models", {})
        table: Dict[str, Dict[str, float]] = {}
        for model_name, info in models.items():
            table[model_name] = {
                "prompt_per_1k": float(info["prompt_per_1k"]),
                "completion_per_1k": float(info["completion_per_1k"]),
            }
        return table

    def _compute_average_rate(self, pricing: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not pricing:
            return FALLBACK_UNKNOWN_MODEL_RATE
        prompt_sum = Decimal("0")
        completion_sum = Decimal("0")
        count = Decimal(str(len(pricing)))
        for rate in pricing.values():
            prompt_sum += Decimal(str(rate["prompt_per_1k"]))
            completion_sum += Decimal(str(rate["completion_per_1k"]))
        prompt_avg = (prompt_sum / count).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        completion_avg = (completion_sum / count).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        return {"prompt_per_1k": float(prompt_avg), "completion_per_1k": float(completion_avg)}

    def reset_session(self) -> None:
        self.turns = []

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        rate = self.pricing.get(model, self.average_rate)
        prompt_cost = (Decimal(prompt_tokens) / Decimal(1000)) * Decimal(
            str(rate["prompt_per_1k"])
        )
        completion_cost = (Decimal(completion_tokens) / Decimal(1000)) * Decimal(
            str(rate["completion_per_1k"])
        )
        total = (prompt_cost + completion_cost).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        return float(total)

    def log_turn(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnUsage:
        total_tokens = prompt_tokens + completion_tokens
        usage = TurnUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=self.estimate_cost(model, prompt_tokens, completion_tokens),
            metadata=metadata or {},
        )
        self.turns.append(usage)
        return usage

    def log_from_usage(
        self, usage: Dict[str, Any], model: str, metadata: Optional[Dict[str, Any]] = None
    ) -> TurnUsage:
        """Adapter for OpenAI/Anthropic style usage payloads."""
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or 0
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        return self.log_turn(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata=metadata,
        )

    def get_session_summary(self) -> Dict[str, Any]:
        total_prompt = sum(turn.prompt_tokens for turn in self.turns)
        total_completion = sum(turn.completion_tokens for turn in self.turns)
        total_tokens = sum(turn.total_tokens for turn in self.turns)
        total_cost = round(sum(turn.estimated_cost for turn in self.turns), 6)
        return {
            "turn_count": len(self.turns),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
        }

    def get_last_turn(self) -> Optional[TurnUsage]:
        return self.turns[-1] if self.turns else None

    def get_pricing_basis(self) -> Dict[str, Any]:
        return {
            "snapshot_version": self.snapshot.get("snapshot_version", "unknown"),
            "currency": self.snapshot.get("currency", "USD"),
            "unit": self.snapshot.get("unit", "per_1k_tokens"),
            "known_models": sorted(self.pricing.keys()),
            "unknown_model_strategy": "Use strict arithmetic mean across OpenAI/Anthropic/Google/Qwen snapshot rates.",
            "average_rate_per_1k": self.average_rate,
        }
