"""
LLM client wrapper.
===================

A thin abstraction over the OpenAI-compatible Chat Completions API so the
rest of the codebase never has to care about which provider is live.

Supports three providers via the `LLM_PROVIDER` environment variable:
    - "qwen"   → DashScope (OpenAI-compatible endpoint), cheapest option
    - "openai" → Official OpenAI API
    - "mock"   → Deterministic offline fake, used when no API key is present

Every call records token usage into a module-level ledger so we can surface
cumulative spend (required by the proposal's "Cost and Token Efficiency"
evaluation metric).
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

# Auto-load variables from a project-root `.env` if present, so callers can
# drop a credentials file and run without exporting env vars manually.
try:
    from dotenv import load_dotenv
    for _candidate in (
        pathlib.Path(__file__).resolve().parents[2] / ".env",  # backend/.env
        pathlib.Path(__file__).resolve().parents[3] / ".env",  # repo-root/.env
    ):
        if _candidate.exists():
            load_dotenv(_candidate)
            break
except ImportError:
    pass  # dotenv is optional; OS-level env vars still work.

try:
    from openai import OpenAI
except ImportError:  # openai is optional at import-time for the mock provider
    OpenAI = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Provider configuration
# ─────────────────────────────────────────────────────────────

_QWEN_BASE_URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_QWEN_BASE_URL_CN   = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Approximate pricing (USD per 1M tokens). Update when providers change rates.
# Used only for rough cost estimation in the telemetry ledger.
_PRICING: dict[str, tuple[float, float]] = {
    "qwen-turbo":       (0.05, 0.20),
    "qwen-plus":        (0.40, 1.20),
    "gpt-4o-mini":      (0.15, 0.60),
    "gpt-4o":           (2.50, 10.00),
}


@dataclass
class UsageRecord:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok

        in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
        cost = (in_tok * in_rate + out_tok * out_rate) / 1_000_000
        self.estimated_usd += cost

        entry = self.by_model.setdefault(
            model, {"calls": 0, "in": 0, "out": 0, "usd": 0.0}
        )
        entry["calls"] += 1
        entry["in"] += in_tok
        entry["out"] += out_tok
        entry["usd"] += cost


USAGE = UsageRecord()


# ─────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────

class LLMClient:
    """Unified JSON-first chat client."""

    def __init__(
        self,
        provider: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.default_model = default_model or self._default_model_for(self.provider)
        self._client = self._build_client()

    def _default_model_for(self, provider: str) -> str:
        if provider == "qwen":
            return "qwen-plus"
        if provider == "openai":
            return "gpt-4o-mini"
        return "mock"

    def _build_client(self):
        if self.provider == "mock":
            return None
        if OpenAI is None:
            raise RuntimeError(
                "openai package is required for non-mock providers. "
                "Run: pip install openai"
            )
        if self.provider == "qwen":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("DASHSCOPE_API_KEY is not set.")
            # Allow the caller to pick a region. Defaults to international.
            # DASHSCOPE_REGION=cn   → mainland endpoint (console.aliyun.com / bailian)
            # DASHSCOPE_REGION=intl → international endpoint (default)
            # Or override explicitly via DASHSCOPE_BASE_URL.
            region = (os.getenv("DASHSCOPE_REGION") or "cn").lower()
            base_url = os.getenv("DASHSCOPE_BASE_URL") or (
                _QWEN_BASE_URL_INTL if region == "intl" else _QWEN_BASE_URL_CN
            )
            return OpenAI(api_key=api_key, base_url=base_url)
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            return OpenAI(api_key=api_key)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ── Public API ──────────────────────────────────────────

    def call_json(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> dict:
        """
        Invoke the LLM and parse its reply as JSON.

        Falls back to mock output when running offline. On any JSON parse
        failure we return an empty dict so callers can degrade gracefully
        instead of crashing the dialogue.
        """
        model = model or self.default_model

        if self.provider == "mock":
            from .mock_llm import mock_json_response
            return mock_json_response(system, user)

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        usage = response.usage
        if usage:
            USAGE.add(model, usage.prompt_tokens, usage.completion_tokens)

        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def call_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """Plain-text variant used by the responder for narrative output."""
        model = model or self.default_model

        if self.provider == "mock":
            from .mock_llm import mock_text_response
            return mock_text_response(system, user)

        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

        usage = response.usage
        if usage:
            USAGE.add(model, usage.prompt_tokens, usage.completion_tokens)

        return (response.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────
# Module-level singleton for convenience
# ─────────────────────────────────────────────────────────────

_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_usage() -> None:
    global USAGE
    USAGE = UsageRecord()
