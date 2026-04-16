"""
LLM client wrapper.
===================

A thin abstraction over multiple LLM providers so the rest of the codebase
never has to care which backend is selected.

Supports providers via `LLM_PROVIDER`:
    - "qwen"   → DashScope (OpenAI-compatible endpoint)
    - "openai" → Official OpenAI API
    - "gemini" → Google Gemini via OpenAI-compatible endpoint
    - "claude" → Anthropic Messages API
    - "mock"   → Deterministic offline fake

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
from urllib import error, request

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
_GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
_CLAUDE_BASE_URL = "https://api.anthropic.com"

# Approximate pricing (USD per 1M tokens). Update when providers change rates.
# Used only for rough cost estimation in the telemetry ledger.
_PRICING: dict[str, tuple[float, float]] = {
    "qwen-turbo":       (0.05, 0.20),
    "qwen-plus":        (0.40, 1.20),
    "gpt-4o-mini":      (0.15, 0.60),
    "gpt-4o":           (2.50, 10.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
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
        self.default_model = (
            default_model
            or os.getenv("LLM_MODEL")
            or self._default_model_for(self.provider)
        )
        self._client = self._build_client()

    def _default_model_for(self, provider: str) -> str:
        if provider == "qwen":
            return "qwen-plus"
        if provider == "openai":
            return "gpt-4o-mini"
        if provider == "gemini":
            return "gemini-2.0-flash"
        if provider == "claude":
            return "claude-3-5-sonnet-latest"
        return "mock"

    def _build_client(self):
        if self.provider in {"mock", "claude"}:
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
        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set.")
            base_url = os.getenv("GEMINI_BASE_URL") or _GEMINI_OPENAI_COMPAT_BASE_URL
            return OpenAI(api_key=api_key, base_url=base_url)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai_compatible(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        if usage:
            USAGE.add(model, usage.prompt_tokens, usage.completion_tokens)

        return (response.choices[0].message.content or "").strip()

    def _call_claude(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY is not set.")

        version = os.getenv("CLAUDE_API_VERSION", "2023-06-01")
        base = (os.getenv("CLAUDE_BASE_URL") or _CLAUDE_BASE_URL).rstrip("/")
        url = f"{base}/v1/messages"

        system_prompt = system
        if json_mode:
            system_prompt += (
                "\n\nReturn strictly one JSON object. Do not include markdown fences "
                "or extra text."
            )

        payload = {
            "model": model,
            "max_tokens": int(os.getenv("CLAUDE_MAX_TOKENS", "1024")),
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user}],
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": version,
            },
        )
        try:
            with request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Claude API error ({exc.code}): {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Claude API request failed: {exc.reason}") from exc

        body = json.loads(raw)
        usage = body.get("usage") or {}
        USAGE.add(
            model,
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )

        chunks = []
        for item in body.get("content") or []:
            if item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "".join(chunks).strip()

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
        if self.provider == "claude":
            raw = self._call_claude(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                json_mode=True,
            )
        else:
            raw = self._call_openai_compatible(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                json_mode=True,
            )
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
        if self.provider == "claude":
            return self._call_claude(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                json_mode=False,
            )
        return self._call_openai_compatible(
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            json_mode=False,
        )


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


def reset_client() -> None:
    global _default_client
    _default_client = None
