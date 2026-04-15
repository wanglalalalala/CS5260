from __future__ import annotations

import html
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Make backend importable and trigger its .env autoload before we read any keys.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services import real_agent  # noqa: E402
from utils.token_logger import TokenLogger  # noqa: E402


# Shortcut cards shown on the first turn. `query` is the text we feed the
# agent — kept as a plain product-type keyword so ClarifyAgent can infer
# the canonical `category` slot itself.
CATEGORY_CARDS = [
    {"label": "Headphones",   "emoji": "🎧", "query": "headphones"},
    {"label": "Laptops",      "emoji": "💻", "query": "laptop"},
    {"label": "Smartphones",  "emoji": "📱", "query": "smartphone"},
    {"label": "Tablets",      "emoji": "📲", "query": "tablet"},
    {"label": "Smartwatches", "emoji": "⌚", "query": "smartwatch"},
    {"label": "Cameras",      "emoji": "📷", "query": "camera"},
    {"label": "Speakers",     "emoji": "🔊", "query": "speaker"},
]


st.set_page_config(
    page_title="AI Personal Shopper",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.block-container { padding-top: 2.6rem; padding-bottom: 7rem; max-width: 1200px; }
[data-testid="stMain"] { background: #b2b4b9; }
[data-testid="stMainBlockContainer"] {
    background: transparent;
    border-radius: 0;
    padding-left: 1rem;
    padding-right: 1rem;
}
[data-testid="stSidebar"] {
    background: #e7eaf4;
    border-right: 1px solid rgba(99, 102, 241, 0.22);
}
[data-testid="stSidebarContent"] { background: #e7eaf4; }
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.2rem;
    max-height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
}
.page-title-wrap {
    width: 100%;
    text-align: center;
    margin: 0 0 1rem;
    padding-top: 0.25rem;
}
.page-title-text {
    display: inline-block;
    color: var(--text-color);
    font-size: clamp(1.8rem, 3.2vw, 2.65rem);
    line-height: 1.35;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin: 0;
    padding: 0.25rem 0;
    overflow: visible;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.panel {
    background: color-mix(in srgb, var(--secondary-background-color) 90%, #eef2ff);
    border: 1px solid rgba(99, 102, 241, 0.20);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
}
.sidebar-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-color);
    margin: 0 0 0.55rem;
    letter-spacing: 0.01em;
}
.sidebar-module {
    border-radius: 14px;
    padding: 0.9rem 0.9rem 0.8rem;
    margin-bottom: 0.75rem;
}
.module-filters { background: rgba(59, 130, 246, 0.10); }
.module-recommend { background: rgba(99, 102, 241, 0.10); }
.module-token { background: rgba(148, 163, 184, 0.18); }
.sidebar-muted {
    color: var(--text-color);
    opacity: 0.72;
    font-size: 0.95rem;
}
.token-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.label {
    font-size: 0.76rem;
    color: var(--text-color);
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.value { font-size: 1.08rem; font-weight: 700; color: var(--text-color); }
.item-card {
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.6rem;
    background: color-mix(in srgb, var(--secondary-background-color) 90%, #eef2ff);
}
.item-name { font-size: 0.95rem; font-weight: 700; color: var(--text-color); margin: 0 0 0.2rem; }
.item-meta { font-size: 0.82rem; color: var(--text-color); opacity: 0.88; }
.item-reason { font-size: 0.82rem; color: var(--text-color); opacity: 0.8; margin-top: 0.35rem; }
.filter-tag {
    display: inline-block;
    margin: 0.2rem 0.25rem 0.2rem 0;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    background: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    color: var(--text-color);
    font-size: 0.72rem;
    font-weight: 600;
}
div[data-testid="stChatMessage"] {
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 14px;
    padding: 0.4rem 0.55rem;
    background: color-mix(in srgb, var(--secondary-background-color) 90%, #eef2ff);
    margin-bottom: 0.5rem;
}
div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0.9rem;
    width: 100%;
    max-width: 100%;
    z-index: 1000;
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
    padding: 0.15rem 0.35rem;
}
@media (max-width: 900px) {
    .block-container { padding-top: 2rem; }
    .page-title-text { font-size: clamp(1.45rem, 6vw, 1.95rem); }
    div[data-testid="stChatInput"] { bottom: 0.4rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Provider resolution
# ─────────────────────────────────────────────────────────────

def _hydrate_env_from_secrets() -> None:
    """Copy Streamlit secrets into env so backend llm.py can read them.

    Accessing `st.secrets` raises StreamlitSecretNotFoundError when no
    secrets file is present (normal for local dev — we fall back to .env).
    """
    for key in ("LLM_PROVIDER", "OPENAI_API_KEY", "DASHSCOPE_API_KEY",
                "DASHSCOPE_REGION", "DASHSCOPE_BASE_URL", "HF_TOKEN"):
        if os.environ.get(key):
            continue
        try:
            value = st.secrets[key]
        except Exception:
            continue
        if value:
            os.environ[key] = str(value)


def _resolve_provider_status() -> tuple[bool, str, str]:
    """
    Returns (ok, provider, message).

    backend/llm.py already loads backend/.env on import, so by the time we
    call this, os.environ reflects the user's configuration.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").lower().strip()
    if not provider:
        return False, "", (
            "LLM_PROVIDER is not set. Add it to `backend/.env` or Streamlit "
            "secrets — one of `openai` or `qwen`."
        )
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, provider, (
                "OPENAI_API_KEY is missing. Set it in `backend/.env` or "
                "Streamlit secrets."
            )
        return True, provider, "openai"
    if provider == "qwen":
        if not os.environ.get("DASHSCOPE_API_KEY"):
            return False, provider, (
                "DASHSCOPE_API_KEY is missing. Set it in `backend/.env` or "
                "Streamlit secrets."
            )
        return True, provider, "qwen"
    return False, provider, (
        f"Unsupported LLM_PROVIDER=`{provider}`. Use `openai` or `qwen`."
    )


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I am your AI shopping assistant. I can help you "
                    "find **consumer electronics** — laptops, headphones, "
                    "smartphones, tablets, smartwatches, cameras, and similar "
                    "gadgets.\n\n"
                    "Tell me what you're looking for, ideally with budget, "
                    "brand, or use case. For example: "
                    "_\"Noise-cancelling headphones under $200 for work calls\"_."
                ),
            }
        ]
    if "applied_filters" not in st.session_state:
        st.session_state.applied_filters = []
    if "recommended_items" not in st.session_state:
        st.session_state.recommended_items = []
    if "token_logger" not in st.session_state:
        st.session_state.token_logger = TokenLogger()


def _clear_session() -> None:
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.applied_filters = []
    st.session_state.recommended_items = []
    st.session_state.token_logger.reset_session()
    real_agent.reset_agent()
    st.rerun()


# ─────────────────────────────────────────────────────────────
# Sidebar (filters / recommendations / token usage)
# ─────────────────────────────────────────────────────────────

def render_sidebar_panels(provider_label: str) -> None:
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-module panel">'
            f'<div class="sidebar-title">Backend</div>'
            f'<div class="sidebar-muted">Provider: <b>{html.escape(provider_label)}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Session", use_container_width=True):
            _clear_session()

        if st.session_state.applied_filters:
            tags = "".join(
                [f'<span class="filter-tag">{html.escape(value)}</span>'
                 for value in st.session_state.applied_filters]
            )
        else:
            tags = '<div class="sidebar-muted">No active filters yet.</div>'
        st.markdown(
            '<div class="sidebar-module module-filters">'
            '<div class="sidebar-title">Applied Filters</div>'
            f"{tags}"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.recommended_items:
            cards = []
            for item in st.session_state.recommended_items[:3]:
                cards.append(
                    '<div class="item-card">'
                    f'<div class="item-name">{html.escape(item["name"])}</div>'
                    f'<div class="item-meta">{html.escape(item["brand"])} | ${item["price"]:.0f} | ⭐ {item["rating"]:.1f}</div>'
                    f'<div class="item-reason">{html.escape(item["short_reason"])}</div>'
                    "</div>"
                )
            rec_body = "".join(cards)
        else:
            rec_body = '<div class="sidebar-muted">Products will appear after your first query.</div>'
        st.markdown(
            '<div class="sidebar-module module-recommend">'
            '<div class="sidebar-title">Recommended Products</div>'
            f"{rec_body}"
            "</div>",
            unsafe_allow_html=True,
        )

        summary = st.session_state.token_logger.get_session_summary()
        last = st.session_state.token_logger.get_last_turn()
        last_text = ""
        if last:
            last_text = (
                '<div class="sidebar-muted" style="margin-top: 0.2rem;">'
                f"Last turn: {last.prompt_tokens} prompt + {last.completion_tokens} completion ({html.escape(last.model)})"
                "</div>"
            )
        st.markdown(
            '<div class="sidebar-module module-token">'
            '<div class="sidebar-title">Token Usage</div>'
            '<div class="token-grid">'
            f'<div class="panel"><div class="label">Turns</div><div class="value">{summary["turn_count"]}</div></div>'
            f'<div class="panel"><div class="label">Total Tokens</div><div class="value">{summary["total_tokens"]}</div></div>'
            "</div>"
            f'<div class="panel"><div class="label">Estimated Cost</div><div class="value">${summary["estimated_cost"]:.6f}</div></div>'
            f"{last_text}"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# Chat rendering
# ─────────────────────────────────────────────────────────────

def render_reasoning_panel(reasoning: List[str], tools: List[Dict[str, object]]) -> None:
    if not reasoning and not tools:
        return
    with st.expander("Show reasoning", expanded=False):
        if reasoning:
            st.markdown("**Agent Thinking Flow**")
            for i, step in enumerate(reasoning, start=1):
                st.markdown(f"{i}. {step}")
        if tools:
            st.markdown("**Tool Calls**")
            for tool in tools:
                args = json.dumps(tool.get("args", {}), ensure_ascii=False)
                result = json.dumps(tool.get("result", {}), ensure_ascii=False)
                status = tool.get("status", "unknown")
                st.markdown(
                    f"- `{tool.get('name', 'unknown')}` | status: **{status}**\n"
                    f"  - args: `{args}`\n"
                    f"  - result: `{result}`"
                )


def _escape_dollars_for_label(text: str) -> str:
    """Escape unescaped `$` in a short label so Streamlit's Markdown
    renderer doesn't interpret it as a LaTeX math delimiter."""
    out = []
    for i, ch in enumerate(text):
        if ch == "$" and (i == 0 or text[i - 1] != "\\"):
            out.append("\\$")
        else:
            out.append(ch)
    return "".join(out)


def render_chat() -> Optional[str]:
    """Render history. Return a suggestion-chip query if the user clicked one."""
    messages = st.session_state.messages
    last_assistant_idx = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        default=-1,
    )
    clicked: Optional[str] = None
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning_trace", [])
                tools = msg.get("tool_calls", [])
                render_reasoning_panel(reasoning, tools)
            st.markdown(msg["content"])
            if (
                msg.get("role") == "assistant"
                and idx == last_assistant_idx
                and msg.get("suggestions")
            ):
                suggestions = msg["suggestions"]
                cols = st.columns(len(suggestions))
                for col, text in zip(cols, suggestions):
                    # Escape unescaped `$` so Streamlit's Markdown renderer
                    # doesn't treat "$500 ... $1,000" as a LaTeX formula.
                    label = _escape_dollars_for_label(text)
                    if col.button(
                        label,
                        key=f"sug_{idx}_{text}",
                        use_container_width=True,
                    ):
                        clicked = text
    return clicked


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    _hydrate_env_from_secrets()
    # Importing backend triggers its .env loader.
    import agent.dialogue.llm  # noqa: F401

    init_state()

    st.markdown(
        '<div class="page-title-wrap"><span class="page-title-text">Intelligent Shopping Assistant</span></div>',
        unsafe_allow_html=True,
    )

    ok, provider, message = _resolve_provider_status()
    if not ok:
        st.error(
            f"**Backend not configured.** {message}\n\n"
            "Example `backend/.env`:\n\n"
            "```\nLLM_PROVIDER=qwen\nDASHSCOPE_API_KEY=your_key_here\n```"
        )
        render_sidebar_panels(provider_label=provider or "not configured")
        render_chat()
        st.chat_input("Configure backend to start chatting.", disabled=True)
        return

    # Figure out the display model name for the token logger.
    display_model = (
        os.environ.get("LLM_MODEL")
        or ("gpt-4o-mini" if provider == "openai" else "qwen-plus")
    )

    render_sidebar_panels(provider_label=f"{provider} · {display_model}")
    suggestion_pick = render_chat()

    # First turn only: show a horizontal row of category shortcut buttons.
    # `len(messages) == 1` means only the welcome message is present.
    category_pick: str | None = None
    if len(st.session_state.messages) == 1:
        st.markdown("<div style='margin: 0.4rem 0 0.2rem; opacity: 0.8;'>"
                    "Or pick a category to get started:</div>",
                    unsafe_allow_html=True)
        cols = st.columns(len(CATEGORY_CARDS))
        for col, card in zip(cols, CATEGORY_CARDS):
            if col.button(
                f"{card['emoji']}\n\n{card['label']}",
                key=f"cat_{card['label']}",
                use_container_width=True,
            ):
                category_pick = card["query"]

    typed_input = st.chat_input("Describe the electronics you want (e.g. laptop, headphones)...")
    user_input = typed_input or category_pick or suggestion_pick
    if not user_input:
        return

    _run_turn(user_input, display_model)


def _run_turn(user_input: str, display_model: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = real_agent.generate_reply(
                    user_message=user_input,
                    history=st.session_state.messages,
                )
            except Exception as exc:
                st.error(f"Backend error: {exc}")
                return
        reasoning = response.get("reasoning_trace", [])
        tools = response.get("tool_calls", [])
        render_reasoning_panel(reasoning, tools)
        placeholder = st.empty()
        text = str(response.get("assistant_reply", "No response"))
        assembled = ""
        for chunk in real_agent.stream_text(text):
            assembled += chunk
            placeholder.markdown(assembled + "▌")
            time.sleep(0.02)
        placeholder.markdown(assembled)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assembled,
            "reasoning_trace": response.get("reasoning_trace", []),
            "tool_calls": response.get("tool_calls", []),
            "suggestions": response.get("suggestions", []),
        }
    )
    st.session_state.applied_filters = response.get("applied_filters", [])
    st.session_state.recommended_items = response.get("recommended_items", [])
    usage = response.get("usage", {})
    st.session_state.token_logger.log_from_usage(
        usage=usage,
        model=display_model,
        metadata={"session_id": st.session_state.session_id},
    )
    st.rerun()


if __name__ == "__main__":
    main()
