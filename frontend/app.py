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
:root {
    --font-sans: "Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --ink-navy: #132a3e;
    --ink-navy-soft: #223d56;
    --accent-amber: #d68a2e;
    --accent-amber-soft: #f7d9a6;
    --paper: #f8f3e9;
    --paper-strong: #fffdfa;
    --mist-blue: #e7eff8;
    --line-warm: rgba(30, 50, 70, 0.15);
    --shadow-soft: 0 8px 26px rgba(14, 31, 45, 0.08);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
[data-testid="stButton"] button, [data-testid="stChatInput"] input, label, p, span {
    font-family: var(--font-sans) !important;
}

.block-container { padding-top: 2.2rem; padding-bottom: 7rem; max-width: 1220px; }
[data-testid="stMain"] { background: var(--paper); }
[data-testid="stMainBlockContainer"] {
    background: transparent;
    padding-left: 1rem;
    padding-right: 1rem;
}
[data-testid="stSidebar"] {
    background: #f1e9da;
    border-right: 1px solid rgba(19, 42, 62, 0.14);
}
[data-testid="stSidebarContent"] { background: #f1e9da; }
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.2rem;
    max-height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
}

.page-title-wrap {
    width: 100%;
    text-align: center;
    margin: 0 0 1.2rem;
}
.page-title-text {
    display: inline-block;
    font-family: "Playfair Display", "Source Serif 4", "Times New Roman", serif !important;
    color: var(--paper-strong);
    font-size: clamp(1.95rem, 3.3vw, 2.9rem);
    line-height: 1.3;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 0;
    padding: 0.5rem 1.1rem;
    border-radius: 12px;
    background: linear-gradient(100deg, #152f47 0%, #1f4260 60%, #2a4e6f 100%);
    border: 1px solid rgba(214, 138, 46, 0.28);
    box-shadow: 0 12px 30px rgba(15, 35, 50, 0.22);
}

.panel {
    background: rgba(255, 253, 248, 0.78);
    border: 1px solid rgba(19, 42, 62, 0.13);
    border-radius: 12px;
    padding: 0.72rem 0.88rem;
}
.sidebar-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ink-navy-soft);
    margin: 0 0 0.5rem;
    letter-spacing: 0.02em;
}
.sidebar-module {
    border-radius: 16px;
    padding: 0.95rem;
    margin-bottom: 1.1rem;
}
.module-filters { background: rgba(214, 138, 46, 0.09); }
.module-recommend { background: rgba(19, 42, 62, 0.06); }
.module-token { background: rgba(19, 42, 62, 0.09); }
.sidebar-muted {
    color: var(--ink-navy-soft);
    opacity: 0.82;
    font-size: 0.87rem;
}
.label {
    font-size: 0.7rem;
    color: var(--ink-navy-soft);
    opacity: 0.78;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.value {
    font-size: 1.02rem;
    font-weight: 700;
    color: var(--ink-navy);
}
.token-meter {
    margin-top: 0.42rem;
}
.meter-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.24rem;
    font-size: 0.76rem;
    color: var(--ink-navy-soft);
}
.meter-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: rgba(19, 42, 62, 0.12);
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #d68a2e 0%, #ecb76d 100%);
}

.item-card {
    border: 1px solid rgba(19, 42, 62, 0.13);
    border-radius: 12px;
    padding: 0.78rem;
    margin-bottom: 0.66rem;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: var(--shadow-soft);
}
.item-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--ink-navy);
    margin: 0 0 0.2rem;
}
.item-meta {
    font-size: 0.8rem;
    color: var(--ink-navy-soft);
    opacity: 0.92;
}
.item-reason {
    font-size: 0.79rem;
    color: var(--ink-navy-soft);
    opacity: 0.9;
    margin-top: 0.34rem;
}
.filter-tag {
    display: inline-block;
    margin: 0.22rem 0.28rem 0.22rem 0;
    padding: 0.24rem 0.58rem;
    border-radius: 999px;
    background: rgba(255, 251, 243, 0.92);
    border: 1px solid rgba(214, 138, 46, 0.42);
    color: var(--ink-navy-soft);
    font-size: 0.72rem;
    font-weight: 600;
}

div[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 0.72rem;
    padding: 0.52rem 0.68rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border: 1px solid rgba(19, 42, 62, 0.16);
    border-left: 5px solid rgba(214, 138, 46, 0.72);
    background: linear-gradient(180deg, rgba(255, 250, 242, 0.98) 0%, rgba(255, 247, 234, 0.94) 100%);
    box-shadow: 0 10px 24px rgba(15, 35, 50, 0.1);
    margin-right: 6%;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: rgba(255, 255, 252, 0.58);
    border: 1px solid rgba(19, 42, 62, 0.08);
    border-radius: 12px;
    padding: 0.6rem 0.75rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    line-height: 1.72;
    font-size: 0.98rem;
    color: #1a3146;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p:first-of-type {
    margin-top: 0.05rem;
    margin-bottom: 0.9rem;
    font-weight: 700;
    font-size: 1.02rem;
    color: #15324a;
    letter-spacing: 0.01em;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] ol {
    margin: 0.22rem 0 0.7rem 1.2rem;
    padding: 0;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li {
    margin-bottom: 0.5rem;
    padding-left: 0.1rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li > p {
    margin: 0.14rem 0;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] a {
    color: #1f4f78;
    font-weight: 700;
    text-underline-offset: 2px;
    text-decoration-thickness: 1.5px;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] a:hover {
    color: #0f3f67;
    text-decoration-color: rgba(31, 79, 120, 0.7);
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] hr {
    border: none;
    border-top: 1px solid rgba(19, 42, 62, 0.14);
    margin: 0.95rem 0 0.95rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p:last-of-type {
    margin-top: 1rem;
    padding-top: 0.72rem;
    border-top: 1px dashed rgba(214, 138, 46, 0.45);
    color: #284764;
    font-weight: 600;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border: 1px solid rgba(33, 70, 107, 0.22);
    border-right: 5px solid rgba(31, 79, 120, 0.5);
    background: linear-gradient(180deg, rgba(232, 243, 255, 0.98) 0%, rgba(222, 237, 252, 0.92) 100%);
    box-shadow: 0 10px 22px rgba(20, 44, 66, 0.1);
    margin-left: 14%;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) hr {
    border: none;
    border-top: 1px solid rgba(214, 138, 46, 0.22);
    margin: 0.72rem 0;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) blockquote {
    border-left: 3px solid rgba(214, 138, 46, 0.72);
    background: rgba(255, 252, 247, 0.8);
    padding: 0.4rem 0.65rem;
    border-radius: 0 10px 10px 0;
}

details[data-testid="stExpander"] {
    border: 1px solid rgba(19, 42, 62, 0.12);
    border-radius: 12px;
    background: rgba(255, 253, 248, 0.68);
    margin-bottom: 0.7rem;
}
details[data-testid="stExpander"] summary {
    color: #415f78;
    font-weight: 600;
    font-size: 0.83rem;
}
details[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
    font-size: 0.86rem;
    color: #2f4e67;
}

div[data-testid="stChatMessage"] div[data-testid="stButton"] > button {
    border-radius: 999px;
    border: 1px solid rgba(214, 138, 46, 0.45);
    background: linear-gradient(180deg, #fff9ef 0%, #ffefd2 100%);
    color: var(--ink-navy-soft);
    min-height: 2.25rem;
    padding: 0.42rem 0.95rem;
    box-shadow: 0 2px 0 rgba(19, 42, 62, 0.04), 0 5px 14px rgba(214, 138, 46, 0.08);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease, background 0.16s ease;
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] > button p {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    line-height: 1.1;
    margin: 0;
    font-variant-numeric: tabular-nums;
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    background: linear-gradient(180deg, #fffaf2 0%, #ffe8c0 100%);
    box-shadow: 0 8px 20px rgba(20, 44, 66, 0.14), 0 0 0 1px rgba(214, 138, 46, 0.2) inset;
    border-color: rgba(214, 138, 46, 0.72);
}
div[data-testid="stChatMessage"] div[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid rgba(214, 138, 46, 0.55);
    outline-offset: 2px;
}

div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0.85rem;
    width: 100%;
    max-width: 100%;
    z-index: 1000;
    background: rgba(255, 253, 248, 0.96);
    border: 1px solid rgba(19, 42, 62, 0.18);
    border-radius: 14px;
    box-shadow: 0 10px 26px rgba(18, 40, 59, 0.12);
    padding: 0.15rem 0.35rem;
}
div.composer-hint {
    margin-top: 0.38rem;
    margin-bottom: 0.22rem;
    padding: 0.46rem 0.72rem;
    border-radius: 10px;
    border: 1px dashed rgba(214, 138, 46, 0.45);
    background: rgba(255, 249, 237, 0.75);
    color: #36526d;
    font-size: 0.82rem;
}
div.composer-hint b {
    color: #1f3f5d;
    font-weight: 700;
}

@media (max-width: 900px) {
    .block-container { padding-top: 1.9rem; }
    .page-title-text { font-size: clamp(1.45rem, 6vw, 2.1rem); }
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
        prompt_total = int(summary.get("prompt_tokens", 0))
        completion_total = int(summary.get("completion_tokens", 0))
        total_tokens = max(1, prompt_total + completion_total)
        prompt_pct = int((prompt_total / total_tokens) * 100)
        completion_pct = 100 - prompt_pct
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
            f'<div class="panel"><div class="label">Turns</div><div class="value">{summary["turn_count"]}</div></div>'
            f'<div class="panel"><div class="label">Total Tokens</div><div class="value">{summary["total_tokens"]}</div>'
            '<div class="token-meter">'
            f'<div class="meter-row"><span>Prompt</span><span>{prompt_total}</span></div>'
            f'<div class="meter-track"><div class="meter-fill" style="width:{prompt_pct}%;"></div></div>'
            f'<div class="meter-row" style="margin-top:0.35rem;"><span>Completion</span><span>{completion_total}</span></div>'
            f'<div class="meter-track"><div class="meter-fill" style="width:{completion_pct}%;"></div></div>'
            '</div></div>'
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

    # Show this guidance after the user has already started the flow
    # (e.g., picked a category or sent the first query).
    if len(st.session_state.messages) > 1:
        st.markdown(
            '<div class="composer-hint"><b>Type in the chat box below:</b> '
            'for example, "Action camera under $300", '
            '"Only Sony noise-cancelling headphones", '
            '"Compare #1 and #2", '
            'or "Show another brand".</div>',
            unsafe_allow_html=True,
        )
    typed_input = st.chat_input("Type specific details: budget / brand / compare targets / product type")
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
