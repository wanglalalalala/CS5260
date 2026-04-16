from __future__ import annotations

import html
import importlib
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

PROVIDER_OPTIONS = ("openai", "claude", "gemini", "qwen")
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "claude": "CLAUDE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}
PROVIDER_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "claude": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.0-flash",
    "qwen": "qwen-plus",
}


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
    --font-sans: "Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
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
[data-testid="stButton"] button, [data-testid="stChatInput"] input, label, p {
    font-family: var(--font-sans) !important;
}

/* Keep Streamlit's icon ligature fonts intact (do not override to Inter). */
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-icons,
[class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
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
    line-height: 1.3;
}
.item-name a.item-link {
    color: var(--ink-navy);
    text-decoration: none;
    border-bottom: 1px solid rgba(214, 138, 46, 0.55);
}
.item-name a.item-link:hover {
    color: #0f3f67;
    border-bottom-color: rgba(214, 138, 46, 0.9);
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
    for key in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "CLAUDE_API_KEY",
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_REGION",
        "DASHSCOPE_BASE_URL",
        "GEMINI_BASE_URL",
        "HF_TOKEN",
    ):
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
            "secrets — choose one of: openai / claude / gemini / qwen."
        )
    key_env = PROVIDER_KEY_ENV.get(provider)
    if not key_env:
        return False, provider, (
            f"Unsupported LLM_PROVIDER=`{provider}`. Use one of: "
            "openai / claude / gemini / qwen."
        )
    if not os.environ.get(key_env):
        return False, provider, (
            f"{key_env} is missing. Set it in `backend/.env`, Streamlit "
            "secrets, or apply it in the sidebar."
        )
    return True, provider, provider


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
                    "_\"Noise-cancelling headphones under \\$200 for work calls\"_."
                ),
            }
        ]
    if "applied_filters" not in st.session_state:
        st.session_state.applied_filters = []
    if "recommended_items" not in st.session_state:
        st.session_state.recommended_items = []
    if "token_logger" not in st.session_state:
        st.session_state.token_logger = TokenLogger()
    if "provider_input" not in st.session_state:
        provider = (os.environ.get("LLM_PROVIDER") or "openai").lower().strip()
        if provider not in PROVIDER_OPTIONS:
            provider = "openai"
        st.session_state.provider_input = provider
    if "last_provider_input" not in st.session_state:
        st.session_state.last_provider_input = st.session_state.provider_input
    if "model_name_input" not in st.session_state:
        st.session_state.model_name_input = (
            os.environ.get("LLM_MODEL")
            or PROVIDER_DEFAULT_MODEL[st.session_state.provider_input]
        )
    if "api_key_input" not in st.session_state:
        key_env = PROVIDER_KEY_ENV[st.session_state.provider_input]
        st.session_state.api_key_input = os.environ.get(key_env, "")
    if "qwen_region_input" not in st.session_state:
        st.session_state.qwen_region_input = os.environ.get("DASHSCOPE_REGION", "cn")
    if "config_feedback" not in st.session_state:
        st.session_state.config_feedback = None
    if "show_settings_welcome" not in st.session_state:
        st.session_state.show_settings_welcome = True
    if "config_confirmed" not in st.session_state:
        st.session_state.config_confirmed = False
    if "configured_provider" not in st.session_state:
        st.session_state.configured_provider = st.session_state.provider_input
    if "configured_model" not in st.session_state:
        st.session_state.configured_model = st.session_state.model_name_input
    if "configured_api_key" not in st.session_state:
        st.session_state.configured_api_key = st.session_state.api_key_input
    if "configured_qwen_region" not in st.session_state:
        st.session_state.configured_qwen_region = st.session_state.qwen_region_input
    if "sidebar_settings_open" not in st.session_state:
        st.session_state.sidebar_settings_open = False


def _clear_session() -> None:
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.applied_filters = []
    st.session_state.recommended_items = []
    st.session_state.token_logger.reset_session()
    real_agent.reset_agent()
    st.rerun()


def _apply_runtime_backend_config(
    provider: str,
    model_name: str,
    api_key: str,
    qwen_region: str,
) -> tuple[bool, str]:
    provider = (provider or "").strip().lower()
    model_name = (model_name or "").strip()
    api_key = (api_key or "").strip()
    if provider not in PROVIDER_OPTIONS:
        return False, "Please choose one provider from openai / claude / gemini / qwen."
    if not model_name:
        return False, "Model name cannot be empty."
    if not api_key:
        return False, "API key cannot be empty."

    key_env = PROVIDER_KEY_ENV[provider]
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model_name
    os.environ[key_env] = api_key

    if provider == "qwen":
        os.environ["DASHSCOPE_REGION"] = qwen_region or "cn"

    dialogue_llm = importlib.import_module("agent.dialogue.llm")
    # Streamlit may keep an older in-memory module during hot reload; support
    # both new and old backend builds when resetting the singleton client.
    dialogue_llm = importlib.reload(dialogue_llm)
    if hasattr(dialogue_llm, "reset_client"):
        dialogue_llm.reset_client()
    elif hasattr(dialogue_llm, "_default_client"):
        dialogue_llm._default_client = None
    real_agent.reset_agent()
    return True, f"Applied {provider} with model `{model_name}`."


def _render_settings_controls(hide_welcome_on_confirm: bool = False) -> Dict[str, str]:
    st.markdown("### Session Controls")
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_OPTIONS),
        key="provider_input",
        format_func=lambda x: x.upper(),
    )
    if st.session_state.last_provider_input != provider:
        prev_provider = st.session_state.last_provider_input
        prev_default = PROVIDER_DEFAULT_MODEL.get(prev_provider, "")
        if (
            not st.session_state.model_name_input
            or st.session_state.model_name_input == prev_default
        ):
            st.session_state.model_name_input = PROVIDER_DEFAULT_MODEL[provider]
        st.session_state.api_key_input = ""
        st.session_state.last_provider_input = provider

    model_name = st.text_input(
        "Model Name",
        key="model_name_input",
        placeholder=PROVIDER_DEFAULT_MODEL[provider],
    )
    api_key = st.text_input(
        f"{PROVIDER_KEY_ENV[provider]}",
        key="api_key_input",
        type="password",
        placeholder="Enter API key",
    )
    if api_key:
        st.caption("API key provided.")
    else:
        st.caption("Please enter an API key for the selected provider.")

    qwen_region = st.session_state.qwen_region_input
    if provider == "qwen":
        qwen_region = st.selectbox(
            "Qwen Region",
            options=["cn", "intl"],
            key="qwen_region_input",
            help="cn uses the mainland endpoint, intl uses the global endpoint.",
        )

    if st.button("Clear Session", use_container_width=True):
        _clear_session()

    if st.button("Confirm Configuration", type="primary", use_container_width=True):
        with st.spinner("Applying backend configuration..."):
            ok, message = _apply_runtime_backend_config(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                qwen_region=qwen_region,
            )
        st.session_state.config_feedback = ("success" if ok else "error", message)
        if ok:
            st.session_state.config_confirmed = True
            st.session_state.configured_provider = provider
            st.session_state.configured_model = model_name
            st.session_state.configured_api_key = api_key
            st.session_state.configured_qwen_region = qwen_region
            countdown = st.empty()
            for sec in [3, 2, 1]:
                countdown.info(f"Configuration successful. Auto closing in {sec}s...")
                time.sleep(1)
            if hide_welcome_on_confirm:
                st.session_state.show_settings_welcome = False
            else:
                st.session_state.sidebar_settings_open = False
            st.rerun()
        else:
            st.session_state.config_confirmed = False

    feedback = st.session_state.config_feedback
    if feedback:
        level, message = feedback
        if level == "success":
            st.success(message)
        else:
            st.error(message)

    return {
        "provider": provider,
        "model": model_name,
        "api_key": api_key,
        "qwen_region": qwen_region,
    }


def render_settings_entry(in_sidebar: bool = False) -> Dict[str, str]:
    if st.session_state.show_settings_welcome and not in_sidebar:
        with st.container(border=True):
            left, right = st.columns([10, 1])
            with left:
                st.markdown("### Quick Setup")
                st.caption(
                    "Initial settings panel. You can reopen these options from the "
                    "sidebar Settings button."
                )
            with right:
                if st.button("✕", key="close_welcome_settings", help="Close this panel"):
                    st.session_state.show_settings_welcome = False
                    st.rerun()
            return _render_settings_controls(hide_welcome_on_confirm=True)

    if st.session_state.show_settings_welcome and in_sidebar:
        return {
            "provider": st.session_state.provider_input,
            "model": st.session_state.model_name_input,
            "api_key": st.session_state.api_key_input,
            "qwen_region": st.session_state.qwen_region_input,
        }

    if in_sidebar:
        label = (
            "⚙ Settings  ▴"
            if st.session_state.sidebar_settings_open
            else "⚙ Settings  ▾"
        )
        if st.button(label, key="toggle_sidebar_settings", use_container_width=False):
            st.session_state.sidebar_settings_open = not st.session_state.sidebar_settings_open
            st.rerun()
        if not st.session_state.sidebar_settings_open:
            return {
                "provider": st.session_state.provider_input,
                "model": st.session_state.model_name_input,
                "api_key": st.session_state.api_key_input,
                "qwen_region": st.session_state.qwen_region_input,
            }
        with st.container(border=True):
            return _render_settings_controls()

    return _render_settings_controls()


def render_backend_settings_panel() -> tuple[bool, str, str, str]:
    with st.sidebar:
        render_settings_entry(in_sidebar=True)

    ok, active_provider, message = _resolve_provider_status()
    display_model = os.environ.get("LLM_MODEL") or PROVIDER_DEFAULT_MODEL.get(
        active_provider or st.session_state.provider_input,
        "unknown",
    )
    return ok, active_provider, message, display_model


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
                name = html.escape(item["name"])
                url = item.get("url") or ""
                if url:
                    name_html = (
                        f'<a href="{html.escape(url)}" target="_blank" '
                        f'rel="noopener" class="item-link">{name}</a>'
                    )
                else:
                    name_html = name
                price_val = item.get("price") or 0
                if item.get("price_is_estimate"):
                    price_str = f"~${price_val:.0f} (est.)"
                elif price_val:
                    price_str = f"${price_val:.0f}"
                else:
                    price_str = "price N/A"
                cards.append(
                    '<div class="item-card">'
                    f'<div class="item-name">{name_html}</div>'
                    f'<div class="item-meta">{html.escape(item["brand"])} · '
                    f'{html.escape(price_str)} · ⭐ {item["rating"]:.1f}</div>'
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
    importlib.import_module("agent.dialogue.llm")

    init_state()

    st.markdown(
        '<div class="page-title-wrap"><span class="page-title-text">Intelligent Shopping Assistant</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.show_settings_welcome:
        render_settings_entry(in_sidebar=False)

    ok, provider, message, display_model = render_backend_settings_panel()
    if not ok:
        render_sidebar_panels(provider_label=provider or "not configured")
        render_chat()
        st.chat_input("Configure backend to start chatting.", disabled=True)
        return

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
