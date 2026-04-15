from __future__ import annotations

import html
import json
import time
import uuid
from typing import Dict, List

import streamlit as st

from services.mock_agent import generate_reply, stream_text
from utils.token_logger import TokenLogger


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
[data-testid="stMain"] {
    background: #b2b4b9;
}
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
[data-testid="stSidebarContent"] {
    background: #e7eaf4;
}
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
div[data-testid="stPopover"] button p {
    font-size: 0.78rem !important;
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
    left: auto;
    transform: none;
    width: 100%;
    max-width: 100%;
    z-index: 1000;
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
    padding: 0.15rem 0.35rem;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] {
    margin-bottom: 0.35rem;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button {
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 12px;
    background: color-mix(in srgb, var(--background-color) 92%, #e0e7ff);
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button:hover {
    border-color: rgba(99, 102, 241, 0.46);
    background: color-mix(in srgb, var(--background-color) 84%, #ddd6fe);
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


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I am your AI shopping assistant. "
                    "Tell me your budget and use case, for example: "
                    "'Need wireless headphones under $200 for work calls'."
                ),
            }
        ]
    if "applied_filters" not in st.session_state:
        st.session_state.applied_filters = []
    if "recommended_items" not in st.session_state:
        st.session_state.recommended_items = []
    if "token_logger" not in st.session_state:
        st.session_state.token_logger = TokenLogger()
    if "model_name_input" not in st.session_state:
        st.session_state.model_name_input = "gpt-4o"
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
    if "backend_mode" not in st.session_state:
        st.session_state.backend_mode = "mock"
    if "show_settings_welcome" not in st.session_state:
        st.session_state.show_settings_welcome = True
    if "config_confirmed" not in st.session_state:
        st.session_state.config_confirmed = False
    if "configured_model" not in st.session_state:
        st.session_state.configured_model = ""
    if "configured_api_key" not in st.session_state:
        st.session_state.configured_api_key = ""
    if "configured_backend_mode" not in st.session_state:
        st.session_state.configured_backend_mode = "mock"
    if "config_feedback" not in st.session_state:
        st.session_state.config_feedback = None
    if "sidebar_settings_open" not in st.session_state:
        st.session_state.sidebar_settings_open = False


def _clear_session() -> None:
    st.session_state.messages = st.session_state.messages[:1]
    st.session_state.applied_filters = []
    st.session_state.recommended_items = []
    st.session_state.token_logger.reset_session()
    st.rerun()


def _apply_backend_configuration(model_name: str, api_key: str, backend_mode: str) -> tuple[bool, str]:
    model_name = model_name.strip()
    if not model_name:
        return False, "Please enter a model name."
    if backend_mode.startswith("real") and not api_key.strip():
        return False, "API key is required when using real backend mode."
    # Placeholder for future backend setup call.
    return True, f"Configuration applied: {model_name} ({backend_mode})."


def _render_settings_controls(hide_welcome_on_confirm: bool = False) -> Dict[str, str]:
    st.markdown("### Session Controls")
    model_name = st.text_input(
        "Model Name",
        key="model_name_input",
        placeholder="e.g. gpt-4o-mini",
    )
    api_key = st.text_input(
        "API Key",
        key="api_key_input",
        type="password",
        placeholder="Enter your provider API key",
    )
    if api_key:
        st.caption("API key provided.")
    else:
        st.caption("Enter API key when connecting real backend.")
    backend_mode = st.radio(
        "Backend",
        options=["mock", "real (not connected yet)"],
        key="backend_mode",
        horizontal=True,
    )
    if st.button("Clear Session", use_container_width=True):
        _clear_session()

    if st.button("Confirm Configuration", type="primary", use_container_width=True):
        with st.spinner("Applying backend configuration..."):
            ok, message = _apply_backend_configuration(model_name, api_key, backend_mode)
        if ok:
            st.session_state.config_confirmed = True
            st.session_state.configured_model = model_name
            st.session_state.configured_api_key = api_key
            st.session_state.configured_backend_mode = backend_mode
            st.session_state.config_feedback = ("success", message)
            st.success(message)
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
            st.session_state.config_feedback = ("error", message)
            st.error(message)
            return {"model": model_name, "api_key": api_key, "backend_mode": backend_mode}

    feedback = st.session_state.config_feedback
    if feedback:
        level, message = feedback
        if level == "success":
            st.success(message)
        else:
            st.error(message)

    return {"model": model_name, "api_key": api_key, "backend_mode": backend_mode}


def render_settings_entry(in_sidebar: bool = False) -> Dict[str, str]:
    if st.session_state.show_settings_welcome and not in_sidebar:
        with st.container(border=True):
            left, right = st.columns([10, 1])
            with left:
                st.markdown("### Quick Setup")
                st.caption("Initial settings panel. You can reopen these options from the ⚙ button later.")
            with right:
                if st.button("✕", key="close_welcome_settings", help="Close this panel"):
                    st.session_state.show_settings_welcome = False
                    st.rerun()
            return _render_settings_controls(hide_welcome_on_confirm=True)

    if st.session_state.show_settings_welcome and in_sidebar:
        return {
            "model": st.session_state.model_name_input,
            "api_key": st.session_state.api_key_input,
            "backend_mode": st.session_state.backend_mode,
        }

    if in_sidebar:
        label = "⚙ Settings  ▴" if st.session_state.sidebar_settings_open else "⚙ Settings  ▾"
        if st.button(label, key="toggle_sidebar_settings", use_container_width=False):
            st.session_state.sidebar_settings_open = not st.session_state.sidebar_settings_open
            st.rerun()
        if not st.session_state.sidebar_settings_open:
            return {
                "model": st.session_state.model_name_input,
                "api_key": st.session_state.api_key_input,
                "backend_mode": st.session_state.backend_mode,
            }
        with st.container(border=True):
            return _render_settings_controls()

    return _render_settings_controls()


def render_sidebar_panels() -> Dict[str, str]:
    with st.sidebar:
        settings = render_settings_entry(in_sidebar=True)
        if st.session_state.applied_filters:
            tags = "".join(
                [f'<span class="filter-tag">{html.escape(value)}</span>' for value in st.session_state.applied_filters]
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
                    (
                        '<div class="item-card">'
                        f'<div class="item-name">{html.escape(item["name"])}</div>'
                        f'<div class="item-meta">{html.escape(item["brand"])} | ${item["price"]:.0f} | ⭐ {item["rating"]:.1f}</div>'
                        f'<div class="item-reason">{html.escape(item["short_reason"])}</div>'
                        "</div>"
                    )
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
    return settings

def run_agent(
    user_message: str,
    history: List[Dict[str, str]],
    model_name: str,
    backend_mode: str,
) -> Dict[str, object]:
    if backend_mode.startswith("mock"):
        return generate_reply(user_message=user_message, history=history)
    raise NotImplementedError(
        "Real backend adapter is not connected yet. "
        "Switch backend mode to mock for now."
    )


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


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning_trace", [])
                tools = msg.get("tool_calls", [])
                render_reasoning_panel(reasoning, tools)
            st.markdown(msg["content"])


def main() -> None:
    init_state()

    st.markdown(
        '<div class="page-title-wrap"><span class="page-title-text">Intelligent Shopping Assistant</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.show_settings_welcome:
        render_settings_entry(in_sidebar=False)

    settings = render_sidebar_panels()
    render_chat()

    if st.session_state.config_confirmed:
        configured_model = st.session_state.configured_model
        configured_backend_mode = st.session_state.configured_backend_mode
        user_input = st.chat_input("Describe what you want to buy...")
    else:
        configured_model = settings["model"]
        configured_backend_mode = settings["backend_mode"]
        user_input = st.chat_input(
            "Please confirm settings before chatting.",
            disabled=True,
        )

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(
                user_message=user_input,
                history=st.session_state.messages,
                model_name=configured_model,
                backend_mode=configured_backend_mode,
            )
        reasoning = response.get("reasoning_trace", [])
        tools = response.get("tool_calls", [])
        render_reasoning_panel(reasoning, tools)
        placeholder = st.empty()
        text = str(response.get("assistant_reply", "No response"))
        assembled = ""
        for chunk in stream_text(text):
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
        }
    )
    st.session_state.applied_filters = response.get("applied_filters", [])
    st.session_state.recommended_items = response.get("recommended_items", [])
    usage = response.get("usage", {})
    st.session_state.token_logger.log_from_usage(
        usage=usage, model=configured_model, metadata={"session_id": st.session_state.session_id}
    )
    st.rerun()


if __name__ == "__main__":
    main()
