#!/usr/bin/env python3
"""
Streamlit GUI for the multi-agent travel assistant.

Default client: orchestrator (routes weather / travel / dining / calendar).

Run:
  AGENTIC_AI_GUI=1 streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import uuid
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

# Skip calendar auto-open; UI button opens Google Calendar instead.
os.environ["AGENTIC_AI_GUI"] = "1"

import json

from agent_common import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    invoke_agent,
)
from tracing_common import tracing_banner
from calendar_agent import LATEST_INVITE_PATH, build_agent as build_calendar
from orchestrator_agent import build_agent as build_orchestrator
from restaurant_agent import build_agent as build_restaurant
from travel_agent import build_agent as build_travel
from weather_agent import build_agent as build_weather

ClientBuilder = Callable[..., Any]

CLIENTS: dict[str, tuple[str, str, ClientBuilder]] = {
    "orchestrator": (
        "Orchestrator",
        "Routes to weather, travel, dining, and calendar specialists",
        build_orchestrator,
    ),
    "calendar": (
        "Calendar",
        "Travel calendar events (.ics + Google link)",
        build_calendar,
    ),
    "weather": ("Weather", "Conditions and forecasts", build_weather),
    "travel": ("Travel", "Drive / walk / bike times and route towns", build_travel),
    "restaurants": ("Restaurants", "Dining near a place or along a route", build_restaurant),
}

DEFAULT_MODEL_CHOICES = [
    "qwen3.5:latest",
    "qwen2.5:7b",
    "mistral:latest",
]

GCAL_URL_RE = re.compile(
    r"https://calendar\.google\.com/calendar/render\?[^\s\)\]\"']+",
    re.IGNORECASE,
)
ICS_PATH_RE = re.compile(
    r"ICS file \(import into Google Calendar\):\s*(.+)",
    re.IGNORECASE,
)


@dataclass
class CalendarInvite:
    title: str = ""
    when: str = ""
    location: str = ""
    notes: str = ""
    ics_path: str = ""
    gcal_url: str = ""
    raw_snippet: str = ""


def _field(text: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def load_latest_invite_file() -> CalendarInvite | None:
    """Read structured invite written by calendar_agent (survives paraphrased replies)."""
    try:
        if not LATEST_INVITE_PATH.is_file():
            return None
        data = json.loads(LATEST_INVITE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    url = str(data.get("gcal_url") or "").strip()
    if not url:
        return None
    return CalendarInvite(
        title=str(data.get("title") or "Calendar event"),
        when=str(data.get("when") or ""),
        location=str(data.get("location") or ""),
        notes=str(data.get("notes") or ""),
        ics_path=str(data.get("ics_path") or ""),
        gcal_url=url,
        raw_snippet="",
    )


def parse_calendar_invites(
    text: str,
    *,
    include_latest_file: bool = False,
) -> list[CalendarInvite]:
    """Extract invite cards from agent text and optionally latest_invite.json."""
    invites: list[CalendarInvite] = []

    if include_latest_file:
        file_inv = load_latest_invite_file()
        if file_inv:
            invites.append(file_inv)

    if text:
        urls = GCAL_URL_RE.findall(text)
        blocks = re.split(r"(?=^Title:\s)", text, flags=re.MULTILINE)
        for block in blocks:
            url_m = GCAL_URL_RE.search(block)
            title = _field(block, "Title")
            when = _field(block, "When")
            if not title and not url_m:
                continue
            ics_m = ICS_PATH_RE.search(block)
            invites.append(
                CalendarInvite(
                    title=title or "Calendar event",
                    when=when,
                    location=_field(block, "Location"),
                    notes=_field(block, "Notes"),
                    ics_path=(ics_m.group(1).strip() if ics_m else ""),
                    gcal_url=(url_m.group(0) if url_m else ""),
                    raw_snippet=block.strip()[:1200],
                )
            )
        has_text_url = any(inv.gcal_url and inv.raw_snippet for inv in invites)
        if not has_text_url and urls and not include_latest_file:
            invites.append(
                CalendarInvite(
                    title="Calendar event",
                    when=_field(text, "When"),
                    location=_field(text, "Location"),
                    gcal_url=urls[-1],
                    ics_path=(
                        ICS_PATH_RE.search(text).group(1).strip()
                        if ICS_PATH_RE.search(text)
                        else ""
                    ),
                    raw_snippet=text.strip()[:1200],
                )
            )

    seen: set[str] = set()
    unique: list[CalendarInvite] = []
    for inv in invites:
        key = inv.gcal_url or f"{inv.title}|{inv.when}"
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(inv)
    return unique


def open_url(url: str) -> tuple[bool, str]:
    url = (url or "").strip()
    if not url:
        return False, "No Google Calendar URL available."
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", url], check=False)
        else:
            webbrowser.open(url, new=2)
        return True, "Opened Google Calendar in your browser — click Save on that page."
    except Exception as e:
        return False, f"Could not open browser: {e}"


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "client_key" not in st.session_state:
        st.session_state.client_key = "orchestrator"
    if "invites" not in st.session_state:
        st.session_state.invites = []
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = DEFAULT_OLLAMA_MODEL
    if "llm_temperature" not in st.session_state:
        st.session_state.llm_temperature = float(DEFAULT_TEMPERATURE)
    if "llm_top_k" not in st.session_state:
        st.session_state.llm_top_k = int(DEFAULT_TOP_K)


def list_ollama_models() -> list[str]:
    """Return local Ollama tags; fall back to built-in defaults."""
    models: list[str] = []
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            for i, line in enumerate(proc.stdout.splitlines()):
                if i == 0 and line.lower().startswith("name"):
                    continue
                name = line.split()[0].strip() if line.strip() else ""
                # Skip non-chat embed / image tags where possible
                if not name:
                    continue
                lower = name.lower()
                if "embed" in lower or "z-image" in lower:
                    continue
                models.append(name)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    merged: list[str] = []
    for name in DEFAULT_MODEL_CHOICES + models:
        if name not in merged:
            merged.append(name)
    if st.session_state.get("llm_model") and st.session_state.llm_model not in merged:
        merged.insert(0, st.session_state.llm_model)
    return merged or [DEFAULT_OLLAMA_MODEL]


@st.cache_resource(show_spinner="Loading agent…")
def get_graph(
    client_key: str,
    model: str,
    temperature: float,
    top_k: int,
) -> Any:
    _name, _hint, builder = CLIENTS[client_key]
    return builder(model=model, temperature=temperature, top_k=top_k)


def render_invite(inv: CalendarInvite, idx: int) -> None:
    with st.container(border=True):
        st.markdown(f"### {inv.title or 'Calendar invite'}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**When:** {inv.when or '—'}")
            st.markdown(f"**Location:** {inv.location or '—'}")
        with c2:
            if inv.ics_path:
                st.caption("ICS file")
                st.code(inv.ics_path, language=None)
            if inv.notes:
                st.caption("Notes")
                st.write(inv.notes[:400])

        btn_col, link_col = st.columns([1, 2])
        with btn_col:
            if st.button(
                "Add to Calendar",
                key=f"add_cal_{idx}_{hash(inv.gcal_url or inv.title)}",
                type="primary",
                disabled=not inv.gcal_url,
                use_container_width=True,
            ):
                ok, msg = open_url(inv.gcal_url)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        with link_col:
            if inv.gcal_url:
                st.link_button(
                    "Open Google Calendar link",
                    inv.gcal_url,
                    use_container_width=True,
                )
            else:
                st.warning("No Google Calendar URL found in the agent reply.")


def main() -> None:
    st.set_page_config(
        page_title="Agentic AI Travel Assistant",
        page_icon="🧭",
        layout="wide",
    )
    _init_state()

    st.title("Agentic AI Travel Assistant")
    st.caption(
        "Ask about weather, travel time, restaurants, or add a trip to Google Calendar."
    )

    with st.sidebar:
        st.header("Settings")
        client_labels = {k: v[0] for k, v in CLIENTS.items()}
        selected = st.selectbox(
            "Agent",
            options=list(CLIENTS.keys()),
            format_func=lambda k: client_labels[k],
            index=list(CLIENTS.keys()).index(st.session_state.client_key)
            if st.session_state.client_key in CLIENTS
            else 0,
            help="Default is Orchestrator.",
        )
        if selected != st.session_state.client_key:
            st.session_state.client_key = selected
            st.session_state.thread_id = str(uuid.uuid4())
            get_graph.clear()

        st.write(CLIENTS[st.session_state.client_key][1])

        st.divider()
        st.subheader("LLM")
        model_options = list_ollama_models()
        if st.session_state.llm_model not in model_options:
            model_options = [st.session_state.llm_model] + model_options
        st.selectbox(
            "Model (Ollama)",
            options=model_options,
            key="llm_model",
            help="Local Ollama chat models. Pull with `ollama pull <name>`.",
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            step=0.05,
            key="llm_temperature",
            help="Lower = more deterministic tool use; higher = more creative answers.",
        )
        st.slider(
            "Top-k",
            min_value=1,
            max_value=100,
            step=1,
            key="llm_top_k",
            help="Limits sampling to the top-k tokens (Ollama).",
        )
        st.caption(
            f"Active: `{st.session_state.llm_model}` · "
            f"temp={float(st.session_state.llm_temperature):.2f} · "
            f"top_k={int(st.session_state.llm_top_k)}"
        )
        st.caption("Changing model/params rebuilds the agent (cached per setting).")

        st.divider()
        st.subheader("LangSmith")
        st.caption(tracing_banner())
        st.markdown(
            "[Open LangSmith](https://smith.langchain.com) — copy `.env.example` to `.env` "
            "and set `LANGSMITH_API_KEY` to trace orchestrator → specialist → tool → API hops."
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.invites = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        st.divider()
        st.markdown(
            "**Examples**\n\n"
            "- Weather in Tokyo\n"
            "- Drive Mumbai to Hyderabad tomorrow 10 AM\n"
            "- Plan Mumbai → Hyderabad tomorrow 10 AM and add to calendar\n"
            "- Restaurants near London Bridge"
        )
        st.caption("Requires Ollama running locally.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Calendar invites from latest replies
    if st.session_state.invites:
        st.subheader("Calendar invite")
        for i, inv in enumerate(st.session_state.invites):
            render_invite(inv, i)

    prompt = st.chat_input("Ask the assistant…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking / calling specialists…"):
            before_mtime = (
                LATEST_INVITE_PATH.stat().st_mtime
                if LATEST_INVITE_PATH.is_file()
                else 0.0
            )
            try:
                graph = get_graph(
                    st.session_state.client_key,
                    st.session_state.llm_model,
                    float(st.session_state.llm_temperature),
                    int(st.session_state.llm_top_k),
                )
                reply = invoke_agent(
                    graph,
                    prompt,
                    thread_id=st.session_state.thread_id,
                    agent_name=st.session_state.client_key,
                    tags=["streamlit"],
                )
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(reply or "_(empty reply)_")

    st.session_state.messages.append({"role": "assistant", "content": reply or ""})

    after_mtime = (
        LATEST_INVITE_PATH.stat().st_mtime if LATEST_INVITE_PATH.is_file() else 0.0
    )
    found = parse_calendar_invites(
        reply or "",
        include_latest_file=after_mtime > before_mtime,
    )
    if found:
        existing_urls = {inv.gcal_url for inv in st.session_state.invites if inv.gcal_url}
        for inv in found:
            if inv.gcal_url and inv.gcal_url in existing_urls:
                continue
            st.session_state.invites.append(inv)
        st.rerun()


# streamlit run streamlit_app.py executes this file as __main__
if __name__ == "__main__":
    main()
