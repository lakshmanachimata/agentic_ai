from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import streamlit_app
from streamlit_app import CalendarInvite
from tests.fakes import FakeStreamlit, SessionState


def test_load_latest_invite_invalid(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    monkeypatch.setattr(streamlit_app, "LATEST_INVITE_PATH", path)
    path.write_text("{not json", encoding="utf-8")
    assert streamlit_app.load_latest_invite_file() is None
    path.write_text("[]", encoding="utf-8")
    assert streamlit_app.load_latest_invite_file() is None
    path.write_text(json.dumps({"title": "x"}), encoding="utf-8")
    assert streamlit_app.load_latest_invite_file() is None


def test_parse_calendar_invites_empty():
    assert streamlit_app.parse_calendar_invites("") == []
    assert streamlit_app.parse_calendar_invites(None or "") == []


def test_open_url_platforms(monkeypatch):
    monkeypatch.setattr(streamlit_app.sys, "platform", "darwin")
    ran = []
    monkeypatch.setattr(streamlit_app.subprocess, "run", lambda *a, **k: ran.append(a))
    ok, msg = streamlit_app.open_url("https://calendar.google.com/x")
    assert ok is True
    assert "Opened" in msg
    assert ran

    monkeypatch.setattr(streamlit_app.sys, "platform", "linux")
    monkeypatch.setattr(streamlit_app.webbrowser, "open", lambda *_a, **_k: True)
    ok, msg = streamlit_app.open_url("https://calendar.google.com/x")
    assert ok is True

    def boom(*_a, **_k):
        raise OSError("blocked")

    monkeypatch.setattr(streamlit_app.webbrowser, "open", boom)
    ok, msg = streamlit_app.open_url("https://calendar.google.com/x")
    assert ok is False
    assert "Could not open" in msg


def test_list_ollama_models(monkeypatch):
    fake_st = FakeStreamlit()
    fake_st.session_state = SessionState(llm_model="custom:tag")
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    class Proc:
        returncode = 0
        stdout = (
            "NAME ID\n"
            "qwen3.5:latest abc\n"
            "nomic-embed-text xyz\n"
            "\n"
            "z-image-foo 1\n"
            "mistral:latest 2\n"
        )

    monkeypatch.setattr(streamlit_app.subprocess, "run", lambda *_a, **_k: Proc())
    models = streamlit_app.list_ollama_models()
    assert "qwen3.5:latest" in models
    assert "mistral:latest" in models
    assert "custom:tag" in models
    assert all("embed" not in m for m in models)

    def missing(*_a, **_k):
        raise FileNotFoundError("no ollama")

    monkeypatch.setattr(streamlit_app.subprocess, "run", missing)
    fallback = streamlit_app.list_ollama_models()
    assert fallback[0] == "custom:tag"


def test_get_graph_unwrap(monkeypatch):
    monkeypatch.setitem(
        streamlit_app.CLIENTS,
        "weather",
        ("Weather", "hint", lambda **k: {"model": k["model"], "top_k": k["top_k"]}),
    )
    fn = getattr(streamlit_app.get_graph, "__wrapped__", streamlit_app.get_graph)
    graph = fn("weather", "mistral", 0.2, 7)
    assert graph["model"] == "mistral"
    assert graph["top_k"] == 7


def test_render_invite_click(monkeypatch):
    fake = FakeStreamlit(add_calendar=True)
    monkeypatch.setattr(streamlit_app, "st", fake)
    monkeypatch.setattr(streamlit_app, "open_url", lambda _u: (True, "Opened"))
    inv = CalendarInvite(
        title="Trip",
        when="tomorrow",
        location="Mumbai",
        notes="drive",
        ics_path="/tmp/a.ics",
        gcal_url="https://calendar.google.com/calendar/render?action=TEMPLATE",
    )
    streamlit_app.render_invite(inv, 0)
    assert fake.successes

    monkeypatch.setattr(streamlit_app, "open_url", lambda _u: (False, "fail"))
    streamlit_app.render_invite(inv, 1)
    assert fake.errors

    empty = CalendarInvite(title="No url")
    streamlit_app.render_invite(empty, 2)


def test_init_state(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake)
    streamlit_app._init_state()
    assert fake.session_state.client_key == "orchestrator"
    streamlit_app._init_state()
    assert fake.session_state.messages == []


def test_main_empty_prompt(monkeypatch):
    fake = FakeStreamlit(chat_prompt=None)
    monkeypatch.setattr(streamlit_app, "st", fake)
    monkeypatch.setattr(streamlit_app, "list_ollama_models", lambda: ["qwen3.5:latest"])
    streamlit_app.main()
    assert fake.titles


def test_main_clear_chat(monkeypatch):
    fake = FakeStreamlit(clear_chat=True, chat_prompt=None)
    monkeypatch.setattr(streamlit_app, "st", fake)
    monkeypatch.setattr(streamlit_app, "list_ollama_models", lambda: ["qwen3.5:latest"])
    with pytest.raises(RuntimeError, match="streamlit-rerun"):
        streamlit_app.main()


def test_main_switch_client_and_query(monkeypatch, tmp_path):
    fake = FakeStreamlit(chat_prompt="Weather in Rome", selected_client="weather")
    monkeypatch.setattr(streamlit_app, "st", fake)
    monkeypatch.setattr(streamlit_app, "list_ollama_models", lambda: ["qwen3.5:latest"])
    monkeypatch.setattr(streamlit_app, "get_graph", MagicMock(return_value="graph"))
    streamlit_app.get_graph.clear = lambda: None
    monkeypatch.setattr(streamlit_app, "invoke_agent", lambda *_a, **_k: "Sunny")
    monkeypatch.setattr(
        streamlit_app,
        "last_token_usage",
        lambda: {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "llm_calls": 1,
            "by_agent": {},
        },
    )
    invite_path = tmp_path / "latest.json"
    monkeypatch.setattr(streamlit_app, "LATEST_INVITE_PATH", invite_path)
    streamlit_app.main()
    assert any(m.get("role") == "assistant" for m in fake.session_state.messages)


def test_main_invoke_error_and_invite_rerun(monkeypatch, tmp_path):
    url = "https://calendar.google.com/calendar/render?action=TEMPLATE&text=Trip"
    reply = (
        "Title: Drive\n"
        f"When: tomorrow\nQuick-add link: {url}\n"
    )
    fake = FakeStreamlit(chat_prompt="add trip")
    fake.session_state.messages = [
        {
            "role": "assistant",
            "content": "hi",
            "tokens": {"llm_calls": 1, "total_tokens": 4},
        }
    ]
    fake.session_state.invites = [
        CalendarInvite(title="old", gcal_url="https://other"),
    ]
    monkeypatch.setattr(streamlit_app, "st", fake)
    monkeypatch.setattr(streamlit_app, "list_ollama_models", lambda: ["qwen3.5:latest"])
    monkeypatch.setattr(streamlit_app, "get_graph", MagicMock(return_value="graph"))
    streamlit_app.get_graph.clear = lambda: None

    def boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(streamlit_app, "invoke_agent", boom)
    monkeypatch.setattr(
        streamlit_app,
        "last_token_usage",
        lambda: {"llm_calls": 0, "total_tokens": 0},
    )
    streamlit_app.main()
    assert "Error:" in fake.session_state.messages[-1]["content"]

    fake.chat_prompt = "add"
    monkeypatch.setattr(streamlit_app, "invoke_agent", lambda *_a, **_k: reply)
    invite_path = tmp_path / "latest.json"
    invite_path.write_text(
        json.dumps(
            {
                "title": "Drive",
                "when": "x",
                "location": "",
                "notes": "",
                "ics_path": "/tmp/a.ics",
                "gcal_url": url,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(streamlit_app, "LATEST_INVITE_PATH", invite_path)
    with pytest.raises(RuntimeError, match="streamlit-rerun"):
        streamlit_app.main()
