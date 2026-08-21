from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import agent_common
from tracing_common import last_token_usage


def test_extract_assistant_reply_ai_and_plain():
    assert agent_common.extract_assistant_reply([]) == ""
    assert (
        agent_common.extract_assistant_reply([AIMessage(content="  hi  ")]) == "hi"
    )
    assert agent_common.extract_assistant_reply([HumanMessage(content="q")]) == "q"


def test_make_chat_ollama_defaults_and_overrides():
    llm = agent_common.make_chat_ollama()
    assert llm.model == agent_common.DEFAULT_OLLAMA_MODEL
    assert llm.temperature == agent_common.DEFAULT_TEMPERATURE

    custom = agent_common.make_chat_ollama(
        model="  mistral:latest  ",
        temperature=0.7,
        top_k=20,
        base_url="http://127.0.0.1:11434",
    )
    assert custom.model == "mistral:latest"
    assert custom.temperature == 0.7
    assert custom.top_k == 20

    empty = agent_common.make_chat_ollama(model="   ", base_url="  ")
    assert empty.model == agent_common.DEFAULT_OLLAMA_MODEL


def test_invoke_agent_reraises_after_finishing_span():
    class Boom:
        def invoke(self, *_a, **_k):
            raise RuntimeError("llm down")

    try:
        agent_common.invoke_agent(Boom(), "hello", agent_name="weather")
    except RuntimeError as e:
        assert "llm down" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_invoke_agent_empty_question():
    class Boom:
        def invoke(self, *_a, **_k):
            raise AssertionError("should not run")

    assert agent_common.invoke_agent(Boom(), "   ") == ""


def test_invoke_agent_named_run_and_tokens(monkeypatch):
    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)

    class FakeGraph:
        def invoke(self, payload, config):
            assert payload["messages"][0].content == "Weather in Rome"
            assert config["run_name"] == "weather"
            assert "weather" in config["tags"]
            assert "cli" in config["tags"]
            return {
                "messages": [
                    AIMessage(
                        content="Sunny, 22°C",
                        usage_metadata={
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "total_tokens": 16,
                        },
                    )
                ]
            }

    reply = agent_common.invoke_agent(
        FakeGraph(),
        "Weather in Rome",
        agent_name="weather",
        tags=["unit"],
    )
    assert reply == "Sunny, 22°C"
    usage = last_token_usage()
    assert usage["total_tokens"] == 16
    assert usage["llm_calls"] == 1
    assert usage["by_agent"]["weather"]["input_tokens"] == 12


def test_invoke_agent_attaches_trip_after_specialist(monkeypatch):
    import trip_state as ts
    from contextlib import contextmanager

    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)
    captured: dict = {}

    class Run:
        def set(self, **kwargs):
            captured["set"] = kwargs

    @contextmanager
    def fake_trace(name, **kwargs):
        captured["inputs"] = kwargs.get("inputs")
        captured["tags"] = kwargs.get("tags")
        yield Run()

    monkeypatch.setattr(agent_common, "agent_trace", fake_trace)

    class FakeGraph:
        def invoke(self, payload, config):
            ts.update_trip(
                origin="Mumbai",
                destination="Hyderabad",
                start_time="tomorrow 10:00 AM",
                duration_s=31620,
            )
            return {"messages": [AIMessage(content="8 hrs 47 min")]}

    reply = agent_common.invoke_agent(
        FakeGraph(),
        "plan the drive",
        thread_id="sess-ls",
        agent_name="travel",
    )
    assert reply == "8 hrs 47 min"
    trip = captured["set"]["outputs"]["trip"]
    assert trip["origin"] == "Mumbai"
    assert trip["duration_s"] == 31620
    assert trip["trip_id"] == "sess-ls"
    assert captured["set"]["metadata"]["trip"]["start_time"] == "tomorrow 10:00 AM"

    class FollowUp:
        def invoke(self, *_a, **_k):
            return {"messages": [AIMessage(content="added")]}

    follow = agent_common.invoke_agent(
        FollowUp(),
        "add that to my calendar",
        thread_id="sess-ls",
        agent_name="calendar",
    )
    assert follow == "added"
    assert captured["inputs"]["trip"]["origin"] == "Mumbai"
    assert captured["inputs"]["trip"]["duration_s"] == 31620
    assert "trip" in captured["tags"]


def test_invoke_agent_gui_skips_token_print(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_AI_GUI", "1")

    class FakeGraph:
        def invoke(self, payload, config):
            assert "gui" in config["tags"]
            return {"messages": [AIMessage(content="ok")]}

    assert agent_common.invoke_agent(FakeGraph(), "hi", agent_name="weather") == "ok"
    assert capsys.readouterr().err == ""


def test_run_interactive_reset_and_eof(monkeypatch, capsys):
    seq = ["", "/reset", "hello"]

    def fake_input(_prompt):
        if not seq:
            raise EOFError
        return seq.pop(0)

    class Graph:
        def invoke(self, *_a, **_k):
            return {"messages": [AIMessage(content="reply")]}

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)
    agent_common.run_interactive("Title", "hint.", Graph(), agent_name="weather")
    out = capsys.readouterr().out
    assert "session cleared" in out
    assert "reply" in out


def test_run_interactive_keyboard_interrupt(monkeypatch, capsys):
    calls = {"n": 0}

    def fake_input(_prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        if calls["n"] == 2:
            return "go"
        raise EOFError

    class Graph:
        def invoke(self, *_a, **_k):
            raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", fake_input)
    agent_common.run_interactive("T", "h.", Graph(), agent_name="weather")
    out = capsys.readouterr().out
    assert "interrupted" in out


def test_nested_invoke_inherits_trip_state(monkeypatch):
    import trip_state as ts

    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)

    class Inner:
        def invoke(self, *_a, **_k):
            ts.update_trip(origin="Mumbai", destination="Hyderabad", duration_s=31620)
            return {"messages": [AIMessage(content="route ok")]}

    class Outer:
        def invoke(self, *_a, **_k):
            inner_reply = agent_common.invoke_agent(Inner(), "inner", agent_name="travel")
            assert inner_reply == "route ok"
            assert ts.get_trip().duration_s == 31620
            return {"messages": [AIMessage(content="done")]}

    assert (
        agent_common.invoke_agent(
            Outer(), "plan", thread_id="sess-trip", agent_name="orchestrator"
        )
        == "done"
    )
    token = ts.begin_trip_scope("sess-trip")
    try:
        assert ts.get_trip().duration_s == 31620
        assert ts.get_trip().origin == "Mumbai"
    finally:
        ts.reset_trip_scope(token)


def test_invoke_agent_blocks_abuse_without_calling_graph():
    class Boom:
        def invoke(self, *_a, **_k):
            raise AssertionError("guardrail should stop the graph")

    out = agent_common.invoke_agent(
        Boom(),
        "you are an asshole, weather in Rome",
        agent_name="orchestrator",
    )
    assert "abusive" in out.lower()


def test_invoke_agent_blocks_wrong_specialist_hop():
    class Boom:
        def invoke(self, *_a, **_k):
            raise AssertionError("misrouted hop should not run")

    out = agent_common.invoke_agent(
        Boom(),
        "best sushi restaurants in Osaka",
        agent_name="weather",
    )
    assert "weather" in out
    assert "does not belong" in out.lower()
