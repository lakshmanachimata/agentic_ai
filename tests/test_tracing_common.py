from __future__ import annotations

from types import SimpleNamespace

import tracing_common as tc


class FakeRoute:
    origin_label = "Mumbai"
    dest_label = "Hyderabad"
    profile = "driving"
    duration_s = 31620.0
    distance_m = 704000.0


def test_truthy_values():
    assert tc._truthy("true") is True
    assert tc._truthy("OFF") is False
    assert tc._truthy("") is None
    assert tc._truthy("maybe") is None


def test_drop_http_client_strips_client_and_shrinks_route():
    out = tc.drop_http_client({"client": object(), "place": "Paris", "route": FakeRoute()})
    assert "client" not in out
    assert out["place"] == "Paris"
    assert out["route"]["origin"] == "Mumbai"
    assert out["route"]["destination"] == "Hyderabad"


def test_usage_from_messages_usage_metadata_and_ollama_fallback():
    m1 = SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={},
    )
    m2 = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"prompt_eval_count": 20, "eval_count": 7},
    )
    m3 = SimpleNamespace(usage_metadata=None, response_metadata={})
    total = tc.usage_from_messages([m1, m2, m3])
    assert total["input_tokens"] == 30
    assert total["output_tokens"] == 12
    assert total["total_tokens"] == 42
    assert total["llm_calls"] == 2


def test_usage_from_message_empty():
    assert tc.usage_from_message(SimpleNamespace()) is None


def test_token_rollup_nested_agents():
    tc.begin_usage_span()  # parent
    tc.begin_usage_span()  # child
    child = tc.finish_usage_span(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "llm_calls": 1},
        agent_name="weather",
        run=None,
    )
    parent = tc.finish_usage_span(
        {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11, "llm_calls": 1},
        agent_name="orchestrator",
        run=None,
    )
    assert child["by_agent"]["weather"]["total_tokens"] == 15
    assert parent["total_tokens"] == 26
    assert parent["llm_calls"] == 2
    assert parent["by_agent"]["weather"]["total_tokens"] == 15
    assert parent["by_agent"]["orchestrator"]["total_tokens"] == 11
    assert tc.last_token_usage()["total_tokens"] == 26
    line = tc.format_token_usage(parent)
    assert "26 total" in line
    assert "weather 15" in line


def test_finish_usage_span_attaches_to_run():
    class Run:
        def __init__(self):
            self.extra = {"metadata": {}}
            self.outputs = None
            self.called = None

        def set(self, **kwargs):
            self.called = kwargs

    run = Run()
    tc.begin_usage_span()
    usage = tc.finish_usage_span(
        {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10, "llm_calls": 1},
        agent_name="travel",
        run=run,
        reply="ok",
        model="qwen3.5:latest",
    )
    assert usage["total_tokens"] == 10
    assert run.called["usage_metadata"]["input_tokens"] == 4
    assert "Tokens:" in run.called["outputs"]["token_summary"]
    assert run.called["metadata"]["ls_provider"] == "ollama"


def test_configure_langsmith_disabled(monkeypatch, reset_langsmith_config):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    status = tc.configure_langsmith()
    assert status["enabled"] is False
    assert "OFF" in tc.tracing_banner()


def test_configure_langsmith_on_with_key(monkeypatch, reset_langsmith_config):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "unit-test-project")
    status = tc.configure_langsmith()
    assert status["enabled"] is True
    assert status["project"] == "unit-test-project"
    assert "ON" in tc.tracing_banner()


def test_tracing_banner_missing_key(monkeypatch, reset_langsmith_config):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    status = tc.configure_langsmith()
    assert status["enabled"] is False
    assert "missing" in tc.tracing_banner()


def test_print_tracing_banner_skips_gui(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_AI_GUI", "1")
    tc._BANNER_PRINTED = False
    tc.print_tracing_banner()
    assert capsys.readouterr().err == ""


def test_print_tracing_banner_once(monkeypatch, capsys, reset_langsmith_config):
    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    tc.configure_langsmith()
    tc._BANNER_PRINTED = False
    tc.print_tracing_banner()
    tc.print_tracing_banner()
    err = capsys.readouterr().err
    assert err.count("LangSmith") == 1


def test_configure_cached_and_status(monkeypatch, reset_langsmith_config):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    first = tc.configure_langsmith()
    second = tc.configure_langsmith()
    assert first is second
    assert tc.tracing_status() is first


def test_configure_langsmith_dotenv_and_ls_errors(monkeypatch, reset_langsmith_config):
    import sys
    import types

    fake_dotenv = types.ModuleType("dotenv")
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "   ")

    class BoomLS:
        @staticmethod
        def configure(**_k):
            raise RuntimeError("nope")

    monkeypatch.setitem(sys.modules, "langsmith", BoomLS)
    status = tc.configure_langsmith()
    assert status["enabled"] is True
    assert status["project"] == tc.DEFAULT_LANGSMITH_PROJECT


def test_as_int_and_finish_span_fallbacks():
    assert tc._as_int("nope") == 0
    assert tc.format_token_usage(None).startswith("Tokens:")

    class RunSetFail:
        extra = {"metadata": {}}
        outputs = None

        def set(self, **_k):
            raise RuntimeError("set fail")

    tc.begin_usage_span()
    usage = tc.finish_usage_span(
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "llm_calls": 1},
        agent_name="weather",
        run=RunSetFail(),
        reply="x",
        model="qwen",
    )
    assert usage["total_tokens"] == 2
    assert RunSetFail.extra["metadata"]["usage_metadata"]["total_tokens"] == 2

    class RunBothFail:
        extra = None

        def set(self, **_k):
            raise RuntimeError("set fail")

    tc.begin_usage_span()
    usage2 = tc.finish_usage_span(
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0},
        agent_name="travel",
        run=RunBothFail(),
    )
    assert usage2["llm_calls"] == 0


def test_configure_enabled_when_key_without_flag(monkeypatch, reset_langsmith_config):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-key-not-real")
    status = tc.configure_langsmith()
    assert status["enabled"] is True


def test_agent_trace_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "langsmith" and fromlist and "trace" in fromlist:
            raise ImportError("no trace")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with tc.agent_trace("x", inputs={}, tags=[], metadata={}) as run:
        assert run is None
