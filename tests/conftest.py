"""Shared pytest fixtures. Keep LangSmith quiet and reset token-rollup state."""

from __future__ import annotations

import os

# Must run before test modules import tracing_common / agents.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")
os.environ.setdefault("AGENTIC_AI_GUARDRAILS", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pytest

import tracing_common as tc


@pytest.fixture(autouse=True)
def _stub_langsmith_sdk_configure(monkeypatch):
    """Keep unit tests from exporting traces to smith.langchain.com."""
    try:
        import langsmith as ls

        monkeypatch.setattr(ls, "configure", lambda **_k: None)
    except Exception:
        pass
    tc._BANNER_PRINTED = False
    tc._USAGE_STACK.set(None)
    tc._LAST_USAGE.set(None)
    yield
    tc._USAGE_STACK.set(None)
    tc._LAST_USAGE.set(None)


@pytest.fixture(autouse=True)
def _no_network_sleep(monkeypatch):
    """Nominatim throttling sleeps 1s+; skip that in unit tests."""
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    import restaurant_agent
    import route_common

    monkeypatch.setattr(restaurant_agent.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(route_common.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture(autouse=True)
def _reset_trip_state():
    import trip_state as ts

    ts.clear_all_trips()
    yield
    ts.clear_all_trips()


@pytest.fixture
def reset_langsmith_config():
    """Allow tests to re-run configure_langsmith() against a fresh env."""
    prev = tc._STATUS
    tc._STATUS = None
    yield
    tc._STATUS = prev
