from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import httpx

import weather_agent as wa
from tests.fakes import FakeResponse, wttr_payload


def test_fetch_wttr_http_and_json(monkeypatch):
    monkeypatch.setattr(
        wa.httpx,
        "get",
        lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    assert "HTTP error" in wa._fetch_wttr_json("Paris")

    monkeypatch.setattr(wa.httpx, "get", lambda *_a, **_k: FakeResponse({}, json_error=True))
    assert "invalid data" in wa._fetch_wttr_json("Paris")

    monkeypatch.setattr(wa.httpx, "get", lambda *_a, **_k: FakeResponse(wttr_payload()))
    data = wa._fetch_wttr_json("Paris")
    assert isinstance(data, dict)
    assert "current_condition" in data


def test_get_weather_success_and_parse_error(monkeypatch):
    monkeypatch.setattr(wa, "_fetch_wttr_json", lambda _loc: wttr_payload())
    text = wa.get_weather.invoke({"location": "Paris"})
    assert "Sunny" in text
    assert "Paris" in text
    assert "2026-08-21" in text

    monkeypatch.setattr(wa, "_fetch_wttr_json", lambda _loc: "Weather service HTTP error: x")
    assert "HTTP error" in wa.get_weather.invoke({"location": "Paris"})

    monkeypatch.setattr(wa, "_fetch_wttr_json", lambda _loc: {"nope": True})
    assert "Could not parse" in wa.get_weather.invoke({"location": "Paris"})

    payload = wttr_payload()
    payload["weather"][0].pop("date")
    monkeypatch.setattr(wa, "_fetch_wttr_json", lambda _loc: payload)
    no_date = wa.get_weather.invoke({"location": "Paris"})
    assert "Now:" in no_date


def test_weather_build_run_main(monkeypatch):
    monkeypatch.setattr(wa, "make_chat_ollama", lambda **_k: "llm")
    monkeypatch.setattr(wa, "create_agent", lambda *_a, **_k: "graph")
    assert wa.build_agent() == "graph"
    monkeypatch.setattr(wa, "invoke_agent", lambda *_a, **_k: "sunny")
    assert wa.run_query("g", "Paris") == "sunny"
    monkeypatch.setattr(wa, "build_agent", lambda **_k: "g")
    monkeypatch.setattr(wa.sys, "argv", ["weather_agent.py", "Paris"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))
    wa.main()
    assert printed
    monkeypatch.setattr(wa.sys, "argv", ["weather_agent.py"])
    called = []
    monkeypatch.setattr(wa, "run_interactive", lambda *a, **k: called.append(True))
    wa.main()
    assert called
