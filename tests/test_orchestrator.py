from __future__ import annotations

import orchestrator_agent as oa


def test_orchestrator_specialist_tools(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(oa, "build_weather_agent", lambda **_k: "wg")
    monkeypatch.setattr(oa, "build_travel_agent", lambda **_k: "tg")
    monkeypatch.setattr(oa, "build_restaurant_agent", lambda **_k: "rg")
    monkeypatch.setattr(oa, "build_calendar_agent", lambda **_k: "cg")
    monkeypatch.setattr(oa, "make_chat_ollama", lambda **_k: "llm")
    monkeypatch.setattr(oa, "run_weather_query", lambda g, q: f"w:{g}:{q}")
    monkeypatch.setattr(oa, "run_travel_query", lambda g, q: f"t:{g}:{q}")
    monkeypatch.setattr(oa, "run_restaurant_query", lambda g, q: f"r:{g}:{q}")
    monkeypatch.setattr(oa, "run_calendar_query", lambda g, q: f"c:{g}:{q}")

    def fake_create(_llm, tools, **_k):
        captured["tools"] = tools
        return "orch-graph"

    monkeypatch.setattr(oa, "create_agent", fake_create)
    assert oa.build_agent(model="mistral") == "orch-graph"
    by_name = {t.name: t for t in captured["tools"]}
    assert by_name["ask_weather_specialist"].invoke({"query": " Tokyo "}) == "w:wg:Tokyo"
    assert by_name["ask_travel_specialist"].invoke({"query": " drive "}) == "t:tg:drive"
    assert by_name["ask_restaurant_specialist"].invoke({"query": " eat "}) == "r:rg:eat"
    assert by_name["ask_calendar_specialist"].invoke({"query": " add "}) == "c:cg:add"


def test_orchestrator_run_and_main(monkeypatch):
    monkeypatch.setattr(oa, "invoke_agent", lambda *_a, **_k: "ok")
    assert oa.run_query("g", "q") == "ok"
    monkeypatch.setattr(oa, "build_agent", lambda **_k: "g")
    monkeypatch.setattr(oa.sys, "argv", ["orchestrator_agent.py", "hello"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))
    oa.main()
    assert printed
    monkeypatch.setattr(oa.sys, "argv", ["orchestrator_agent.py"])
    called = []
    monkeypatch.setattr(oa, "run_interactive", lambda *a, **k: called.append(True))
    oa.main()
    assert called
