from __future__ import annotations

from types import SimpleNamespace

import run_agents


def test_parse_args_defaults_and_client():
    args = run_agents._parse_args([])
    assert args.client == "orchestrator"
    assert args.queries == []
    args = run_agents._parse_args(["-c", "weather", "-q", "Tokyo"])
    assert args.client == "weather"
    assert args.queries == ["Tokyo"]


def test_collect_queries_flags_and_positional():
    flags = SimpleNamespace(queries=["  one  ", "", "two"], positional_query=["ignored"])
    assert run_agents._collect_queries(flags) == ["one", "two"]

    pos = SimpleNamespace(queries=[], positional_query=["Weather", "in", "Rome"])
    assert run_agents._collect_queries(pos) == ["Weather in Rome"]

    empty = SimpleNamespace(queries=[], positional_query=[])
    assert run_agents._collect_queries(empty) == []


def test_main_one_shot_and_interactive(monkeypatch, capsys):
    monkeypatch.setattr(
        run_agents,
        "_CLIENTS",
        {
            "weather": ("Weather", "hint", lambda: "graph"),
        },
    )
    monkeypatch.setattr(run_agents, "invoke_agent", lambda *_a, **_k: "sunny")
    monkeypatch.setattr(run_agents, "print_tracing_banner", lambda: None)
    rc = run_agents.main(["-c", "weather", "-q", "Tokyo"])
    assert rc == 0
    assert "sunny" in capsys.readouterr().out

    called = []
    monkeypatch.setattr(run_agents, "run_interactive", lambda *a, **k: called.append(True))
    rc = run_agents.main(["-c", "weather"])
    assert rc == 0
    assert called

    monkeypatch.setattr(run_agents, "invoke_agent", lambda *_a, **_k: "one")
    rc = run_agents.main(["-c", "weather", "-q", "a", "-q", "b"])
    assert rc == 0
