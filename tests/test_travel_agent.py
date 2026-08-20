from __future__ import annotations

import json
from datetime import datetime

import httpx

import travel_agent as ta
from route_common import OsrmRoute, RouteStop
from tests.fakes import FakeClient, FakeResponse, client_factory


def test_get_travel_time_paths(monkeypatch):
    monkeypatch.setattr(ta.httpx, "Client", client_factory(FakeClient()))
    assert "Unsupported" in ta.get_travel_time.invoke(
        {"origin": "A", "destination": "B", "mode": "teleport"}
    )

    monkeypatch.setattr(ta, "_geocode", lambda *_a, **_k: "bad origin")
    assert ta.get_travel_time.invoke({"origin": "A", "destination": "B"}) == "bad origin"

    def geo(_c, place):
        if place == "A":
            return {"lat": 19.0, "lon": 72.8, "label": "A-lab"}
        return "bad dest"

    monkeypatch.setattr(ta, "_geocode", geo)
    assert ta.get_travel_time.invoke({"origin": "A", "destination": "B"}) == "bad dest"

    monkeypatch.setattr(
        ta,
        "_geocode",
        lambda _c, place: {"lat": 19.0, "lon": 72.8, "label": place},
    )

    def http_fail(*_a, **_k):
        raise httpx.ConnectError("down")

    inner = FakeClient(get=http_fail)
    monkeypatch.setattr(ta.httpx, "Client", client_factory(inner))
    assert "HTTP error" in ta.get_travel_time.invoke({"origin": "A", "destination": "B"})

    inner = FakeClient(get=lambda *_a, **_k: FakeResponse({}, json_error=True))
    monkeypatch.setattr(ta.httpx, "Client", client_factory(inner))
    assert "invalid data" in ta.get_travel_time.invoke({"origin": "A", "destination": "B"})

    inner = FakeClient(get=lambda *_a, **_k: FakeResponse({"code": "NoRoute", "message": "x"}))
    monkeypatch.setattr(ta.httpx, "Client", client_factory(inner))
    assert "Routing failed" in ta.get_travel_time.invoke({"origin": "A", "destination": "B"})

    inner = FakeClient(get=lambda *_a, **_k: FakeResponse({"code": "Ok", "routes": []}))
    monkeypatch.setattr(ta.httpx, "Client", client_factory(inner))
    assert "Could not parse" in ta.get_travel_time.invoke({"origin": "A", "destination": "B"})

    inner = FakeClient(
        get=lambda *_a, **_k: FakeResponse(
            {"code": "Ok", "routes": [{"duration": 3600, "distance": 100000}]}
        )
    )
    monkeypatch.setattr(ta.httpx, "Client", client_factory(inner))
    ok = ta.get_travel_time.invoke(
        {"origin": "Paris", "destination": "Lyon", "mode": "walk"}
    )
    assert "Estimated time" in ok
    assert "on foot" in ok

    bike = ta.get_travel_time.invoke(
        {"origin": "Paris", "destination": "Lyon", "mode": "cycling"}
    )
    assert "bicycle" in bike


def test_get_route_stops_with_weather(monkeypatch):
    monkeypatch.setattr(ta.httpx, "Client", client_factory(FakeClient()))
    assert "Unsupported" in ta.get_route_stops_with_weather.invoke(
        {"origin": "A", "destination": "B", "mode": "teleport"}
    )

    monkeypatch.setattr(ta, "fetch_osrm_route", lambda *_a, **_k: "nope")
    assert (
        ta.get_route_stops_with_weather.invoke({"origin": "A", "destination": "B"})
        == "nope"
    )

    route = OsrmRoute(
        poly=[(48.8, 2.3), (45.7, 4.8)],
        duration_s=14400,
        distance_m=460000,
        origin_label="Paris, France",
        dest_label="Lyon, France",
        origin_lat=48.8,
        origin_lon=2.3,
        dest_lat=45.7,
        dest_lon=4.8,
        profile="driving",
    )
    monkeypatch.setattr(ta, "fetch_osrm_route", lambda *_a, **_k: route)
    monkeypatch.setattr(ta, "discover_intermediate_stops", lambda *_a, **_k: [])
    monkeypatch.setattr(ta, "weather_summary_at_time", lambda *_a, **_k: "Sunny, 20°C")
    empty = ta.get_route_stops_with_weather.invoke(
        {"origin": "Paris", "destination": "Lyon", "start_time": "9 AM"}
    )
    assert "No distinct intermediate" in empty
    assert "Sunny" in empty

    stops = [
        RouteStop(
            town="Beaune",
            lat=47.0,
            lon=4.8,
            distance_km=200,
            duration_from_start_s=7200,
            arrival=datetime(2026, 8, 21, 11, 0),
        )
    ]
    monkeypatch.setattr(ta, "discover_intermediate_stops", lambda *_a, **_k: stops)
    listed = ta.get_route_stops_with_weather.invoke(
        {"origin": "Paris", "destination": "Lyon", "mode": "driving"}
    )
    assert "Beaune" in listed
    assert "Weather then" in listed


def test_travel_geocode_wrapper(monkeypatch):
    monkeypatch.setattr(ta, "geocode", lambda *_a, **_k: {"lat": 1.0})
    assert ta._geocode(FakeClient(), "X")["lat"] == 1.0


def test_travel_build_run_main(monkeypatch):
    monkeypatch.setattr(ta, "make_chat_ollama", lambda **_k: "llm")
    monkeypatch.setattr(ta, "create_agent", lambda *_a, **_k: "graph")
    assert ta.build_agent() == "graph"
    monkeypatch.setattr(ta, "invoke_agent", lambda *_a, **_k: "ok")
    assert ta.run_query("g", "q") == "ok"
    monkeypatch.setattr(ta, "build_agent", lambda **_k: "g")
    monkeypatch.setattr(ta.sys, "argv", ["travel_agent.py", "drive"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))
    ta.main()
    assert printed
    monkeypatch.setattr(ta.sys, "argv", ["travel_agent.py"])
    called = []
    monkeypatch.setattr(ta, "run_interactive", lambda *a, **k: called.append(True))
    ta.main()
    assert called
