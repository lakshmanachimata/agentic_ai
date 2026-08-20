from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import httpx

import restaurant_agent as ra
from route_common import OsrmRoute, RouteStop
from tests.fakes import FakeClient, FakeResponse, client_factory


POLY = [(19.07, 72.87), (18.52, 73.85), (17.38, 78.48)]


def _patch_client(monkeypatch, inner: FakeClient | None = None):
    monkeypatch.setattr(ra.httpx, "Client", client_factory(inner or FakeClient()))


def test_find_places_to_eat_geo_error(monkeypatch):
    monkeypatch.setattr(ra, "_geocode", lambda *_a, **_k: "Could not find a location")
    _patch_client(monkeypatch)
    out = ra.find_places_to_eat.invoke({"area": "Nowhere"})
    assert "Could not find" in out


def test_find_places_to_eat_overpass_and_rows(monkeypatch):
    monkeypatch.setattr(
        ra,
        "_geocode",
        lambda *_a, **_k: {"lat": 18.52, "lon": 73.85, "label": "Pune, India"},
    )
    _patch_client(monkeypatch)

    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, status_code=429),
    )
    assert "rate limit" in ra.find_places_to_eat.invoke({"area": "Pune"})

    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, status_code=502),
    )
    assert "busy" in ra.find_places_to_eat.invoke({"area": "Pune"})

    def boom(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(ra, "_post_overpass", boom)
    assert "HTTP error" in ra.find_places_to_eat.invoke({"area": "Pune"})

    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, json_error=True),
    )
    assert "invalid JSON" in ra.find_places_to_eat.invoke({"area": "Pune"})

    payload = {
        "elements": [
            {"id": 1, "type": "node", "tags": {}},
            {
                "id": 2,
                "type": "node",
                "lat": 18.521,
                "lon": 73.851,
                "tags": {
                    "name": "Cafe Good",
                    "amenity": "cafe",
                    "cuisine": "indian",
                    "opening_hours": "9-21",
                },
            },
            {
                "id": 3,
                "type": "way",
                "center": {"lat": 18.522, "lon": 73.852},
                "tags": {"amenity": "restaurant"},
            },
            {"id": 4, "type": "node", "tags": {"amenity": "cafe"}},
            {
                "id": 5,
                "type": "node",
                "lat": "x",
                "lon": 1,
                "tags": {"amenity": "cafe", "name": "Bad"},
            },
        ]
    }
    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse(payload))
    listed = ra.find_places_to_eat.invoke(
        {"area": "Pune", "cuisine": "Indian", "max_results": 5}
    )
    assert "Cafe Good" in listed
    assert "cuisine=indian" in listed
    assert "Pune" in listed

    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse({"elements": []}))
    empty = ra.find_places_to_eat.invoke({"area": "Pune", "cuisine": "thai"})
    assert "No OSM-tagged" in empty
    assert "thai" in empty

    empty2 = ra.find_places_to_eat.invoke({"area": "Pune"})
    assert "larger radius" in empty2


def test_find_places_to_eat_along_route(monkeypatch):
    _patch_client(monkeypatch)
    monkeypatch.setattr(
        ra,
        "_fetch_route_polyline",
        lambda *_a, **_k: (
            None,
            "no route",
            None,
            None,
            "driving",
            None,
            None,
            None,
            None,
        ),
    )
    assert ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    ) == "no route"

    monkeypatch.setattr(
        ra,
        "_fetch_route_polyline",
        lambda *_a, **_k: (
            POLY,
            "",
            "Mumbai",
            "Hyderabad",
            "driving",
            19.07,
            72.87,
            17.38,
            78.48,
        ),
    )
    monkeypatch.setattr(ra, "_reverse_place_label", lambda *_a, **_k: "Pune")
    monkeypatch.setattr(ra, "_sample_route_middle_interval", lambda *_a, **_k: [])
    assert "No intermediate" in ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )

    monkeypatch.setattr(
        ra,
        "_sample_route_middle_interval",
        lambda *_a, **_k: [(18.52, 73.85)],
    )

    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, status_code=429),
    )
    assert "rate limit" in ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )
    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, status_code=503),
    )
    assert "busy" in ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )
    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, json_error=True),
    )
    assert "invalid JSON" in ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )

    def status_err(*_a, **_k):
        req = httpx.Request("POST", "https://overpass")
        raise httpx.HTTPStatusError(
            "bad",
            request=req,
            response=httpx.Response(418, request=req),
        )

    monkeypatch.setattr(ra, "_post_overpass", status_err)
    assert "HTTP error" in ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )

    payload = {
        "elements": [
            {"id": None, "type": "node"},
            {"id": 9, "type": "other"},
            {
                "id": 1,
                "type": "node",
                "lat": 18.52,
                "lon": 73.85,
                "tags": {
                    "name": "Mid Cafe",
                    "amenity": "cafe",
                    "cuisine": "indian",
                    "opening_hours": "10-22",
                },
            },
            {
                "id": 2,
                "type": "node",
                "lat": 19.07,
                "lon": 72.87,
                "tags": {"name": "At Origin", "amenity": "cafe"},
            },
            {
                "id": 3,
                "type": "way",
                "center": {"lat": 18.53, "lon": 73.86},
                "tags": {"amenity": "restaurant"},
            },
            {
                "id": 4,
                "type": "node",
                "lat": "x",
                "lon": 1,
                "tags": {"amenity": "cafe"},
            },
            {
                "id": 5,
                "type": "node",
                "tags": {"amenity": "cafe", "name": "nocoords"},
            },
        ]
    }
    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse(payload))
    listed = ra.find_places_to_eat_along_route.invoke(
        {
            "origin": "Mumbai",
            "destination": "Hyderabad",
            "cuisine": "indian",
            "exclude_endpoints_meters": 800,
        }
    )
    assert "Mid Cafe" in listed
    assert "At Origin" not in listed
    assert "In-between stops" in listed

    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse({"elements": []}))
    empty = ra.find_places_to_eat_along_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad", "cuisine": "thai"}
    )
    assert "No OSM restaurants" in empty
    assert "thai" in empty

    short_poly = [(19.07, 72.87), (19.08, 72.88)]
    monkeypatch.setattr(
        ra,
        "_fetch_route_polyline",
        lambda *_a, **_k: (
            short_poly,
            "",
            "A",
            "B",
            "driving",
            19.07,
            72.87,
            19.08,
            72.88,
        ),
    )
    monkeypatch.setattr(ra, "_reverse_place_label", lambda *_a, **_k: "")
    short = ra.find_places_to_eat_along_route.invoke(
        {"origin": "A", "destination": "B", "exclude_endpoints_meters": 5000}
    )
    assert "No OSM restaurants" in short or "short trip" in short or "In-between" in short


def test_find_restaurants_at_towns_on_route(monkeypatch):
    _patch_client(monkeypatch)
    monkeypatch.setattr(ra, "fetch_osrm_route", lambda *_a, **_k: "Routing failed")
    assert (
        ra.find_restaurants_at_towns_on_route.invoke(
            {"origin": "Mumbai", "destination": "Hyderabad"}
        )
        == "Routing failed"
    )

    route = OsrmRoute(
        poly=POLY,
        duration_s=31620,
        distance_m=704000,
        origin_label="Mumbai, India",
        dest_label="Hyderabad, India",
        origin_lat=19.07,
        origin_lon=72.87,
        dest_lat=17.38,
        dest_lon=78.48,
        profile="driving",
    )
    monkeypatch.setattr(ra, "fetch_osrm_route", lambda *_a, **_k: route)
    monkeypatch.setattr(ra, "discover_intermediate_stops", lambda *_a, **_k: [])
    assert "No distinct intermediate" in ra.find_restaurants_at_towns_on_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )

    stops = [
        RouteStop(
            town="Pune",
            lat=18.52,
            lon=73.85,
            distance_km=150,
            duration_from_start_s=7200,
            arrival=datetime(2026, 8, 21, 10, 0),
        )
    ]
    monkeypatch.setattr(ra, "discover_intermediate_stops", lambda *_a, **_k: stops)

    monkeypatch.setattr(
        ra,
        "_post_overpass",
        lambda *_a, **_k: FakeResponse({}, status_code=503),
    )
    busy = ra.find_restaurants_at_towns_on_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad", "start_time": "9 AM", "cuisine": "indian"}
    )
    assert "Overpass busy" in busy

    def boom(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(ra, "_post_overpass", boom)
    err = ra.find_restaurants_at_towns_on_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )
    assert "search error" in err

    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse({"elements": []}))
    none = ra.find_restaurants_at_towns_on_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )
    assert "no OSM restaurants" in none

    payload = {
        "elements": [
            {
                "lat": 18.52,
                "lon": 73.85,
                "tags": {"name": "Town Cafe", "amenity": "cafe", "cuisine": "indian"},
            }
        ]
    }
    monkeypatch.setattr(ra, "_post_overpass", lambda *_a, **_k: FakeResponse(payload))
    listed = ra.find_restaurants_at_towns_on_route.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad"}
    )
    assert "Town Cafe" in listed
    assert "Pune" in listed


def test_restaurant_build_run_main(monkeypatch):
    monkeypatch.setattr(ra, "make_chat_ollama", lambda **_k: "llm")
    monkeypatch.setattr(ra, "create_agent", lambda *_a, **_k: "graph")
    assert ra.build_agent(model="m") == "graph"
    monkeypatch.setattr(ra, "invoke_agent", lambda *_a, **_k: "reply")
    assert ra.run_query("graph", "eat") == "reply"

    monkeypatch.setattr(ra, "build_agent", lambda **_k: "graph")
    monkeypatch.setattr(ra.sys, "argv", ["restaurant_agent.py", "food", "near", "Pune"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))
    ra.main()
    assert printed

    monkeypatch.setattr(ra.sys, "argv", ["restaurant_agent.py"])
    called = []
    monkeypatch.setattr(ra, "run_interactive", lambda *a, **k: called.append(k))
    ra.main()
    assert called
