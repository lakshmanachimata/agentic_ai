from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx
import pytest

import route_common as rc
from route_common import OsrmRoute
from tests.fakes import FakeClient, FakeResponse, wttr_payload


def test_format_duration_hours_and_secs():
    assert rc.format_duration(3601) == "1 hr 1 sec"
    assert rc.format_duration(7200) == "2 hrs"
    assert rc.format_duration(0) == "0 sec"


def test_parse_start_time_fallbacks():
    iso_t = rc.parse_start_time("2026-08-21T14:15")
    assert iso_t == datetime(2026, 8, 21, 14, 15)
    with_sec = rc.parse_start_time("2026-08-21 14:15:30")
    assert with_sec.second == 30
    weird = rc.parse_start_time("not-a-time")
    assert weird.hour == 8
    noon = rc.parse_start_time("12 am")
    assert noon.hour == 0
    pm = rc.parse_start_time("1:05pm")
    assert pm.hour == 13
    assert pm.minute == 5


def test_geocode_json_and_missing_coords():
    bad_json = FakeClient(get=lambda *_a, **_k: FakeResponse({}, json_error=True))
    assert "invalid data" in rc.geocode(bad_json, "Paris")
    missing = FakeClient(
        get=lambda *_a, **_k: FakeResponse([{"display_name": "x"}])
    )
    assert "missing coordinates" in rc.geocode(missing, "Paris")


def test_polyline_skips_short_pairs():
    poly = rc.polyline_from_osrm_route(
        {"geometry": {"type": "LineString", "coordinates": [[1], [2, 3]]}}
    )
    assert poly == [(3.0, 2.0)]


def test_point_along_clamps_to_end():
    poly = [(0.0, 0.0), (0.01, 0.0)]
    end = rc.point_along_route(poly, 10**9)
    assert end == poly[-1]


def test_sample_middle_points_short_and_long():
    short = [(19.07, 72.87), (19.0701, 72.8701)]
    pts = rc.sample_middle_points(short, max_points=3)
    assert pts
    long = [(19.07, 72.87), (18.52, 73.85), (17.38, 78.48)]
    many = rc.sample_middle_points(long, max_points=5)
    assert len(many) >= 3


def test_reverse_town_name_paths():
    boom = FakeClient(get=lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("x")))
    assert rc.reverse_town_name(boom, 1, 2) == ""
    city = FakeClient(
        get=lambda *_a, **_k: FakeResponse({"address": {"city": "Lyon"}, "display_name": "x"})
    )
    assert rc.reverse_town_name(city, 1, 2) == "Lyon"
    display = FakeClient(
        get=lambda *_a, **_k: FakeResponse({"address": {}, "display_name": "Nantes, France"})
    )
    assert rc.reverse_town_name(display, 1, 2) == "Nantes"
    empty = FakeClient(get=lambda *_a, **_k: FakeResponse({"address": {}}))
    assert rc.reverse_town_name(empty, 1, 2) == ""


def test_weather_summary_at_time(monkeypatch):
    assert "No location" in rc.weather_summary_at_time("  ", datetime.now())

    monkeypatch.setattr(
        rc.httpx,
        "get",
        lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    assert "unavailable" in rc.weather_summary_at_time("Paris", datetime.now())

    monkeypatch.setattr(
        rc.httpx, "get", lambda *_a, **_k: FakeResponse({}, json_error=True)
    )
    assert "invalid data" in rc.weather_summary_at_time("Paris", datetime.now())

    monkeypatch.setattr(
        rc.httpx, "get", lambda *_a, **_k: FakeResponse({"weather": []})
    )
    assert "No forecast" in rc.weather_summary_at_time("Paris", datetime.now())

    monkeypatch.setattr(
        rc.httpx,
        "get",
        lambda *_a, **_k: FakeResponse(
            {
                "weather": [{"hourly": []}],
                "current_condition": [
                    {"weatherDesc": [{"value": "Rain"}], "temp_C": "12"}
                ],
            }
        ),
    )
    current = rc.weather_summary_at_time("Paris", datetime(2026, 1, 1, 9, 0))
    assert "Rain" in current

    monkeypatch.setattr(rc.httpx, "get", lambda *_a, **_k: FakeResponse(wttr_payload()))
    sunny = rc.weather_summary_at_time("Paris", datetime(2026, 1, 1, 12, 0))
    assert "Sunny" in sunny or "Partly" in sunny

    monkeypatch.setattr(
        rc.httpx,
        "get",
        lambda *_a, **_k: FakeResponse(
            {
                "weather": [
                    {
                        "hourly": [
                            {"time": "not-int", "tempC": "1", "chanceofrain": "0"}
                        ]
                    }
                ]
            }
        ),
    )
    fallback = rc.weather_summary_at_time("Paris", datetime(2026, 1, 1, 8, 0))
    assert "°C" in fallback

    monkeypatch.setattr(rc.httpx, "get", lambda *_a, **_k: FakeResponse({"weather": [{}]}))
    assert "Could not parse" in rc.weather_summary_at_time("Paris", datetime.now())


def test_fetch_osrm_route_error_paths():
    client = FakeClient(get=lambda *_a, **_k: FakeResponse([]))
    assert "Could not find" in rc.fetch_osrm_route(client, "A", "B", "drive")

    def dest_fail(url, **kwargs):
        q = kwargs.get("params", {}).get("q", "")
        if q == "Mumbai":
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": "Mumbai"}])
        if "nominatim" in url:
            return FakeResponse([])
        return FakeResponse({"code": "Ok", "routes": []})

    assert "Could not find" in rc.fetch_osrm_route(
        FakeClient(get=dest_fail), "Mumbai", "Ghost", "car"
    )

    def osrm_http(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": q}])
        raise httpx.ConnectError("down")

    assert "HTTP error" in rc.fetch_osrm_route(
        FakeClient(get=osrm_http), "A", "B", "walking"
    )

    def osrm_json(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": q}])
        return FakeResponse({}, json_error=True)

    assert "invalid data" in rc.fetch_osrm_route(
        FakeClient(get=osrm_json), "A", "B", "bike"
    )

    def not_ok(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": q}])
        return FakeResponse({"code": "NoRoute"})

    assert "Routing failed" in rc.fetch_osrm_route(
        FakeClient(get=not_ok), "A", "B", "driving"
    )

    def short(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": q}])
        return FakeResponse(
            {
                "code": "Ok",
                "routes": [
                    {
                        "duration": 1,
                        "distance": 1,
                        "geometry": {"type": "LineString", "coordinates": [[1, 2]]},
                    }
                ],
            }
        )

    assert "too short" in rc.fetch_osrm_route(FakeClient(get=short), "A", "B", "driving")

    def parse_fail(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19", "lon": "72", "display_name": q}])
        return FakeResponse(
            {
                "code": "Ok",
                "routes": [
                    {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[72.8, 19.0], [78.4, 17.4]],
                        }
                    }
                ],
            }
        )

    assert "Could not parse" in rc.fetch_osrm_route(
        FakeClient(get=parse_fail), "A", "B", "driving"
    )


def test_discover_intermediate_stops(monkeypatch):
    route = OsrmRoute(
        poly=[(19.07, 72.87), (18.52, 73.85), (17.38, 78.48)],
        duration_s=31620,
        distance_m=704000,
        origin_label="Mumbai, Maharashtra, India",
        dest_label="Hyderabad, Telangana, India",
        origin_lat=19.07,
        origin_lon=72.87,
        dest_lat=17.38,
        dest_lon=78.48,
        profile="driving",
    )
    names = iter(["Mumbai", "", "Pune", "Pune", "Hyderabad", "Solapur", "Nanded"])

    def fake_reverse(_c, lat, lon):
        try:
            return next(names)
        except StopIteration:
            return "Nagpur"

    monkeypatch.setattr(rc, "reverse_town_name", fake_reverse)
    stops = rc.discover_intermediate_stops(
        FakeClient(), route, datetime(2026, 8, 21, 8, 0), max_towns=2
    )
    assert any(s.town == "Pune" for s in stops)
    assert all(s.town not in ("Mumbai", "Hyderabad") for s in stops)
    assert len(stops) <= 2


def test_town_in_label_false_when_missing():
    assert rc._town_in_label("Pune", "") is False
