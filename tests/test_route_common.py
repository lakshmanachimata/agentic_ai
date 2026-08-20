from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

import route_common
from route_common import OsrmRoute


def test_format_duration_distance_time():
    assert route_common.format_duration(45) == "45 sec"
    assert "1 hr" in route_common.format_duration(3661)
    assert route_common.format_distance(250) == "250 m"
    assert route_common.format_distance(1500) == "1.5 km"
    assert route_common.format_time(datetime(2026, 1, 1, 9, 5)) == "09:05"


def test_parse_start_time_clock_and_iso():
    parsed = route_common.parse_start_time("9:30 AM")
    assert parsed.hour == 9
    assert parsed.minute == 30
    iso = route_common.parse_start_time("2026-08-21 14:15")
    assert iso == datetime(2026, 8, 21, 14, 15)
    empty = route_common.parse_start_time("")
    assert empty.hour == 8
    assert empty.minute == 0
    pm = route_common.parse_start_time("2 pm")
    assert pm.hour == 14


def test_polyline_and_length():
    assert route_common.polyline_from_osrm_route({}) == []
    poly = route_common.polyline_from_osrm_route(
        {"geometry": {"type": "LineString", "coordinates": [[72.8, 19.0], [78.4, 17.4]]}}
    )
    assert poly[0] == (19.0, 72.8)
    assert route_common.route_length_m([]) == 0.0
    assert route_common.route_length_m(poly) > 1000
    start = route_common.point_along_route(poly, 0)
    assert start == poly[0]
    assert route_common.point_along_route([], 10) is None
    samples = route_common.sample_middle_points(poly, max_points=4)
    assert samples


def test_town_in_label():
    assert route_common._town_in_label("Pune", "Pune, Maharashtra, India")
    assert not route_common._town_in_label("", "Pune")


def test_geocode_empty_and_success():
    assert "empty" in route_common.geocode(SimpleNamespace(), "  ")

    class Client:
        def get(self, *_a, **_k):
            resp = SimpleNamespace()
            resp.raise_for_status = lambda: None
            resp.json = lambda: [
                {"lat": "19.07", "lon": "72.87", "display_name": "Mumbai, India"}
            ]
            return resp

    hit = route_common.geocode(Client(), "Mumbai")
    assert hit["lat"] == pytest.approx(19.07)
    assert hit["label"].startswith("Mumbai")


def test_geocode_not_found_and_http_error():
    class Empty:
        def get(self, *_a, **_k):
            resp = SimpleNamespace()
            resp.raise_for_status = lambda: None
            resp.json = lambda: []
            return resp

    assert "Could not find" in route_common.geocode(Empty(), "Nowhereville")

    class Boom:
        def get(self, *_a, **_k):
            raise httpx.ConnectError("no network")

    assert "HTTP error" in route_common.geocode(Boom(), "Mumbai")


def test_fetch_osrm_route_unsupported_and_ok():
    assert "Unsupported" in route_common.fetch_osrm_route(
        SimpleNamespace(), "A", "B", "hovercraft"
    )

    class Client:
        def get(self, url, **kwargs):
            resp = SimpleNamespace()
            resp.raise_for_status = lambda: None
            if "nominatim" in url:
                q = kwargs["params"]["q"]
                resp.json = lambda: [
                    {"lat": "19.0", "lon": "72.8", "display_name": q}
                ]
            else:
                resp.json = lambda: {
                    "code": "Ok",
                    "routes": [
                        {
                            "duration": 3600,
                            "distance": 100000,
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[72.8, 19.0], [78.4, 17.4]],
                            },
                        }
                    ],
                }
            return resp

    route = route_common.fetch_osrm_route(Client(), "Mumbai", "Hyderabad", "drive")
    assert isinstance(route, OsrmRoute)
    assert route.profile == "driving"
    assert route.duration_s == 3600
    assert len(route.poly) == 2
