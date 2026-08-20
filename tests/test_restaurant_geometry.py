from __future__ import annotations

import json

import httpx
import pytest

import restaurant_agent as ra
from tests.fakes import FakeClient, FakeResponse, client_factory


POLY = [(19.07, 72.87), (18.52, 73.85), (17.38, 78.48)]


def test_haversine_and_zero_segment_projection():
    d = ra._haversine_m(19.07, 72.87, 18.52, 73.85)
    assert d > 1000
    t = ra._project_fraction_on_segment(19.07, 72.87, 19.07, 72.87, 19.07, 72.87)
    assert t == 0.0
    mid = ra._project_fraction_on_segment(18.8, 73.3, 19.07, 72.87, 18.52, 73.85)
    assert 0.0 <= mid <= 1.0


def test_progress_empty_single_and_tiny_segment():
    assert ra._progress_and_distance_to_polyline(1.0, 2.0, []) == (0.0, float("inf"))
    prog, dist = ra._progress_and_distance_to_polyline(19.07, 72.87, [(19.07, 72.87)])
    assert prog == 0.0
    assert dist == 0.0
    tiny = [(19.0, 72.0), (19.0, 72.000001)]
    prog2, dist2 = ra._progress_and_distance_to_polyline(19.0, 72.0, tiny)
    assert prog2 == 0.0
    assert dist2 >= 0.0
    prog3, dist3 = ra._progress_and_distance_to_polyline(18.52, 73.85, POLY)
    assert dist3 < 50


def test_polyline_from_osrm_variants():
    assert ra._polyline_from_osrm_route({}) == []
    assert ra._polyline_from_osrm_route({"geometry": {"type": "Point"}}) == []
    out = ra._polyline_from_osrm_route(
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [[72.8, 19.0], [1], [78.4, 17.4]],
            }
        }
    )
    assert out == [(19.0, 72.8), (17.4, 78.4)]


def test_sample_polyline_evenly_branches():
    assert ra._sample_polyline_evenly([]) == []
    assert ra._sample_polyline_evenly([(1.0, 2.0)]) == [(1.0, 2.0)]
    same = ra._sample_polyline_evenly([(1.0, 2.0), (1.0, 2.0)])
    assert same[0] == (1.0, 2.0)
    sampled = ra._sample_polyline_evenly(POLY, max_samples=4)
    assert len(sampled) >= 2


def test_route_length_and_point_along():
    assert ra._route_length_m([]) == 0.0
    assert ra._route_length_m([(1.0, 1.0)]) == 0.0
    assert ra._route_length_m(POLY) > 1000
    assert ra._point_along_route_at_distance([], 10) is None
    assert ra._point_along_route_at_distance([(1.0, 2.0)], 10) == (1.0, 2.0)
    start = ra._point_along_route_at_distance(POLY, 0)
    assert start == POLY[0]
    end = ra._point_along_route_at_distance(POLY, 10**12)
    assert end == POLY[-1]
    mid = ra._point_along_route_at_distance(POLY, 50_000)
    assert mid is not None


def test_sample_route_middle_interval():
    short = ra._sample_route_middle_interval(POLY, 0, 10, 4)
    assert len(short) == 1
    wide = ra._sample_route_middle_interval(POLY, 1_000, 200_000, 5)
    assert len(wide) >= 1
    assert ra._sample_route_middle_interval([], 0, 100, 3) == []


def test_reverse_place_label_paths():
    boom = FakeClient(get=lambda *_a, **_k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    assert ra._reverse_place_label(boom, 18.5, 73.8) == ""

    town = FakeClient(
        get=lambda *_a, **_k: FakeResponse({"address": {"town": "Pune"}, "display_name": "x"})
    )
    assert ra._reverse_place_label(town, 18.5, 73.8) == "Pune"

    display = FakeClient(
        get=lambda *_a, **_k: FakeResponse({"address": {}, "display_name": "Nashik, India"})
    )
    assert ra._reverse_place_label(display, 18.5, 73.8) == "Nashik"

    empty = FakeClient(get=lambda *_a, **_k: FakeResponse({"address": {}}))
    assert ra._reverse_place_label(empty, 18.5, 73.8) == ""


def test_fetch_route_polyline_error_and_success():
    client = FakeClient()
    poly, err, *_rest = ra._fetch_route_polyline(client, "A", "B", "teleport")
    assert poly is None
    assert "Unsupported" in err

    bad_o = FakeClient(get=lambda url, **k: FakeResponse([]))
    poly, err, *_ = ra._fetch_route_polyline(bad_o, "Nowhere", "Hyderabad", "drive")
    assert poly is None
    assert "Could not find" in err

    def dest_missing(url, **kwargs):
        q = kwargs["params"]["q"]
        if q == "Mumbai":
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": "Mumbai"}])
        return FakeResponse([])

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=dest_missing), "Mumbai", "Ghost", "driving"
    )
    assert poly is None
    assert "Could not find" in err

    def http_fail(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
        raise httpx.ConnectError("osrm down")

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=http_fail), "Mumbai", "Hyderabad", "driving"
    )
    assert "HTTP error" in err

    def bad_json(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
        return FakeResponse({}, json_error=True)

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=bad_json), "Mumbai", "Hyderabad", "driving"
    )
    assert "invalid data" in err

    def not_ok(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
        return FakeResponse({"code": "NoRoute", "message": "blocked"})

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=not_ok), "Mumbai", "Hyderabad", "driving"
    )
    assert "blocked" in err

    def missing_routes(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
        return FakeResponse({"code": "Ok", "routes": []})

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=missing_routes), "Mumbai", "Hyderabad", "driving"
    )
    assert "missing route" in err

    def short_geom(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
        return FakeResponse(
            {
                "code": "Ok",
                "routes": [
                    {
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[72.8, 19.0]],
                        }
                    }
                ],
            }
        )

    poly, err, *_ = ra._fetch_route_polyline(
        FakeClient(get=short_geom), "Mumbai", "Hyderabad", "driving"
    )
    assert "too short" in err

    def ok(url, **kwargs):
        if "nominatim" in url:
            q = kwargs["params"]["q"]
            return FakeResponse([{"lat": "19.0", "lon": "72.8", "display_name": q}])
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

    poly, err, o, d, profile, *_ = ra._fetch_route_polyline(
        FakeClient(get=ok), "Mumbai", "Hyderabad", "driving"
    )
    assert err == ""
    assert len(poly) == 2
    assert profile == "driving"
    assert o.startswith("Mumbai")


def test_post_overpass_retry_and_errors():
    calls = {"n": 0}

    def post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse({}, status_code=503)
        return FakeResponse({"elements": []}, status_code=200)

    resp = ra._post_overpass(FakeClient(post=post), "query")
    assert resp.status_code == 200
    assert calls["n"] == 2

    with pytest.raises(httpx.HTTPStatusError):
        ra._post_overpass(
            FakeClient(post=lambda *_a, **_k: FakeResponse({}, status_code=400)),
            "q",
        )

    last = ra._post_overpass(
        FakeClient(post=lambda *_a, **_k: FakeResponse({}, status_code=429)),
        "q",
    )
    assert last.status_code == 429


def test_poi_lines_from_elements():
    elements = [
        {"tags": {}},
        {"tags": {"amenity": "cafe"}, "lat": "bad", "lon": 1},
        {"tags": {"amenity": "cafe", "name": "NoCoords"}},
        {
            "tags": {"amenity": "restaurant", "name": "Good", "cuisine": "indian"},
            "center": {"lat": 18.52, "lon": 73.85},
        },
        {
            "tags": {"amenity": "cafe"},
            "lat": 18.53,
            "lon": 73.86,
        },
    ]
    lines = ra._poi_lines_from_elements(elements, 18.52, 73.85, cap=5)
    assert any("Good" in line for line in lines)
    assert any("no name" in line for line in lines)
