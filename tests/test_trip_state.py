from __future__ import annotations

import trip_state as ts
from trip_state import compose_specialist_query, fill_route_args, harvest_travel_metrics


def test_update_trip_and_prompt_block():
    ts.update_trip(
        origin="Mumbai",
        destination="Hyderabad",
        start_time="tomorrow 10:00 AM",
        mode="driving",
        duration_s=31620,
        distance_m=704000,
        origin_label="Mumbai, Maharashtra",
        dest_label="Hyderabad, Telangana",
    )
    trip = ts.get_trip()
    assert trip.has_route()
    block = trip.as_prompt_block()
    assert "origin: Mumbai" in block
    assert "duration_s:" in block
    assert "start_time: tomorrow 10:00 AM" in block
    snap = trip.as_dict()
    assert snap["origin"] == "Mumbai"
    traced = trip.for_trace()
    assert traced["origin"] == "Mumbai"
    assert traced["duration_s"] == 31620
    assert "duration_human" in traced
    assert "distance_human" in traced
    scoped = ts.begin_trip_scope("sess-trace")
    try:
        ts.update_trip(origin="Mumbai", destination="Hyderabad")
        payload = ts.snapshot_for_trace()
        assert payload["trip_id"] == "sess-trace"
        assert payload["origin"] == "Mumbai"
    finally:
        ts.reset_trip_scope(scoped)
    assert ts.TripState().for_trace() == {}


def test_route_change_clears_duration():
    ts.update_trip(origin="A", destination="B", duration_s=3600)
    ts.update_trip(destination="C")
    assert ts.get_trip().duration_s is None
    assert ts.get_trip().destination == "C"


def test_fill_route_args_prefers_explicit():
    ts.update_trip(origin="Mumbai", destination="Pune", mode="walking", start_time="9 AM")
    o, d, m, s = fill_route_args("", "", "", "")
    assert o == "Mumbai"
    assert d == "Pune"
    assert m == "walking"
    assert s == "9 AM"
    o2, d2, m2, s2 = fill_route_args("Delhi", "Agra", "driving", "8 AM")
    assert (o2, d2, m2, s2) == ("Delhi", "Agra", "driving", "8 AM")


def test_harvest_travel_metrics():
    harvest_travel_metrics("Estimated time: 8 hrs 47 min\nDistance: 704.0 km")
    assert ts.get_trip().duration_s == 8 * 3600 + 47 * 60
    ts.update_trip(duration_s=10)
    harvest_travel_metrics("Estimated time: 1 hr")
    assert ts.get_trip().duration_s == 10


def test_compose_specialist_query_roles():
    assert compose_specialist_query("hello", specialist="travel") == "hello"
    ts.update_trip(origin="Mumbai", destination="Hyderabad", start_time="tomorrow 10 AM")
    cal = compose_specialist_query("add to calendar", specialist="calendar")
    assert "Typed trip state" in cal
    assert "create_travel_calendar" in cal
    assert "Mumbai" in cal
    travel = compose_specialist_query("plan", specialist="travel")
    assert "origin/destination" in travel
    food = compose_specialist_query("", specialist="restaurants")
    assert "food along this trip" in food


def test_clear_trip_and_scope():
    token = ts.begin_trip_scope("sess-1")
    ts.update_trip(origin="Rome")
    assert ts.get_trip().origin == "Rome"
    ts.clear_trip("sess-1")
    assert ts.get_trip().origin == ""
    ts.reset_trip_scope(token)
    assert ts.current_trip_id() is None
    assert ts.snapshot_for_trace() == {}
    empty_scope = ts.begin_trip_scope("empty-sess")
    try:
        assert ts.snapshot_for_trace() == {}
    finally:
        ts.reset_trip_scope(empty_scope)


def test_prompt_block_labels_without_raw_places():
    ts.update_trip(
        origin_label="Milan, Italy",
        dest_label="Rome, Italy",
        mode="cycling",
        duration_s=14400,
        distance_m=580000,
    )
    trip = ts.get_trip()
    assert trip.has_route()
    block = trip.as_prompt_block()
    assert "origin_label: Milan, Italy" in block
    assert "dest_label: Rome, Italy" in block
    assert "mode: cycling" in block
    assert "start_time" not in block
    assert "origin:" not in block
    assert ts.TripState().as_prompt_block() == ""
    o, d, m, s = fill_route_args("", "", "", "")
    assert o == "Milan, Italy"
    assert d == "Rome, Italy"
    assert m == "cycling"
    assert s == ""


def test_snapshot_for_trace_without_scope_omits_trip_id():
    ts.update_trip(origin="Paris", destination="Lyon")
    payload = ts.snapshot_for_trace()
    assert payload["origin"] == "Paris"
    assert "trip_id" not in payload


def test_update_trip_skips_empty_unknown_and_none():
    ts.update_trip(origin="A", destination="B", duration_s=3600, start_time="9 AM")
    ts.update_trip(origin=None, destination="  ", mode=None, not_a_field="x")
    trip = ts.get_trip()
    assert trip.origin == "A"
    assert trip.destination == "B"
    assert trip.duration_s == 3600
    ts.update_trip(start_time=None, origin_label="  ", dest_label=None)
    assert ts.get_trip().start_time == "9 AM"
    assert ts.get_trip().origin_label == ""
    ts.update_trip(origin="a")  # case-only change does not clear duration
    assert ts.get_trip().duration_s == 3600
    ts.update_trip(destination="C", duration_s=7200)
    assert ts.get_trip().duration_s == 7200
    ts.clear_trip()
    assert ts.get_trip().origin == ""


def test_for_trace_skips_non_positive_metrics():
    ts.update_trip(origin="X", destination="Y", duration_s=0, distance_m=-1)
    traced = ts.get_trip().for_trace()
    assert "duration_s" not in traced
    assert "distance_m" not in traced
    assert traced["origin"] == "X"


def test_harvest_travel_metrics_variants():
    assert harvest_travel_metrics("no duration here").duration_s is None
    assert harvest_travel_metrics("").duration_s is None
    harvest_travel_metrics("Total: 45 sec")
    assert ts.get_trip().duration_s == 45
    ts.clear_trip()
    harvest_travel_metrics("Estimated time: 2 hrs")
    assert ts.get_trip().duration_s == 7200
    ts.clear_trip()
    harvest_travel_metrics("Total: 0 min")
    assert ts.get_trip().duration_s is None


def test_compose_specialist_query_other_roles():
    ts.update_trip(origin="Mumbai", destination="Pune")
    weather = compose_specialist_query("rain?", specialist="weather")
    assert "Typed trip state" in weather
    assert "create_travel_calendar" not in weather
    food = compose_specialist_query("lunch spots", specialist="restaurants")
    assert "lunch spots" in food
    assert "food along this trip" in food
