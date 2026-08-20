from __future__ import annotations

from restaurant_agent import _overpass_query, _sanitize_cuisine_token
from travel_agent import _normalize_profile
from weather_agent import _wttr_value, get_weather


def test_wttr_value():
    assert _wttr_value([{"value": "Sunny"}]) == "Sunny"
    assert _wttr_value("plain") == "plain"


def test_get_weather_empty_location():
    assert "empty location" in get_weather.invoke({"location": "  "})


def test_normalize_travel_profile():
    assert _normalize_profile("") == "driving"
    assert _normalize_profile("Car") == "driving"
    assert _normalize_profile("walk") == "walking"
    assert _normalize_profile("bicycle") == "cycling"
    assert _normalize_profile("teleport") == "teleport"


def test_sanitize_cuisine_and_overpass_query():
    assert _sanitize_cuisine_token("  ") is None
    assert _sanitize_cuisine_token("Italian!") == "italian"
    q = _overpass_query(19.07, 72.87, 800, "indian")
    assert '["cuisine"="indian"]' in q
    assert "restaurant" in q
    tight = _overpass_query(0, 0, 10, None)
    assert "around:200," in tight
    wide = _overpass_query(0, 0, 99999, None)
    assert "around:2500," in wide
