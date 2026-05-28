"""Shared OSRM route geometry, intermediate towns, start-time parsing, and wttr.in weather."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
OSRM_URL = "https://router.project-osrm.org/route/v1"
USER_AGENT = "agentic_ai-route-common/1.0 (local study project)"
WTTR_USER_AGENT = "curl/8.0 (agentic-ai-route; +https://github.com/chubin/wttr.in)"


@dataclass
class RouteStop:
    town: str
    lat: float
    lon: float
    distance_km: float
    duration_from_start_s: float
    arrival: datetime


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _seg_len_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return _haversine_m(a[0], a[1], b[0], b[1])


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{total} sec"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hr" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} min")
    elif hours and secs:
        parts.append(f"{secs} sec")
    return " ".join(parts) if parts else f"{secs} sec"


def format_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{int(round(meters))} m"


def format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def parse_start_time(start_time: str) -> datetime:
    """Parse departure time; default today at 08:00 if empty."""
    raw = (start_time or "").strip()
    base = datetime.now().replace(second=0, microsecond=0)
    if not raw:
        return base.replace(hour=8, minute=0)

    lowered = raw.lower()
    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%H:%M",
        "%I:%M %p",
        "%I %p",
    ):
        try:
            parsed = datetime.strptime(raw if "%p" not in fmt else raw.title(), fmt)
            if fmt in ("%H:%M", "%I:%M %p", "%I %p"):
                return base.replace(hour=parsed.hour, minute=parsed.minute)
            return parsed
        except ValueError:
            continue

    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", lowered)
    if m:
        h = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and h < 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
        return base.replace(hour=h, minute=minute)

    return base.replace(hour=8, minute=0)


def geocode(client: httpx.Client, place: str) -> dict[str, Any] | str:
    place = place.strip()
    if not place:
        return "Error: empty place name."
    try:
        resp = client.get(
            NOMINATIM_SEARCH,
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )
        resp.raise_for_status()
        results = resp.json()
    except httpx.HTTPError as e:
        return f"Geocoding HTTP error for '{place}': {e}"
    except json.JSONDecodeError:
        return f"Geocoding returned invalid data for '{place}'."
    if not results:
        return f"Could not find a location for '{place}'."
    hit = results[0]
    try:
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "label": hit.get("display_name", place),
        }
    except (KeyError, TypeError, ValueError):
        return f"Geocoding response missing coordinates for '{place}'."


def polyline_from_osrm_route(route: dict[str, Any]) -> list[tuple[float, float]]:
    geom = route.get("geometry")
    if not isinstance(geom, dict) or geom.get("type") != "LineString":
        return []
    out: list[tuple[float, float]] = []
    for pair in geom.get("coordinates") or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            out.append((float(pair[1]), float(pair[0])))
    return out


def route_length_m(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return sum(_seg_len_m(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def point_along_route(poly: list[tuple[float, float]], distance_m: float) -> tuple[float, float] | None:
    if not poly:
        return None
    target = max(0.0, distance_m)
    cum = 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        seg = _seg_len_m(a, b)
        if cum + seg >= target:
            t = max(0.0, min(1.0, (target - cum) / (seg if seg > 1e-6 else 1.0)))
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        cum += seg
    return poly[-1]


def sample_middle_points(poly: list[tuple[float, float]], max_points: int = 7) -> list[tuple[float, float, float]]:
    """Return (lat, lon, distance_m along route) for samples in the middle 76% of the path."""
    total = route_length_m(poly)
    if total < 500:
        mid = total / 2
        pt = point_along_route(poly, mid)
        return [(pt[0], pt[1], mid)] if pt else []

    lo, hi = total * 0.12, total * 0.88
    n = max(3, min(max_points, 8))
    out: list[tuple[float, float, float]] = []
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.5
        d = lo + frac * (hi - lo)
        pt = point_along_route(poly, d)
        if pt:
            out.append((pt[0], pt[1], d))
    return out


def reverse_town_name(client: httpx.Client, lat: float, lon: float) -> str:
    try:
        resp = client.get(
            NOMINATIM_REVERSE,
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError, TypeError):
        return ""
    addr = data.get("address") or {}
    if isinstance(addr, dict):
        for key in ("city", "town", "village", "municipality", "county", "state_district"):
            v = addr.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:80]
    dn = data.get("display_name")
    if isinstance(dn, str) and dn.strip():
        return dn.split(",")[0].strip()[:80]
    return ""


def _wttr_value(obj: Any) -> str:
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "value" in obj[0]:
        return str(obj[0]["value"])
    return str(obj)


def weather_summary_at_time(location: str, when: datetime) -> str:
    """Forecast snippet from wttr.in for the given local time (same-day hourly)."""
    loc = location.strip()
    if not loc:
        return "No location for weather."
    url = f"https://wttr.in/{quote(loc, safe='')}?format=j1"
    try:
        resp = httpx.get(url, headers={"User-Agent": WTTR_USER_AGENT}, timeout=45.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return f"Weather unavailable: {e}"
    except json.JSONDecodeError:
        return "Weather service returned invalid data."

    try:
        days = data.get("weather") or []
        if not days:
            return "No forecast data."
        day_idx = 0
        target_minutes = when.hour * 60 + when.minute
        hourly = days[day_idx].get("hourly") or []
        if not hourly:
            cur = data["current_condition"][0]
            desc = _wttr_value(cur.get("weatherDesc", [{}]))
            return f"{desc}, {cur.get('temp_C', '?')}°C"
        best = hourly[0]
        best_diff = 9999
        for h in hourly:
            t_raw = str(h.get("time", "0"))
            try:
                # wttr uses 3-digit time: 0 -> 00:00, 300 -> 03:00, 1200 -> 12:00
                t_val = int(t_raw)
                h_m = (t_val // 100) * 60 + (t_val % 100)
            except ValueError:
                h_m = 0
            diff = abs(h_m - target_minutes)
            if diff < best_diff:
                best_diff = diff
                best = h
        desc = _wttr_value(best.get("weatherDesc", [{}]))
        temp = best.get("tempC", "?")
        rain = best.get("chanceofrain", "?")
        return f"{desc}, {temp}°C, rain chance {rain}%"
    except (KeyError, IndexError, TypeError):
        return "Could not parse weather forecast."


def _town_in_label(town: str, label: str) -> bool:
    if not town or not label:
        return False
    t = town.lower()
    lab = label.lower()
    return t in lab or lab.split(",")[0].strip().lower() == t


@dataclass
class OsrmRoute:
    poly: list[tuple[float, float]]
    duration_s: float
    distance_m: float
    origin_label: str
    dest_label: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    profile: str


def fetch_osrm_route(
    client: httpx.Client,
    origin: str,
    destination: str,
    mode: str,
) -> OsrmRoute | str:
    profile = mode.strip().lower() or "driving"
    aliases = {
        "drive": "driving",
        "car": "driving",
        "walk": "walking",
        "foot": "walking",
        "bike": "cycling",
        "bicycle": "cycling",
    }
    profile = aliases.get(profile, profile)
    if profile not in ("driving", "walking", "cycling"):
        return f"Unsupported mode '{mode}'."

    o_res = geocode(client, origin)
    if isinstance(o_res, str):
        return o_res
    d_res = geocode(client, destination)
    if isinstance(d_res, str):
        return d_res

    coords = f"{o_res['lon']},{o_res['lat']};{d_res['lon']},{d_res['lat']}"
    url = f"{OSRM_URL}/{profile}/{coords}"
    try:
        resp = client.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            headers={"User-Agent": USER_AGENT},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return f"Routing HTTP error: {e}"
    except json.JSONDecodeError:
        return "Routing service returned invalid data."

    if data.get("code") != "Ok":
        return f"Routing failed: {data.get('message', data.get('code', 'unknown'))}"

    try:
        route = data["routes"][0]
        poly = polyline_from_osrm_route(route)
        if len(poly) < 2:
            return "Route geometry too short."
        return OsrmRoute(
            poly=poly,
            duration_s=float(route["duration"]),
            distance_m=float(route["distance"]),
            origin_label=o_res["label"],
            dest_label=d_res["label"],
            origin_lat=o_res["lat"],
            origin_lon=o_res["lon"],
            dest_lat=d_res["lat"],
            dest_lon=d_res["lon"],
            profile=profile,
        )
    except (KeyError, TypeError, ValueError) as e:
        return f"Could not parse routing response: {e}"


def discover_intermediate_stops(
    client: httpx.Client,
    route: OsrmRoute,
    depart: datetime,
    max_towns: int = 6,
) -> list[RouteStop]:
    """Major towns between origin and destination with estimated arrival times."""
    samples = sample_middle_points(route.poly, max_points=max_towns + 2)
    seen: set[str] = set()
    stops: list[RouteStop] = []

    for i, (lat, lon, prog_m) in enumerate(samples):
        if i > 0:
            time.sleep(1.06)
        town = reverse_town_name(client, lat, lon)
        if not town:
            continue
        key = town.lower()
        if key in seen:
            continue
        if _town_in_label(town, route.origin_label) or _town_in_label(town, route.dest_label):
            continue
        if _haversine_m(lat, lon, route.origin_lat, route.origin_lon) < 2500:
            continue
        if _haversine_m(lat, lon, route.dest_lat, route.dest_lon) < 2500:
            continue
        seen.add(key)
        path_len = route_length_m(route.poly)
        frac = prog_m / path_len if path_len > 0 else 0.0
        dur_s = route.duration_s * frac
        arrival = depart + timedelta(seconds=dur_s)
        stops.append(
            RouteStop(
                town=town,
                lat=lat,
                lon=lon,
                distance_km=prog_m / 1000.0,
                duration_from_start_s=dur_s,
                arrival=arrival,
            )
        )
        if len(stops) >= max_towns:
            break
    return stops
