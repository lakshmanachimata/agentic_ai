"""
LangChain agent: Ollama + nearby places to eat from OpenStreetMap (free tier).

- Geocoding: Nominatim (forward + light reverse for area labels; throttle between calls)
- POIs: Overpass API (public instance; keep queries small)
- Routes: OSRM public demo (same as travel_agent) to find places along a driving/walking/cycling path

Requires Ollama running locally with the model pulled:
  ollama pull qwen3.5:4b

Interactive mode keeps session memory across turns (``/reset`` clears it).

Run (interactive; Ctrl+D / EOF to exit):
  python restaurant_agent.py

One-off:
  python restaurant_agent.py "Where can I eat near Shibuya Station?"
  python restaurant_agent.py "Casual food along the drive from Boston to Providence"
"""

from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive
from route_common import (
    OsrmRoute,
    discover_intermediate_stops,
    fetch_osrm_route,
    format_duration,
    format_time,
    parse_start_time,
)
from travel_agent import OSRM_URL, USER_AGENT as TRAVEL_HTTP_USER_AGENT, _geocode, _normalize_profile

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "agentic_ai-restaurant-agent/1.0 (local project; contact via repo)"

# Nominatim asks for identifiable UA; Overpass public instances ask for modest use / small queries.


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _seg_len_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return _haversine_m(a[0], a[1], b[0], b[1])


def _project_fraction_on_segment(
    pla: float,
    plo: float,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
) -> float:
    """Projection parameter t in [0,1] of point (pla,plo) onto segment a→b (local planar)."""
    mid_lat = (a_lat + b_lat) / 2.0
    cos_ref = math.cos(math.radians(mid_lat)) * 111_320.0
    lat_m = 111_320.0
    bx = (b_lon - a_lon) * cos_ref
    by = (b_lat - a_lat) * lat_m
    px = (plo - a_lon) * cos_ref
    py = (pla - a_lat) * lat_m
    len2 = bx * bx + by * by
    if len2 < 1e-9:
        return 0.0
    t = (px * bx + py * by) / len2
    return max(0.0, min(1.0, t))


def _progress_and_distance_to_polyline(
    plat: float,
    plon: float,
    poly: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return (distance_m along route from start to closest route point, perpendicular distance_m)."""
    if len(poly) < 2:
        if not poly:
            return 0.0, float("inf")
        return 0.0, _haversine_m(plat, plon, poly[0][0], poly[0][1])

    best_dist = float("inf")
    best_prog = 0.0
    cum = 0.0
    for i in range(len(poly) - 1):
        a_lat, a_lon = poly[i]
        b_lat, b_lon = poly[i + 1]
        seg_len = _seg_len_m(poly[i], poly[i + 1])
        if seg_len < 0.5:
            d = _haversine_m(plat, plon, a_lat, a_lon)
            prog = cum
        else:
            frac = _project_fraction_on_segment(plat, plon, a_lat, a_lon, b_lat, b_lon)
            ilat = a_lat + frac * (b_lat - a_lat)
            ilon = a_lon + frac * (b_lon - a_lon)
            d = _haversine_m(plat, plon, ilat, ilon)
            prog = cum + frac * seg_len
        if d < best_dist:
            best_dist = d
            best_prog = prog
        cum += seg_len
    return best_prog, best_dist


def _polyline_from_osrm_route(route: dict[str, Any]) -> list[tuple[float, float]]:
    geom = route.get("geometry")
    if not isinstance(geom, dict) or geom.get("type") != "LineString":
        return []
    raw = geom.get("coordinates") or []
    out: list[tuple[float, float]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        lon, lat = float(pair[0]), float(pair[1])
        out.append((lat, lon))
    return out


def _sample_polyline_evenly(poly: list[tuple[float, float]], max_samples: int = 7) -> list[tuple[float, float]]:
    """Pick points spaced by distance along the line (for small Overpass queries)."""
    if not poly:
        return []
    if len(poly) == 1:
        return [poly[0]]
    segments = [_seg_len_m(poly[i], poly[i + 1]) for i in range(len(poly) - 1)]
    total = sum(segments)
    if total < 1e-6:
        return [poly[0], poly[-1]]

    n = max(3, min(int(max_samples), 10))
    out: list[tuple[float, float]] = []
    for s in range(n):
        tgt = total * (s / (n - 1))
        cum = 0.0
        for i, L in enumerate(segments):
            if cum + L >= tgt or i == len(segments) - 1:
                denom = L if L > 1e-6 else 1.0
                t = max(0.0, min(1.0, (tgt - cum) / denom))
                a, b = poly[i], poly[i + 1]
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                break
            cum += L

    deduped: list[tuple[float, float]] = []
    for p in out:
        if not deduped or _seg_len_m(deduped[-1], p) >= 200.0:
            deduped.append(p)
    return deduped or out


def _route_length_m(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return sum(_seg_len_m(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _point_along_route_at_distance(
    poly: list[tuple[float, float]], distance_m: float
) -> tuple[float, float] | None:
    """Point at ``distance_m`` meters along ``poly`` from the start (clamped)."""
    if not poly:
        return None
    if len(poly) == 1:
        return poly[0]
    target = max(0.0, distance_m)
    cum = 0.0
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        seg_len = _seg_len_m(a, b)
        if cum + seg_len >= target:
            denom = seg_len if seg_len > 1e-6 else 1.0
            t = max(0.0, min(1.0, (target - cum) / denom))
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        cum += seg_len
    return poly[-1]


def _sample_route_middle_interval(
    poly: list[tuple[float, float]],
    lo_m: float,
    hi_m: float,
    max_samples: int,
) -> list[tuple[float, float]]:
    """Evenly spaced samples between ``lo_m`` and ``hi_m`` along the route (by path distance)."""
    lo_m = max(0.0, lo_m)
    hi_m = max(lo_m, hi_m)
    span = hi_m - lo_m
    if span < 80.0:
        p = _point_along_route_at_distance(poly, (lo_m + hi_m) / 2.0)
        return [p] if p else []

    n = max(2, min(int(max_samples), 8))
    out: list[tuple[float, float]] = []
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.5
        d = lo_m + frac * span
        pt = _point_along_route_at_distance(poly, d)
        if pt:
            out.append(pt)

    deduped: list[tuple[float, float]] = []
    for p in out:
        if not deduped or _seg_len_m(deduped[-1], p) >= 250.0:
            deduped.append(p)
    return deduped or out


def _reverse_place_label(client: httpx.Client, lat: float, lon: float) -> str:
    """Settlement-style label from Nominatim reverse (free; caller should throttle)."""
    try:
        resp = client.get(
            NOMINATIM_REVERSE_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 12,
                "addressdetails": "1",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
        return ""

    addr = data.get("address") or {}
    if isinstance(addr, dict):
        for key in ("town", "city", "village", "hamlet", "suburb", "neighbourhood", "county"):
            v = addr.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()[:80]
    dn = data.get("display_name")
    if isinstance(dn, str) and dn.strip():
        return dn.split(",")[0].strip()[:80]
    return ""


def _fetch_route_polyline(
    client: httpx.Client,
    origin: str,
    destination: str,
    mode: str,
) -> tuple[
    list[tuple[float, float]] | None,
    str,
    str | None,
    str | None,
    str,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    """Return polyline, error, labels, profile, and endpoint coordinates (lat/lon) if success."""
    profile = _normalize_profile(mode)
    if profile not in ("driving", "walking", "cycling"):
        return (
            None,
            f"Unsupported mode '{mode}'. Use driving, walking, or cycling.",
            None,
            None,
            str(profile),
            None,
            None,
            None,
            None,
        )

    o_res = _geocode(client, origin)
    if isinstance(o_res, str):
        return None, o_res, None, None, str(profile), None, None, None, None
    d_res = _geocode(client, destination)
    if isinstance(d_res, str):
        return (
            None,
            d_res,
            None,
            None,
            str(profile),
            float(o_res["lat"]),
            float(o_res["lon"]),
            None,
            None,
        )

    o_lon, o_lat = o_res["lon"], o_res["lat"]
    d_lon, d_lat = d_res["lon"], d_res["lat"]
    coords = f"{o_lon},{o_lat};{d_lon},{d_lat}"
    url = f"{OSRM_URL}/{profile}/{coords}"

    try:
        resp = client.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "false"},
            headers={"User-Agent": TRAVEL_HTTP_USER_AGENT},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return (
            None,
            f"Routing HTTP error: {e}",
            o_res.get("label"),
            d_res.get("label"),
            str(profile),
            float(o_lat),
            float(o_lon),
            float(d_lat),
            float(d_lon),
        )
    except json.JSONDecodeError:
        return (
            None,
            "Routing service returned invalid data.",
            o_res.get("label"),
            d_res.get("label"),
            str(profile),
            float(o_lat),
            float(o_lon),
            float(d_lat),
            float(d_lon),
        )

    if data.get("code") != "Ok":
        msg = data.get("message", data.get("code", "unknown error"))
        return (
            None,
            f"Routing failed: {msg}",
            o_res.get("label"),
            d_res.get("label"),
            str(profile),
            float(o_lat),
            float(o_lon),
            float(d_lat),
            float(d_lon),
        )

    try:
        route = data["routes"][0]
    except (KeyError, IndexError, TypeError):
        return (
            None,
            "Routing response missing route.",
            o_res.get("label"),
            d_res.get("label"),
            str(profile),
            float(o_lat),
            float(o_lon),
            float(d_lat),
            float(d_lon),
        )

    poly = _polyline_from_osrm_route(route)
    if len(poly) < 2:
        return (
            None,
            "Route geometry too short to search along.",
            o_res.get("label"),
            d_res.get("label"),
            str(profile),
            float(o_lat),
            float(o_lon),
            float(d_lat),
            float(d_lon),
        )

    return (
        poly,
        "",
        o_res.get("label"),
        d_res.get("label"),
        str(profile),
        float(o_lat),
        float(o_lon),
        float(d_lat),
        float(d_lon),
    )


def _sanitize_cuisine_token(raw: str) -> str | None:
    s = raw.strip().lower()
    if not s:
        return None
    safe = "".join(c for c in s if c.isalnum() or c in ("-", "_"))[:48]
    return safe or None


def _overpass_query(lat: float, lon: float, radius_m: int, cuisine: str | None) -> str:
    r = max(200, min(int(radius_m), 2500))
    cuisine_line = ""
    if cuisine:
        cuisine_line = f'["cuisine"="{cuisine}"]'

    # Small query: restaurants, fast food, cafes — nodes + ways with out center.
    return f"""[out:json][timeout:25];
(
  node["amenity"="restaurant"]{cuisine_line}(around:{r},{lat},{lon});
  way["amenity"="restaurant"]{cuisine_line}(around:{r},{lat},{lon});
  node["amenity"="fast_food"]{cuisine_line}(around:{r},{lat},{lon});
  way["amenity"="fast_food"]{cuisine_line}(around:{r},{lat},{lon});
  node["amenity"="cafe"]{cuisine_line}(around:{r},{lat},{lon});
  way["amenity"="cafe"]{cuisine_line}(around:{r},{lat},{lon});
);
out center tags;
"""


def _post_overpass(client: httpx.Client, query: str) -> httpx.Response:
    headers = {"User-Agent": USER_AGENT, "Content-Type": "text/plain"}
    last: httpx.Response | None = None
    for url in OVERPASS_URLS:
        resp = client.post(url, content=query, headers=headers, timeout=45.0)
        last = resp
        if resp.status_code == 200:
            return resp
        if resp.status_code not in (429, 502, 503, 504):
            resp.raise_for_status()
    assert last is not None
    return last


@tool
def find_places_to_eat_along_route(
    origin: str,
    destination: str,
    mode: str = "driving",
    exclude_endpoints_meters: int = 2500,
    intermediate_search_radius_meters: int = 800,
    max_distance_from_route_meters: int = 550,
    cuisine: str = "",
    max_results: int = 14,
) -> str:
    """Find restaurants **between** origin and destination — not at the endpoints (free OSRM + OSM + Nominatim).

    Builds an OSRM route, picks intermediate segments (excluding the first/last part of the path
    and dropping POIs within a buffer of the geocoded start/end), runs small Overpass searches
    around sampled **in-between** locations, and optionally labels those locations via Nominatim
    reverse (throttled). Venues right at source or destination are excluded.

    Args:
        origin: Start location (city, address, landmark, or 'lat,lon').
        destination: End location.
        mode: driving (default), walking, or cycling.
        exclude_endpoints_meters: Omit the first/last this many meters along the route **and** drop
            POIs closer than this (straight-line) to the geocoded origin or destination (800–5000; default 2500).
        intermediate_search_radius_meters: Overpass search radius around each midpoint sample (400–1000; default 800).
        max_distance_from_route_meters: Keep POIs within this distance of the route polyline (250–800; default 550).
        cuisine: Optional OSM cuisine tag filter. Leave empty for any.
        max_results: Cap on listed venues (max 20).

    Data is incomplete by nature; confirm before visiting.
    """
    cap = max(1, min(int(max_results), 20))
    buf = max(800, min(int(exclude_endpoints_meters), 5000))
    search_r = max(400, min(int(intermediate_search_radius_meters), 1000))
    cor = max(250, min(int(max_distance_from_route_meters), 800))
    cuisine_token = _sanitize_cuisine_token(cuisine) if cuisine else None

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    samples: list[tuple[float, float]] = []
    route: list[tuple[float, float]] | None = None
    o_lab = d_lab = ""
    profile = mode
    o_la = o_lo = d_la = d_lo = 0.0
    lo_prog = hi_prog = 0.0
    short_route_fallback = False

    with httpx.Client() as client:
        (
            poly,
            err,
            o_lab,
            d_lab,
            profile,
            o_lat_raw,
            o_lon_raw,
            d_lat_raw,
            d_lon_raw,
        ) = _fetch_route_polyline(client, origin.strip(), destination.strip(), mode)
        if poly is None:
            return err

        route = poly
        assert o_lat_raw is not None and o_lon_raw is not None and d_lat_raw is not None and d_lon_raw is not None
        o_la, o_lo, d_la, d_lo = o_lat_raw, o_lon_raw, d_lat_raw, d_lon_raw

        total_len = _route_length_m(poly)
        min_gap = 400.0
        if total_len <= 2 * buf + min_gap:
            short_route_fallback = True
            lo_prog = total_len * 0.25
            hi_prog = total_len * 0.75
            if hi_prog - lo_prog < 200.0:
                mid = total_len / 2.0
                lo_prog = max(0.0, mid - 120.0)
                hi_prog = min(total_len, mid + 120.0)
        else:
            lo_prog = float(buf)
            hi_prog = total_len - float(buf)

        samples = _sample_route_middle_interval(poly, lo_prog, hi_prog, max_samples=6)
        if not samples:
            return (
                f"No intermediate waypoints on route {o_lab or origin} → {d_lab or destination}. "
                "Try a longer trip or different place names."
            )

        area_labels: list[str] = []
        n_s = len(samples)
        for i, (slat, slon) in enumerate(samples):
            if i > 0:
                time.sleep(1.06)
            lbl = _reverse_place_label(client, slat, slon)
            frac = i / max(1, n_s - 1) if n_s > 1 else 0.5
            km_mark = (lo_prog + frac * (hi_prog - lo_prog)) / 1000.0
            area_labels.append(lbl if lbl else f"en route ~{km_mark:.1f} km from start")

        for slat, slon in samples:
            q = _overpass_query(slat, slon, search_r, cuisine_token)
            try:
                resp = _post_overpass(client, q)
                if resp.status_code == 429:
                    return "Overpass rate limit — wait a minute and try again."
                if resp.status_code >= 500:
                    return "Overpass servers busy — try again in a few minutes."
                resp.raise_for_status()
                chunk = resp.json()
            except httpx.HTTPStatusError as e:
                return f"Overpass HTTP error: {e}"
            except httpx.HTTPError as e:
                return f"Overpass HTTP error: {e}"
            except json.JSONDecodeError:
                return "Overpass returned invalid JSON."

            for el in (chunk.get("elements") or [])[:85]:
                eid = el.get("id")
                et = el.get("type")
                if eid is None or et not in ("node", "way", "relation"):
                    continue
                merged[(str(et), int(eid))] = el

    assert route is not None

    rows: list[tuple[float, float, str]] = []
    for el in merged.values():
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        amenity = (tags.get("amenity") or "").strip()
        if not amenity:
            continue

        el_lat = el.get("lat")
        el_lon = el.get("lon")
        if el_lat is None or el_lon is None:
            center = el.get("center") or {}
            el_lat = center.get("lat")
            el_lon = center.get("lon")
        if el_lat is None or el_lon is None:
            continue

        try:
            plat, plon = float(el_lat), float(el_lon)
        except (TypeError, ValueError):
            continue

        if _haversine_m(plat, plon, o_la, o_lo) < buf:
            continue
        if _haversine_m(plat, plon, d_la, d_lo) < buf:
            continue

        prog_m, off_m = _progress_and_distance_to_polyline(plat, plon, route)
        slack = 180.0
        if prog_m < lo_prog - slack or prog_m > hi_prog + slack:
            continue
        if off_m > cor * 1.28:
            continue

        nearest_area = "between endpoints"
        if samples and area_labels:
            bi = min(
                range(len(samples)),
                key=lambda j: _haversine_m(plat, plon, samples[j][0], samples[j][1]),
            )
            nearest_area = area_labels[bi] if bi < len(area_labels) else nearest_area

        cuisine_v = (tags.get("cuisine") or "").strip()
        km_along = prog_m / 1000.0
        parts = [
            f"- {name or '(no name in OSM)'}",
            f"  [{amenity}]",
            f"  area: {nearest_area}",
            f"  ~{km_along:.1f} km along route, ~{int(round(off_m))} m from path",
        ]
        if cuisine_v:
            parts.append(f"  cuisine: {cuisine_v}")
        opening = (tags.get("opening_hours") or "").strip()
        if opening:
            parts.append(f"  hours (OSM): {opening}")
        rows.append((prog_m, off_m, "\n".join(parts)))

    rows.sort(key=lambda t: (t[0], t[1]))
    rows = rows[:cap]

    if not rows:
        msg = (
            f"No OSM restaurants/cafés/fast food found **between** {o_lab or origin} and {d_lab or destination} "
            f"(excluding ~{buf} m around each endpoint)."
        )
        if cuisine_token:
            msg += f" Cuisine filter: {cuisine_token}."
        msg += " Try increasing exclude_endpoints_meters slightly, intermediate_search_radius_meters, or drop cuisine."
        return msg

    mid_note = (
        " (short trip: used middle segment of path)"
        if short_route_fallback
        else ""
    )
    header = [
        f"In-between stops ({profile}){mid_note}: {o_lab} → {d_lab}",
        (
            f"Excluded origin/destination buffers: ~{buf} m along-route window [{lo_prog / 1000:.1f}–{hi_prog / 1000:.1f}] km "
            f"and POIs within ~{buf} m straight-line of each endpoint | {len(samples)} intermediate areas | "
            f"Search radius ~{search_r} m each (OSRM + Overpass + Nominatim)"
        ),
    ]
    if cuisine_token:
        header.append(f"Filter: cuisine={cuisine_token}")
    footer = (
        "\nSource: OSRM route, OpenStreetMap via Overpass, labels via Nominatim (ODbL / OSM). "
        "Intermediate coverage is approximate — confirm before you go."
    )
    return "\n".join(header) + "\n" + "\n".join(r[2] for r in rows) + footer


@tool
def find_places_to_eat(
    area: str,
    radius_meters: int = 800,
    cuisine: str = "",
    max_results: int = 12,
) -> str:
    """Find restaurants, fast food, and cafés near a place using OpenStreetMap (free, no API key).

    Args:
        area: Neighborhood, city, landmark, or address (e.g. 'Le Marais, Paris').
        radius_meters: Search radius in meters around the geocoded point (200–2500; default 800).
        cuisine: Optional OSM cuisine tag value to filter (e.g. 'italian', 'japanese'). Leave empty for any.
        max_results: Cap on listed venues (max 20).

    Data is community-contributed and may be incomplete; verify before visiting.
    """
    cap = max(1, min(int(max_results), 20))
    lat_lon = None
    with httpx.Client() as client:
        geo = _geocode(client, area)
        if isinstance(geo, str):
            return geo
        lat, lon = geo["lat"], geo["lon"]
        lat_lon = geo

        cuisine_token = _sanitize_cuisine_token(cuisine) if cuisine else None
        q = _overpass_query(lat, lon, radius_meters, cuisine_token)

        try:
            resp = _post_overpass(client, q)
            if resp.status_code == 429:
                return "Overpass rate limit — wait a minute and try again, or reduce radius."
            if resp.status_code >= 500:
                return "Overpass servers busy — try again in a few minutes or use a smaller radius."
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Overpass HTTP error: {e}"
        except httpx.HTTPError as e:
            return f"Overpass HTTP error: {e}"
        except json.JSONDecodeError:
            return "Overpass returned invalid JSON."

    elements = (data.get("elements") or [])[:100]
    rows: list[tuple[float, str]] = []

    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        amenity = (tags.get("amenity") or "").strip()
        if not amenity:
            continue

        el_lat = el.get("lat")
        el_lon = el.get("lon")
        if el_lat is None or el_lon is None:
            center = el.get("center") or {}
            el_lat = center.get("lat")
            el_lon = center.get("lon")
        if el_lat is None or el_lon is None:
            continue

        try:
            plat, plon = float(el_lat), float(el_lon)
        except (TypeError, ValueError):
            continue

        dist = _haversine_m(lat_lon["lat"], lat_lon["lon"], plat, plon)
        cuisine_v = (tags.get("cuisine") or "").strip()
        parts = [
            f"- {name or '(no name in OSM)'}",
            f"  [{amenity}]",
            f"  ~{int(round(dist))} m",
        ]
        if cuisine_v:
            parts.append(f"  cuisine: {cuisine_v}")
        opening = (tags.get("opening_hours") or "").strip()
        if opening:
            parts.append(f"  hours (OSM): {opening}")
        rows.append((dist, "\n".join(parts)))

    rows.sort(key=lambda t: t[0])
    rows = rows[:cap]

    if not rows:
        msg = (
            f"No OSM-tagged restaurants/cafés/fast food within {max(200, min(int(radius_meters), 2500))} m "
            f"of {lat_lon['label']}"
        )
        if cuisine_token:
            msg += f" with cuisine='{cuisine_token}'."
        else:
            msg += ". Try a larger radius or a different area label."
        return msg

    header = [
        f"Near: {lat_lon['label']}",
        f"Radius: {max(200, min(int(radius_meters), 2500))} m (OpenStreetMap; may be incomplete)",
    ]
    if cuisine_token:
        header.append(f"Filter: cuisine={cuisine_token}")
    footer = (
        "\nSource: OpenStreetMap via Overpass API (ODbL). "
        "Opening hours and names are not verified — confirm before you go."
    )
    return "\n".join(header) + "\n" + "\n".join(r[1] for r in rows) + footer


def _poi_lines_from_elements(
    elements: list[dict[str, Any]],
    ref_lat: float,
    ref_lon: float,
    cap: int,
) -> list[str]:
    rows: list[tuple[float, str]] = []
    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        amenity = (tags.get("amenity") or "").strip()
        if not amenity:
            continue
        el_lat = el.get("lat")
        el_lon = el.get("lon")
        if el_lat is None or el_lon is None:
            center = el.get("center") or {}
            el_lat = center.get("lat")
            el_lon = center.get("lon")
        if el_lat is None or el_lon is None:
            continue
        try:
            plat, plon = float(el_lat), float(el_lon)
        except (TypeError, ValueError):
            continue
        dist = _haversine_m(ref_lat, ref_lon, plat, plon)
        cuisine_v = (tags.get("cuisine") or "").strip()
        parts = [f"  - {name or '(no name)'} [{amenity}] ~{int(round(dist))} m"]
        if cuisine_v:
            parts.append(f"    cuisine: {cuisine_v}")
        rows.append((dist, "\n".join(parts)))
    rows.sort(key=lambda t: t[0])
    return [r[1] for r in rows[:cap]]


@tool
def find_restaurants_at_towns_on_route(
    origin: str,
    destination: str,
    mode: str = "driving",
    start_time: str = "",
    max_towns: int = 5,
    per_town_max: int = 4,
    search_radius_meters: int = 700,
    cuisine: str = "",
) -> str:
    """Restaurants at major **intermediate towns** between origin and destination (not at endpoints).

    Uses OSRM + Nominatim to find towns along the route, then Overpass for eateries in each town.
    ``start_time`` is departure from origin (default **8:00 AM today** if empty) — shown as est. arrival per town.

    Args:
        origin, destination: Place names or addresses.
        mode: driving (default), walking, cycling.
        start_time: e.g. '08:00', '9 AM' — optional.
        max_towns: How many intermediate towns to list (max 6).
        per_town_max: Max restaurants per town (max 6).
        search_radius_meters: Overpass radius around each town center (400–1000).
        cuisine: Optional OSM cuisine filter.
    """
    cap_towns = max(1, min(int(max_towns), 6))
    cap_each = max(1, min(int(per_town_max), 6))
    search_r = max(400, min(int(search_radius_meters), 1000))
    cuisine_token = _sanitize_cuisine_token(cuisine) if cuisine else None
    depart = parse_start_time(start_time)

    with httpx.Client() as client:
        route_result = fetch_osrm_route(client, origin.strip(), destination.strip(), mode)
        if isinstance(route_result, str):
            return route_result
        route: OsrmRoute = route_result
        stops = discover_intermediate_stops(client, route, depart, max_towns=cap_towns)

    if not stops:
        return (
            f"No distinct intermediate towns found between {route.origin_label} and {route.dest_label}. "
            "Try find_places_to_eat_along_route for corridor search, or a longer route."
        )

    lines = [
        f"Restaurants at towns between {route.origin_label.split(',')[0]} → {route.dest_label.split(',')[0]}",
        f"Mode: {route.profile} | Depart ~{depart.strftime('%H:%M')} (8:00 AM default if not specified)",
        f"Excludes dining at origin and destination.",
        "",
    ]

    with httpx.Client() as client:
        for stop in stops:
            q = _overpass_query(stop.lat, stop.lon, search_r, cuisine_token)
            try:
                resp = _post_overpass(client, q)
                if resp.status_code != 200:
                    lines.append(
                        f"### {stop.town} (arrive ~{format_time(stop.arrival)}, "
                        f"{format_duration(stop.duration_from_start_s)} from start)"
                    )
                    lines.append("  (Overpass busy — skip)")
                    continue
                data = resp.json()
            except httpx.HTTPError:
                lines.append(f"### {stop.town} — search error")
                continue

            poi_lines = _poi_lines_from_elements(
                (data.get("elements") or [])[:60],
                stop.lat,
                stop.lon,
                cap_each,
            )
            lines.append(
                f"### {stop.town} — est. arrive {format_time(stop.arrival)} "
                f"({format_duration(stop.duration_from_start_s)} from start, ~{stop.distance_km:.0f} km along route)"
            )
            if poi_lines:
                lines.extend(poi_lines)
            else:
                lines.append("  (no OSM restaurants/cafés/fast food in this radius)")
            lines.append("")

    lines.append(
        "Source: OSRM + Nominatim + Overpass (ODbL). Verify hours and names before visiting."
    )
    return "\n".join(lines)


def build_agent():
    llm = ChatOllama(
        model="qwen3.5:4b",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[
            find_places_to_eat,
            find_places_to_eat_along_route,
            find_restaurants_at_towns_on_route,
        ],
        system_prompt=(
            "You help travelers find places to eat using OpenStreetMap (free data).\n"
            "- **Single area**: find_places_to_eat (radius 2000 - 5000 m, default 3000).\n"
            "- **Source + destination, restaurants grouped by intermediate towns** (excludes "
            "origin/destination): find_restaurants_at_towns_on_route. Pass start_time if given; "
            "otherwise tool assumes 8:00 AM departure.\n"
            "- **Corridor search** along the path (not grouped by town): find_places_to_eat_along_route.\n"
            "Optional cuisine: OSM token (italian, etc.). Never invent venues."
        ),
        checkpointer=MemorySaver(),
    )


def run_query(graph: Any, question: str, *, thread_id: str | None = None) -> str:
    return invoke_agent(graph, question, thread_id=thread_id)


def main() -> None:
    graph = build_agent()
    q_one = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if q_one:
        print(run_query(graph, q_one))
        return

    run_interactive(
        "Restaurant / dining agent",
        "ask for food near an area, or along a route between two places.",
        graph,
    )


if __name__ == "__main__":
    main()
