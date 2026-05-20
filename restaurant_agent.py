"""
LangChain agent: Ollama + nearby places to eat from OpenStreetMap (free tier).

- Geocoding: Nominatim (usage policy: one request per second; identify with User-Agent)
- POIs: Overpass API (public instance; keep queries small)
- Routes: OSRM public demo (same as travel_agent) to find places along a driving/walking/cycling path

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:latest

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
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive
from travel_agent import OSRM_URL, USER_AGENT as TRAVEL_HTTP_USER_AGENT, _geocode, _normalize_profile

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
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


def _fetch_route_polyline(
    client: httpx.Client,
    origin: str,
    destination: str,
    mode: str,
) -> tuple[list[tuple[float, float]] | None, str, str | None, str | None, str]:
    """Return (polyline lat,lon points, error, origin_label, dest_label, profile)."""
    profile = _normalize_profile(mode)
    if profile not in ("driving", "walking", "cycling"):
        return (
            None,
            f"Unsupported mode '{mode}'. Use driving, walking, or cycling.",
            None,
            None,
            str(profile),
        )

    o_res = _geocode(client, origin)
    if isinstance(o_res, str):
        return None, o_res, None, None, str(profile)
    d_res = _geocode(client, destination)
    if isinstance(d_res, str):
        return None, d_res, None, None, str(profile)

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
        return None, f"Routing HTTP error: {e}", o_res.get("label"), d_res.get("label"), str(profile)
    except json.JSONDecodeError:
        return None, "Routing service returned invalid data.", o_res.get("label"), d_res.get("label"), str(profile)

    if data.get("code") != "Ok":
        msg = data.get("message", data.get("code", "unknown error"))
        return None, f"Routing failed: {msg}", o_res.get("label"), d_res.get("label"), str(profile)

    try:
        route = data["routes"][0]
    except (KeyError, IndexError, TypeError):
        return None, "Routing response missing route.", o_res.get("label"), d_res.get("label"), str(profile)

    poly = _polyline_from_osrm_route(route)
    if len(poly) < 2:
        return None, "Route geometry too short to search along.", o_res.get("label"), d_res.get("label"), str(profile)

    return poly, "", o_res.get("label"), d_res.get("label"), str(profile)


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
    corridor_radius_meters: int = 500,
    cuisine: str = "",
    max_results: int = 14,
) -> str:
    """Find restaurants, cafés, and fast food along the route between two places (free OSRM + OSM).

    Uses the public OSRM demo for route geometry (same service style as the travel agent)
    and Overpass queries near a few sampled points on that path. This is approximate:
    very dense urban grids or long detours may miss some venues.

    Args:
        origin: Start location (city, address, landmark, or 'lat,lon').
        destination: End location.
        mode: driving (default), walking, or cycling.
        corridor_radius_meters: Max distance from the route polyline to include a POI (250–800).
        cuisine: Optional OSM cuisine tag filter. Leave empty for any.
        max_results: Cap on listed venues (max 20).

    Data is community-contributed; verify before visiting.
    """
    cap = max(1, min(int(max_results), 20))
    cor = max(250, min(int(corridor_radius_meters), 800))
    cuisine_token = _sanitize_cuisine_token(cuisine) if cuisine else None
    search_r = max(250, min(cor + 80, 750))

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    samples: list[tuple[float, float]] = []

    with httpx.Client() as client:
        route, err, o_lab, d_lab, profile = _fetch_route_polyline(
            client, origin.strip(), destination.strip(), mode
        )
        if route is None:
            return err
        samples = _sample_polyline_evenly(route, max_samples=7)
        for lat, lon in samples:
            q = _overpass_query(lat, lon, search_r, cuisine_token)
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

            for el in (chunk.get("elements") or [])[:80]:
                eid = el.get("id")
                et = el.get("type")
                if eid is None or et not in ("node", "way", "relation"):
                    continue
                merged[(str(et), int(eid))] = el

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

        prog_m, off_m = _progress_and_distance_to_polyline(plat, plon, route)
        if off_m > cor * 1.25:
            continue

        cuisine_v = (tags.get("cuisine") or "").strip()
        km_along = prog_m / 1000.0
        parts = [
            f"- {name or '(no name in OSM)'}",
            f"  [{amenity}]",
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
            f"No OSM-tagged restaurants/cafés/fast food within ~{cor} m of the route from "
            f"{o_lab or origin} to {d_lab or destination}."
        )
        if cuisine_token:
            msg += f" (cuisine filter: {cuisine_token})"
        msg += " Try widening corridor_radius_meters or dropping the cuisine filter."
        return msg

    sample_n = len(samples)
    header = [
        f"Along route ({profile}): {o_lab} → {d_lab}",
        f"Corridor ≈{cor} m from path | {sample_n} sample points (OSM + OSRM; approximate)",
    ]
    if cuisine_token:
        header.append(f"Filter: cuisine={cuisine_token}")
    footer = (
        "\nSource: OSRM route + OpenStreetMap via Overpass (ODbL). "
        "Not all roadside options are included — confirm before you go."
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


def build_agent():
    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[find_places_to_eat, find_places_to_eat_along_route],
        system_prompt=(
            "You help travelers find places to eat using OpenStreetMap (free data).\n"
            "- For a **single area** (neighborhood, city, station, address): use "
            "find_places_to_eat with a clear area string. Default radius is fine unless "
            "the user asks otherwise; keep radius_meters between 200 and 2500.\n"
            "- For **food along a journey** (between two places, 'on the way', 'during the drive', "
            "or combined travel + dining): use find_places_to_eat_along_route with origin and "
            "destination and the same travel mode when specified (driving, walking, cycling; "
            "default driving). Adjust corridor_radius_meters (250–800) only if they ask.\n"
            "You do not compute travel duration — only where to eat along the path; if they need "
            "drive time as well, tell them to use the orchestrator app or a travel-time assistant.\n"
            "Optional cuisine filter: simple OSM token (italian, etc.) or omit. "
            "Summarize clearly; data may be incomplete. Never invent venues."
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
