"""
LangChain agent (ReAct-style graph): Ollama (qwen3.5:4b) + travel time via OSM/Nominatim + OSRM.

Geocoding: Nominatim (OpenStreetMap) — https://nominatim.org
Routing: OSRM public demo — https://project-osrm.org
Weather along route: wttr.in (free)

Requires Ollama running locally with the model pulled:
  ollama pull qwen3.5:4b

Interactive mode keeps session memory across turns (``/reset`` clears it).

Run (interactive; Ctrl+D / EOF to exit):
  python travel_agent.py

One-off:
  python travel_agent.py "Drive from Paris to Lyon leaving at 9 AM — towns and weather on the way"
"""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive
from route_common import (
    OsrmRoute,
    discover_intermediate_stops,
    fetch_osrm_route,
    format_distance,
    format_duration,
    format_time,
    geocode,
    parse_start_time,
    weather_summary_at_time,
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org/route/v1"
USER_AGENT = "python-ai-prep-travel-agent/1.0 (local study project)"

Profile = Literal["driving", "walking", "cycling"]

_PROFILE_ALIASES: dict[str, Profile] = {
    "drive": "driving",
    "driving": "driving",
    "car": "driving",
    "auto": "driving",
    "walk": "walking",
    "walking": "walking",
    "foot": "walking",
    "on foot": "walking",
    "bike": "cycling",
    "bicycle": "cycling",
    "cycling": "cycling",
    "cycle": "cycling",
}


def _normalize_profile(mode: str) -> Profile | str:
    key = mode.strip().lower()
    if not key:
        return "driving"
    return _PROFILE_ALIASES.get(key, key)


def _geocode(client: httpx.Client, place: str) -> dict[str, Any] | str:
    return geocode(client, place)


@tool
def get_travel_time(origin: str, destination: str, mode: str = "driving") -> str:
    """Estimate how long it takes to travel between two places.

    Args:
        origin: Start location (city, address, landmark, or 'lat,lon').
        destination: End location (same formats as origin).
        mode: Travel mode — driving (default), walking, or cycling.

    Uses OpenStreetMap geocoding and OSRM open routing (no API key).
    """
    profile = _normalize_profile(mode)
    if profile not in ("driving", "walking", "cycling"):
        return (
            f"Unsupported mode '{mode}'. Use driving, walking, or cycling. "
            "Public transit is not available on the free routing service."
        )

    with httpx.Client() as client:
        origin_res = _geocode(client, origin)
        if isinstance(origin_res, str):
            return origin_res

        dest_res = _geocode(client, destination)
        if isinstance(dest_res, str):
            return dest_res

        o_lon, o_lat = origin_res["lon"], origin_res["lat"]
        d_lon, d_lat = dest_res["lon"], dest_res["lat"]
        coords = f"{o_lon},{o_lat};{d_lon},{d_lat}"
        url = f"{OSRM_URL}/{profile}/{coords}"

        try:
            resp = client.get(
                url,
                params={"overview": "false"},
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
        return f"Routing failed: {data.get('message', data.get('code', 'unknown error'))}"

    try:
        route = data["routes"][0]
        duration_s = float(route["duration"])
        distance_m = float(route["distance"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return f"Could not parse routing response: {e}"

    mode_label = {"driving": "by car", "walking": "on foot", "cycling": "by bicycle"}[profile]
    lines = [
        f"From: {origin_res['label']}",
        f"To: {dest_res['label']}",
        f"Mode: {profile} ({mode_label})",
        f"Estimated time: {format_duration(duration_s)}",
        f"Distance: {format_distance(distance_m)}",
        "Source: OSRM on OpenStreetMap data (approximate; not live traffic).",
    ]
    return "\n".join(lines)


@tool
def get_route_stops_with_weather(
    origin: str,
    destination: str,
    mode: str = "driving",
    start_time: str = "",
) -> str:
    """Trip plan with major towns between origin and destination, drive time to each, and weather at arrival.

    Args:
        origin: Start location.
        destination: End location.
        mode: driving (default), walking, or cycling.
        start_time: Departure from origin (e.g. '08:00', '9 AM', '2026-05-21 14:30').
            If omitted, assumes **8:00 AM today**.

    Excludes listing origin/destination as intermediate towns; shows estimated arrival at each town
    and wttr.in forecast for that time. Also shows total trip time and weather at destination on arrival.
    """
    profile = _normalize_profile(mode)
    if profile not in ("driving", "walking", "cycling"):
        return f"Unsupported mode '{mode}'. Use driving, walking, or cycling."

    depart = parse_start_time(start_time)

    with httpx.Client() as client:
        route_result = fetch_osrm_route(client, origin, destination, str(profile))
        if isinstance(route_result, str):
            return route_result
        route: OsrmRoute = route_result

        stops = discover_intermediate_stops(client, route, depart, max_towns=6)

    arrive_dest = depart + timedelta(seconds=route.duration_s)
    mode_label = {"driving": "by car", "walking": "on foot", "cycling": "by bicycle"}[profile]

    lines = [
        f"Route ({profile}, {mode_label}): {route.origin_label}",
        f"  → {route.dest_label}",
        f"Depart: {depart.strftime('%Y-%m-%d %H:%M')} (default 08:00 if start time not given)",
        f"Total: {format_duration(route.duration_s)}, {format_distance(route.distance_m)}",
        f"Estimated arrival at destination: {arrive_dest.strftime('%Y-%m-%d %H:%M')}",
        "",
        "Major towns along the way (excluding start and end):",
    ]

    if not stops:
        lines.append("  (No distinct intermediate towns identified on this short or direct route.)")
    else:
        for i, stop in enumerate(stops, 1):
            wx = weather_summary_at_time(stop.town, stop.arrival)
            lines.extend(
                [
                    f"{i}. {stop.town}",
                    f"   ~{format_duration(stop.duration_from_start_s)} from start "
                    f"({stop.distance_km:.0f} km along route)",
                    f"   Est. arrival: {format_time(stop.arrival)}",
                    f"   Weather then: {wx}",
                ]
            )

    dest_wx = weather_summary_at_time(destination, arrive_dest)
    lines.extend(
        [
            "",
            f"Destination ({route.dest_label.split(',')[0]}): arrive ~{format_time(arrive_dest)}",
            f"Weather at arrival: {dest_wx}",
            "",
            "Source: OSRM + Nominatim + wttr.in (estimates; no live traffic).",
        ]
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
        tools=[get_travel_time, get_route_stops_with_weather],
        system_prompt=(
            "You are a concise travel assistant. For simple A→B time questions use get_travel_time.\n"
            "When the user wants towns **between** origin and destination, weather along the way, "
            "or a trip itinerary with stops, use get_route_stops_with_weather with clear origin, "
            "destination, mode (default driving), and start_time when they give one.\n"
            "If they do not give a departure time, leave start_time empty (tool assumes 8:00 AM).\n"
            "Summarize tool output clearly. Times are OSRM estimates without live traffic."
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

    run_interactive("Travel-time agent", "ask about trips between places.", graph)


if __name__ == "__main__":
    main()
