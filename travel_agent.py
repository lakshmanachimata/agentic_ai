"""
LangChain agent (ReAct-style graph): Ollama (qwen2.5:latest) + travel time via OSM/Nominatim + OSRM.

Geocoding: Nominatim (OpenStreetMap) — https://nominatim.org
Routing: OSRM public demo — https://project-osrm.org

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:latest

Interactive mode keeps session memory across turns (``/reset`` clears it).

Run (interactive; Ctrl+D / EOF to exit):
  python travel_agent.py

One-off:
  python travel_agent.py "How long to drive from Paris to Lyon?"
"""

from __future__ import annotations

import json
import sys
from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive

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


def _format_duration(seconds: float) -> str:
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


def _format_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{int(round(meters))} m"


def _geocode(client: httpx.Client, place: str) -> dict[str, Any] | str:
    place = place.strip()
    if not place:
        return "Error: empty place name."

    try:
        resp = client.get(
            NOMINATIM_URL,
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
        return f"Could not find a location for '{place}'. Try a more specific name."

    hit = results[0]
    try:
        lat = float(hit["lat"])
        lon = float(hit["lon"])
    except (KeyError, TypeError, ValueError):
        return f"Geocoding response missing coordinates for '{place}'."

    label = hit.get("display_name", place)
    return {"lat": lat, "lon": lon, "label": label}


@tool
def get_travel_time(origin: str, destination: str, mode: str = "driving") -> str:
    """Estimate how long it takes to travel between two places.

    Args:
        origin: Start location (city, address, landmark, or 'lat,lon').
        destination: End location (same formats as origin).
        mode: Travel mode — driving (default), walking, or cycling.
              Aliases like 'drive', 'walk', 'bike' are accepted.

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
        f"Estimated time: {_format_duration(duration_s)}",
        f"Distance: {_format_distance(distance_m)}",
        "Source: OSRM on OpenStreetMap data (approximate; not live traffic).",
    ]
    return "\n".join(lines)


def build_agent():
    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[get_travel_time],
        system_prompt=(
            "You are a concise travel-time assistant. When the user asks how long "
            "it takes to get from one place to another, call get_travel_time with "
            "clear origin and destination strings and an appropriate mode "
            "(driving, walking, or cycling). Summarize the tool output in plain "
            "language. If the user does not specify a mode, assume driving. "
            "Mention that times are estimates without live traffic. "
            "You cannot route public transit with the available free APIs. "
            "Use earlier messages in this session when the user refers to places "
            "or modes without repeating them (e.g. 'walking instead?', 'back the other way')."
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
