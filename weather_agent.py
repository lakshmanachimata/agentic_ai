"""
LangChain agent (ReAct-style graph): Ollama (qwen2.5:latest) + worldwide weather via wttr.in.

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:latest

Run (interactive; Ctrl+D / EOF to exit):
  python weather_agent.py

One-off:
  python weather_agent.py "What's the weather in Paris?"
"""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def _wttr_value(obj: Any) -> str:
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "value" in obj[0]:
        return str(obj[0]["value"])
    return str(obj)


@tool
def get_weather(location: str) -> str:
    """Look up current weather and a short forecast for any place on Earth.

    Pass a clear location string: city name, region, country, landmark, or
    coordinates like '48.85,2.35'. Use English or local spelling as commonly used.
    """
    loc = location.strip()
    if not loc:
        return "Error: empty location."

    url = f"https://wttr.in/{quote(loc, safe='')}?format=j1"
    headers = {"User-Agent": "curl/8.0 (weather-agent; +https://github.com/chubin/wttr.in)"}

    try:
        resp = httpx.get(url, headers=headers, timeout=45.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return f"Weather service HTTP error: {e}"
    except json.JSONDecodeError:
        return "Weather service returned invalid data."

    try:
        cur = data["current_condition"][0]
        area = data["nearest_area"][0]
        place = _wttr_value(area.get("areaName", []))
        region = _wttr_value(area.get("region", []))
        country = _wttr_value(area.get("country", []))
        desc = _wttr_value(cur.get("weatherDesc", [{}]))

        today = data.get("weather", [{}])[0]
        max_c = today.get("maxtempC", "?")
        min_c = today.get("mintempC", "?")
        date = today.get("date", "")

        lines = [
            f"Resolved area: {place}, {region}, {country}".strip().replace(", ,", ","),
            f"Now: {desc}, {cur.get('temp_C', '?')}°C (feels {cur.get('FeelsLikeC', '?')}°C)",
            f"Wind: {cur.get('windspeedKmph', '?')} km/h {cur.get('winddir16Point', '')}, "
            f"humidity {cur.get('humidity', '?')}%, pressure {cur.get('pressure', '?')} mb",
        ]
        if date:
            lines.append(f"Today ({date}): high {max_c}°C, low {min_c}°C")
        return "\n".join(lines)
    except (KeyError, IndexError, TypeError) as e:
        return f"Could not parse weather response: {e}"


def build_agent():
    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[get_weather],
        system_prompt=(
            "You are a concise weather assistant. When the user asks about weather "
            "for any place, call get_weather with a specific location string. "
            "Summarize the tool output clearly; mention resolved area if it differs "
            "from what the user said. Use metric units as returned by the tool."
        ),
    )


def run_query(graph: Any, question: str) -> str:
    result = graph.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", [])
    if not messages:
        return ""

    last = messages[-1]
    if isinstance(last, AIMessage):
        return (last.content or "").strip()
    return str(getattr(last, "content", last))


def main() -> None:
    graph = build_agent()
    q_one = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if q_one:
        print(run_query(graph, q_one))
        return

    print("Weather agent — ask about any place. Ctrl+D (EOF) to exit.")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if not q:
            continue
        try:
            print(run_query(graph, q))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print()


if __name__ == "__main__":
    main()
