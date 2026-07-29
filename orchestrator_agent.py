"""
Multi-agent orchestrator: routes each query to weather, travel-time, dining,
and travel-calendar specialists.

Specialists are the full ReAct agents from weather_agent.py, travel_agent.py,
restaurant_agent.py, and calendar_agent.py, invoked as tools so the orchestrator
can delegate as needed.

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:7b

Interactive mode keeps session memory across turns (``/reset`` clears it).

Run (interactive; Ctrl+D / EOF to exit):
  python orchestrator_agent.py

One-off:
  python orchestrator_agent.py "Weather in Rome, drive time from Milan to Rome, and cafés near the Colosseum"
"""

from __future__ import annotations

import sys
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive

from calendar_agent import build_agent as build_calendar_agent
from calendar_agent import run_query as run_calendar_query
from restaurant_agent import build_agent as build_restaurant_agent
from restaurant_agent import run_query as run_restaurant_query
from travel_agent import build_agent as build_travel_agent
from travel_agent import run_query as run_travel_query
from weather_agent import build_agent as build_weather_agent
from weather_agent import run_query as run_weather_query


def build_agent():
    weather_graph = build_weather_agent()
    travel_graph = build_travel_agent()
    restaurant_graph = build_restaurant_agent()
    calendar_graph = build_calendar_agent()

    @tool
    def ask_weather_specialist(query: str) -> str:
        """Delegate to the weather specialist agent.

        Use for current conditions, forecasts, temperature, rain, wind, or
        humidity at any place. Pass a clear, self-contained question with
        location names (e.g. 'What is the weather in Tokyo today?').
        """
        return run_weather_query(weather_graph, query.strip())

    @tool
    def ask_travel_specialist(query: str) -> str:
        """Delegate to the travel-time specialist agent.

        Use for travel time, major towns between origin and destination, estimated
        arrival at each town, and weather at those times. Include start_time from
        origin when the user gives it (otherwise specialist assumes 8:00 AM).
        Pass origin, destination, and mode (e.g. 'Drive Boston to NYC leaving 9 AM,
        towns and weather along the way').
        """
        return run_travel_query(travel_graph, query.strip())

    @tool
    def ask_restaurant_specialist(query: str) -> str:
        """Delegate to the dining / restaurant specialist agent.

        Use for dining near an area, or **at intermediate towns** on a route between two places
        (excludes origin/destination; pass start_time if user gives departure, else 8 AM default).
        (e.g. 'Restaurants in towns between Boston and Portland ME' or 'Italian food near Le Marais').
        """
        return run_restaurant_query(restaurant_graph, query.strip())

    @tool
    def ask_calendar_specialist(query: str) -> str:
        """Delegate to the travel-calendar specialist agent.

        Use when the user wants to create a calendar event, add a trip to Google
        Calendar, save an .ics file, or list saved calendar files. Pass a clear
        self-contained request with title/places, start time, and duration when known
        (e.g. 'Create calendar event Drive Paris to Lyon tomorrow 9 AM lasting 4 hours').
        """
        return run_calendar_query(calendar_graph, query.strip())

    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[
            ask_weather_specialist,
            ask_travel_specialist,
            ask_restaurant_specialist,
            ask_calendar_specialist,
        ],
        system_prompt=(
            "You are a coordinator for weather, travel-time, dining, and travel-calendar "
            "specialists. Never invent weather, routing, venue, or calendar file data — "
            "always use the tools.\n"
            "- Weather only → call ask_weather_specialist once with a focused question.\n"
            "- Travel time / route only → call ask_travel_specialist once.\n"
            "- Dining / restaurants / cafés near an area **or along a route** → "
            "call ask_restaurant_specialist once with a focused question.\n"
            "- Calendar / create event / Google Calendar / .ics / add trip to calendar → "
            "call ask_calendar_specialist once with title or origin/destination, times, duration.\n"
            "- Combine as needed (e.g. look up drive time, then create a calendar event for "
            "that trip); you may call several tools in the same turn.\n"
            "Rewrite the user's request into a clear sub-question for each specialist. "
            "After tool results return, give one concise combined answer.\n"
            "Use prior turns in this session when the user refers to places, routes, "
            "or earlier answers (e.g. 'there', 'same trip', 'add that to my calendar')."
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
        "Orchestrator",
        "ask about weather, travel time, places to eat, calendar events, or combine them.",
        graph,
    )


if __name__ == "__main__":
    main()
