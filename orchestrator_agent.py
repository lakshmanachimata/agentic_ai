"""
Multi-agent orchestrator: routes each query to weather, travel-time, dining,
and travel-calendar specialists.

Specialists are the full ReAct agents from weather_agent.py, travel_agent.py,
restaurant_agent.py, and calendar_agent.py, invoked as tools so the orchestrator
can delegate as needed.

Requires Ollama running locally with the model pulled:
  ollama pull qwen3.5:latest

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
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, make_chat_ollama, run_interactive

from calendar_agent import build_agent as build_calendar_agent
from calendar_agent import run_query as run_calendar_query
from restaurant_agent import build_agent as build_restaurant_agent
from restaurant_agent import run_query as run_restaurant_query
from travel_agent import build_agent as build_travel_agent
from travel_agent import run_query as run_travel_query
from weather_agent import build_agent as build_weather_agent
from weather_agent import run_query as run_weather_query


def build_agent(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
):
    llm_kwargs = {"model": model, "temperature": temperature, "top_k": top_k}
    weather_graph = build_weather_agent(**llm_kwargs)
    travel_graph = build_travel_agent(**llm_kwargs)
    restaurant_graph = build_restaurant_agent(**llm_kwargs)
    calendar_graph = build_calendar_agent(**llm_kwargs)

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
        Calendar, save an .ics file, or list saved calendar files.

        Pass a self-contained request that ALWAYS includes:
        - origin and destination for trips
        - start day+time exactly as the user said (e.g. 'tomorrow 10:00 AM')
        - if you already called ask_travel_specialist, include that exact duration
          (e.g. '8 hours 47 minutes') in the query for reference — the calendar tool
          will still compute OSRM duration for road trips
        Never invent a shorter duration like 4h or 6h when travel reported longer.
        """
        return run_calendar_query(calendar_graph, query.strip())

    llm = make_chat_ollama(model=model, temperature=temperature, top_k=top_k)
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
            "specialists. Never invent weather, routing, venue, calendar times, or durations — "
            "always use the tools.\n"
            "- Weather only → ask_weather_specialist.\n"
            "- Travel time / route / plan a road trip → ask_travel_specialist first.\n"
            "- Dining → ask_restaurant_specialist.\n"
            "- Calendar / add trip to calendar / .ics → ask_calendar_specialist.\n"
            "- Plan trip AND add to calendar: (1) ask_travel_specialist for the route, "
            "(2) then ask_calendar_specialist with origin, destination, mode, and the "
            "user's start including the day word ('tomorrow 10:00 AM'). "
            "Do not invent duration; leave timing to the calendar/travel tools. "
            "You may call both in the same turn only if travel can finish first — "
            "prefer sequential: travel, then calendar with the same places/times.\n"
            "Rewrite into clear sub-questions. After tools return, give one concise answer "
            "using the calendar tool's actual start/end datetimes (do not restate a wrong "
            "guessed window like 10 AM–4 PM if the tool said otherwise).\n"
            "Use prior turns for coreferences ('there', 'same trip', 'add that to my calendar')."
        ),
        checkpointer=MemorySaver(),
    )


def run_query(graph: Any, question: str, *, thread_id: str | None = None) -> str:
    return invoke_agent(graph, question, thread_id=thread_id, agent_name="orchestrator")


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
        agent_name="orchestrator",
    )


if __name__ == "__main__":
    main()
