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
from guardrails_common import screen_specialist_hop
from trip_state import compose_specialist_query, harvest_travel_metrics, update_trip


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
    def ask_weather_specialist(query: str, location: str = "") -> str:
        """Delegate to the weather specialist agent.

        Use for current conditions, forecasts, temperature, rain, wind, or
        humidity at any place. Pass a clear, self-contained question with
        location names (e.g. 'What is the weather in Tokyo today?').
        Prefer the location field when you already know the place.
        """
        text = query.strip() if query.strip() else (location or "")
        blocked = screen_specialist_hop(text, "weather")
        if blocked:
            return blocked
        update_trip(destination=location)
        composed = compose_specialist_query(text, specialist="weather")
        return run_weather_query(weather_graph, composed)

    @tool
    def ask_travel_specialist(
        query: str,
        origin: str = "",
        destination: str = "",
        mode: str = "driving",
        start_time: str = "",
    ) -> str:
        """Delegate to the travel-time specialist agent.

        Fill origin, destination, mode, and start_time as structured fields
        (not only inside query). Duration is written back to typed trip state
        for calendar/restaurants — do not invent a drive time.

        Use for travel time, major towns between origin and destination, estimated
        arrival at each town, and weather at those times. Include start_time from
        origin when the user gives it (otherwise specialist assumes 8:00 AM).
        """
        blocked = screen_specialist_hop(query, "travel")
        if blocked:
            return blocked
        update_trip(
            origin=origin,
            destination=destination,
            mode=mode,
            start_time=start_time,
        )
        composed = compose_specialist_query(query, specialist="travel")
        reply = run_travel_query(travel_graph, composed)
        harvest_travel_metrics(reply)
        return reply

    @tool
    def ask_restaurant_specialist(
        query: str,
        origin: str = "",
        destination: str = "",
        start_time: str = "",
        area: str = "",
    ) -> str:
        """Delegate to the dining / restaurant specialist agent.

        Use for dining near an area, or at intermediate towns on a route.
        Pass origin/destination/start_time as fields when this is a trip query.
        """
        text = query.strip() if query.strip() else area
        blocked = screen_specialist_hop(text, "restaurants")
        if blocked:
            return blocked
        update_trip(
            origin=origin,
            destination=destination,
            start_time=start_time,
        )
        composed = compose_specialist_query(text, specialist="restaurants")
        return run_restaurant_query(restaurant_graph, composed)

    @tool
    def ask_calendar_specialist(
        query: str,
        origin: str = "",
        destination: str = "",
        start_time: str = "",
        mode: str = "",
    ) -> str:
        """Delegate to the travel-calendar specialist agent.

        Pass origin, destination, and start_time as structured fields.
        Do NOT pass or invent duration — typed trip state / OSRM owns it.

        Use when the user wants to create a calendar event, add a trip to Google
        Calendar, save an .ics file, or list saved calendar files.
        """
        blocked = screen_specialist_hop(query, "calendar")
        if blocked:
            return blocked
        update_trip(
            origin=origin,
            destination=destination,
            start_time=start_time,
            mode=mode,
        )
        composed = compose_specialist_query(query, specialist="calendar")
        return run_calendar_query(calendar_graph, composed)

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
            "When calling specialists, fill structured fields (origin, destination, "
            "start_time, mode) in addition to query. Typed trip state is shared: after "
            "ask_travel_specialist, duration_s is already stored — do not copy or invent it.\n"
            "- Weather only → ask_weather_specialist.\n"
            "- Travel time / route / plan a road trip → ask_travel_specialist first.\n"
            "- Dining → ask_restaurant_specialist (pass origin/destination if along a trip).\n"
            "- Calendar / add trip to calendar / .ics → ask_calendar_specialist with "
            "origin, destination, and the user's start_time including the day word.\n"
            "- Plan trip AND add to calendar: (1) ask_travel_specialist, (2) ask_calendar_specialist "
            "with the same origin/destination/start_time fields. Leave duration to trip state/OSRM.\n"
            "Rewrite into clear sub-questions. After tools return, give one concise answer "
            "using the calendar tool's actual start/end datetimes (do not restate a wrong "
            "guessed window like 10 AM–4 PM if the tool said otherwise).\n"
            "Use prior turns and typed trip state for coreferences ('there', 'same trip', "
            "'add that to my calendar').\n"
            "Do not send a dining/calendar question to the weather specialist (or similar "
            "cross-routing); tools refuse misrouted hops."
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
