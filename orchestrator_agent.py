"""
Multi-agent orchestrator: routes each query to the weather and/or travel specialists.

Specialists are the full ReAct agents from weather_agent.py and travel_agent.py,
invoked as tools so the orchestrator can call one, the other, or both.

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:latest

Run (interactive; Ctrl+D / EOF to exit):
  python orchestrator_agent.py

One-off:
  python orchestrator_agent.py "Weather in Rome and drive time from Milan to Rome"
"""

from __future__ import annotations

import sys
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from travel_agent import build_agent as build_travel_agent
from travel_agent import run_query as run_travel_query
from weather_agent import build_agent as build_weather_agent
from weather_agent import run_query as run_weather_query


def build_agent():
    weather_graph = build_weather_agent()
    travel_graph = build_travel_agent()

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

        Use for how long it takes to go between places, route distance, or
        driving/walking/cycling duration. Pass origin, destination, and mode
        when known (e.g. 'Driving time from Boston to New York').
        """
        return run_travel_query(travel_graph, query.strip())

    llm = ChatOllama(
        model="qwen2.5:latest",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[ask_weather_specialist, ask_travel_specialist],
        system_prompt=(
            "You are a coordinator for weather and travel-time specialists. "
            "Never invent weather or routing data — always use the tools.\n"
            "- Weather only → call ask_weather_specialist once with a focused question.\n"
            "- Travel time / route only → call ask_travel_specialist once.\n"
            "- Both (e.g. weather at a destination AND how to get there) → call "
            "the relevant tool(s); you may call both in the same turn.\n"
            "Rewrite the user's request into a clear sub-question for each specialist. "
            "After tool results return, give one concise combined answer."
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

    print(
        "Orchestrator — ask about weather, travel time, or both. Ctrl+D (EOF) to exit."
    )
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
