#!/usr/bin/env python3
"""
Unified launcher for the multi-agent assistant.

By default, starts the orchestrator client. It accepts natural-language queries
and delegates to the weather, travel-time, and dining specialist agents as needed.

Interactive mode uses session memory (same thread) so follow-up questions keep
context; type ``/reset`` to clear. Multiple ``-q`` flags share one session.

Requires Ollama running locally with the model pulled:
  ollama pull qwen3.5:4b

Examples:
  python run_agents.py
  python run_agents.py "Weather in Rome and driving time from Milan to Rome"
  python run_agents.py --client weather "What is the forecast in Tokyo?"
  python run_agents.py --client travel "How long to walk from Central Park to Times Square?"
  python run_agents.py -c restaurants "Casual food near London Bridge"
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from typing import Any

from agent_common import invoke_agent, run_interactive
from orchestrator_agent import build_agent as build_orchestrator
from restaurant_agent import build_agent as build_restaurant
from travel_agent import build_agent as build_travel
from weather_agent import build_agent as build_weather

ClientBuilder = Callable[[], Any]

_CLIENTS: dict[str, tuple[str, str, ClientBuilder]] = {
    "orchestrator": (
        "Orchestrator",
        "ask about weather, travel time, dining, or combine them (routes to specialists automatically)",
        build_orchestrator,
    ),
    "restaurants": (
        "Restaurant / dining specialist",
        "ask for restaurants, cafés, or food near an area (OpenStreetMap; free)",
        build_restaurant,
    ),
    "weather": (
        "Weather specialist",
        "ask about current conditions and forecasts for any place",
        build_weather,
    ),
    "travel": (
        "Travel-time specialist",
        "ask how long it takes to travel between places",
        build_travel,
    ),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an agent client and run one or more queries.",
    )
    parser.add_argument(
        "-c",
        "--client",
        choices=sorted(_CLIENTS),
        default="orchestrator",
        help="which client to run (default: orchestrator)",
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="queries",
        action="append",
        default=[],
        metavar="TEXT",
        help="query text (repeatable); omit for interactive mode",
    )
    parser.add_argument(
        "positional_query",
        nargs="*",
        help="query words if -q/--query is not used",
    )
    return parser.parse_args(argv)


def _collect_queries(args: argparse.Namespace) -> list[str]:
    from_flags = [q.strip() for q in args.queries if q and q.strip()]
    if from_flags:
        return from_flags
    joined = " ".join(args.positional_query).strip()
    return [joined] if joined else []


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    name, hint, build = _CLIENTS[args.client]

    print(f"Starting {name.lower()} client…", file=sys.stderr)
    graph = build()

    queries = _collect_queries(args)
    if queries:
        # One shared thread when running several queries in one process invocation.
        tid = str(uuid.uuid4()) if len(queries) > 1 else None
        for question in queries:
            print(invoke_agent(graph, question, thread_id=tid))
        return 0

    run_interactive(name, hint, graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
