"""Shared CLI helpers for agent entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_interactive(
    title: str,
    hint: str,
    graph: Any,
    run_query: Callable[[Any, str], str],
) -> None:
    print(f"{title} — {hint} Ctrl+D (EOF) to exit.")
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
