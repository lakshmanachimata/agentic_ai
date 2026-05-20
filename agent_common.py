"""Shared helpers for agent entry points (invoke, session thread, interactive CLI)."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def extract_assistant_reply(messages: list[Any]) -> str:
    """Return the last assistant text from a graph ``messages`` list."""
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, AIMessage):
        return (last.content or "").strip()
    return str(getattr(last, "content", last))


def invoke_agent(
    graph: Any,
    question: str,
    *,
    thread_id: str | None = None,
) -> str:
    """Run one user turn on ``graph``.

    When ``thread_id`` is set, LangGraph merges this turn into the same
    conversation (session memory). When ``thread_id`` is omitted, a new
    ephemeral thread id is used so each call is isolated (e.g. one-shot CLI,
    or specialist tool calls from the orchestrator).
    """
    q = (question or "").strip()
    if not q:
        return ""

    tid = thread_id if thread_id is not None else uuid.uuid4().hex
    config: dict[str, Any] = {"configurable": {"thread_id": tid}}
    result = graph.invoke({"messages": [HumanMessage(content=q)]}, config)
    messages = result.get("messages", [])
    return extract_assistant_reply(messages)


def run_interactive(title: str, hint: str, graph: Any) -> None:
    """Read-eval-print loop with session memory; type ``/reset`` to start a new thread."""
    session_tid = str(uuid.uuid4())
    print(f"{title} — {hint} Session memory is on. Type /reset to clear context. Ctrl+D (EOF) to exit.")
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
        if q.lower() in ("/reset", "/clear", "reset"):
            session_tid = str(uuid.uuid4())
            print("(session cleared — new thread)\n")
            continue
        try:
            print(invoke_agent(graph, q, thread_id=session_tid))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print()
