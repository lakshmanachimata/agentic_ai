"""Shared helpers for agent entry points (invoke, session thread, interactive CLI)."""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from tracing_common import (
    agent_trace,
    begin_usage_span,
    finish_usage_span,
    format_token_usage,
    print_tracing_banner,
    usage_from_messages,
)

DEFAULT_OLLAMA_MODEL = "qwen3.5:latest"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 10


def make_chat_ollama(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    base_url: str | None = None,
) -> ChatOllama:
    """Build a ChatOllama client with optional GUI/CLI overrides."""
    kwargs: dict[str, Any] = {
        "model": (model or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL,
        "base_url": (base_url or DEFAULT_OLLAMA_BASE_URL).strip()
        or DEFAULT_OLLAMA_BASE_URL,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else float(temperature),
    }
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    return ChatOllama(**kwargs)


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
    agent_name: str = "agent",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Run one user turn on ``graph``.

    When ``thread_id`` is set, LangGraph merges this turn into the same
    conversation (session memory). When ``thread_id`` is omitted, a new
    ephemeral thread id is used so each call is isolated (e.g. one-shot CLI,
    or specialist tool calls from the orchestrator).

    ``agent_name`` becomes the LangSmith run name so orchestrator vs weather /
    travel / restaurants / calendar hops are easy to filter.
    """
    q = (question or "").strip()
    if not q:
        return ""

    tid = thread_id if thread_id is not None else uuid.uuid4().hex
    gui = os.environ.get("AGENTIC_AI_GUI", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    run_tags = [agent_name, "gui" if gui else "cli"]
    if tags:
        run_tags.extend(tags)
    run_meta = {"agent": agent_name, "thread_id": tid, **(metadata or {})}
    config: dict[str, Any] = {
        "configurable": {"thread_id": tid},
        "run_name": agent_name,
        "tags": run_tags,
        "metadata": run_meta,
    }

    reply = ""
    usage: dict[str, Any] = {}
    is_root = begin_usage_span()
    with agent_trace(
        agent_name,
        inputs={"question": q, "thread_id": tid},
        tags=run_tags,
        metadata=run_meta,
    ) as run:
        try:
            result = graph.invoke({"messages": [HumanMessage(content=q)]}, config)
            messages = result.get("messages", [])
            reply = extract_assistant_reply(messages)
            local_usage = usage_from_messages(messages)
            usage = finish_usage_span(
                local_usage,
                agent_name=agent_name,
                run=run,
                reply=reply,
                model=DEFAULT_OLLAMA_MODEL,
            )
        except Exception:
            finish_usage_span(
                usage_from_messages([]),
                agent_name=agent_name,
                run=run,
                reply="",
                model=DEFAULT_OLLAMA_MODEL,
            )
            raise
    if is_root and not gui and usage.get("llm_calls"):
        print(format_token_usage(usage), file=sys.stderr)
    return reply


def run_interactive(
    title: str,
    hint: str,
    graph: Any,
    *,
    agent_name: str = "agent",
) -> None:
    """Read-eval-print loop with session memory; type ``/reset`` to start a new thread."""
    session_tid = str(uuid.uuid4())
    print_tracing_banner()
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
            print(invoke_agent(graph, q, thread_id=session_tid, agent_name=agent_name))
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print()
