"""LangSmith tracing for every agent turn and downstream HTTP helper.

Set ``LANGSMITH_API_KEY`` (and optionally ``LANGSMITH_TRACING=true``) in the
environment or a ``.env`` file. Traces land in project ``agentic-ai`` by default.

https://smith.langchain.com
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

DEFAULT_LANGSMITH_PROJECT = "agentic-ai"

_STATUS: dict[str, Any] | None = None
_USAGE_STACK: ContextVar[list[list[dict[str, Any]]] | None] = ContextVar(
    "agentic_ai_usage_stack",
    default=None,
)
_LAST_USAGE: ContextVar[dict[str, Any] | None] = ContextVar(
    "agentic_ai_last_usage",
    default=None,
)


def _truthy(raw: str) -> bool | None:
    v = (raw or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def configure_langsmith() -> dict[str, Any]:
    """Load ``.env`` and enable LangSmith when an API key is present.

    ``LANGSMITH_TRACING=false`` always disables. Otherwise tracing turns on if
    ``LANGSMITH_TRACING=true`` or ``LANGSMITH_API_KEY`` / ``LANGCHAIN_API_KEY``
    is set.
    """
    global _STATUS
    if _STATUS is not None:
        return _STATUS

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = (
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or ""
    ).strip()
    tracing_flag = _truthy(
        os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    )
    if tracing_flag is False:
        enabled = False
    elif tracing_flag is True:
        enabled = bool(api_key)
    else:
        enabled = bool(api_key)

    project = (
        os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or DEFAULT_LANGSMITH_PROJECT
    ).strip() or DEFAULT_LANGSMITH_PROJECT

    if enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        try:
            import langsmith as ls

            ls.configure(enabled=True, project_name=project)
        except Exception:
            pass
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")

    _STATUS = {
        "enabled": enabled,
        "project": project,
        "has_key": bool(api_key),
    }
    return _STATUS


def tracing_status() -> dict[str, Any]:
    return configure_langsmith()


def tracing_banner() -> str:
    """One-line status for CLI stderr / Streamlit sidebar."""
    s = configure_langsmith()
    if s["enabled"]:
        return (
            f"LangSmith tracing ON · project `{s['project']}` · "
            "https://smith.langchain.com"
        )
    if _truthy(os.environ.get("LANGSMITH_TRACING") or "") is True and not s["has_key"]:
        return "LangSmith tracing requested but LANGSMITH_API_KEY is missing."
    return "LangSmith tracing OFF · set LANGSMITH_API_KEY to debug agent flow."


_BANNER_PRINTED = False


def print_tracing_banner() -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    if os.environ.get("AGENTIC_AI_GUI", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    print(tracing_banner(), file=sys.stderr)
    _BANNER_PRINTED = True


def drop_http_client(inputs: dict[str, Any]) -> dict[str, Any]:
    """Avoid serializing httpx.Client (and shrink OsrmRoute polylines) in traces."""
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if key == "client":
            continue
        if key == "route" and hasattr(value, "origin_label"):
            out[key] = {
                "origin": getattr(value, "origin_label", None),
                "destination": getattr(value, "dest_label", None),
                "profile": getattr(value, "profile", None),
                "duration_s": getattr(value, "duration_s", None),
                "distance_m": getattr(value, "distance_m", None),
            }
            continue
        out[key] = value
    return out


def empty_token_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "by_agent": {},
    }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _add_token_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": _as_int(left.get("input_tokens")) + _as_int(right.get("input_tokens")),
        "output_tokens": _as_int(left.get("output_tokens")) + _as_int(right.get("output_tokens")),
        "total_tokens": _as_int(left.get("total_tokens")) + _as_int(right.get("total_tokens")),
        "llm_calls": _as_int(left.get("llm_calls")) + _as_int(right.get("llm_calls")),
    }


def usage_from_message(message: Any) -> dict[str, Any] | None:
    """Pull Ollama/LangChain token counts off one AI message."""
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict) and any(
        um.get(k) not in (None, 0) for k in ("input_tokens", "output_tokens", "total_tokens")
    ):
        inp = _as_int(um.get("input_tokens"))
        out = _as_int(um.get("output_tokens"))
        tot = _as_int(um.get("total_tokens"), inp + out)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": tot or inp + out,
            "llm_calls": 1,
        }

    meta = getattr(message, "response_metadata", None) or {}
    if isinstance(meta, dict) and (
        "prompt_eval_count" in meta or "eval_count" in meta
    ):
        inp = _as_int(meta.get("prompt_eval_count"))
        out = _as_int(meta.get("eval_count"))
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "llm_calls": 1,
        }
    return None


def usage_from_messages(messages: list[Any] | None) -> dict[str, Any]:
    """Sum token usage across every LLM message in one agent turn."""
    total = empty_token_usage()
    for message in messages or []:
        part = usage_from_message(message)
        if part is None:
            continue
        summed = _add_token_usage(total, part)
        summed["by_agent"] = total["by_agent"]
        total = summed
    return total


def format_token_usage(usage: dict[str, Any] | None) -> str:
    """Human-readable token line for CLI / Streamlit / LangSmith outputs."""
    u = usage or empty_token_usage()
    line = (
        f"Tokens: {u.get('input_tokens', 0)} in · "
        f"{u.get('output_tokens', 0)} out · "
        f"{u.get('total_tokens', 0)} total · "
        f"{u.get('llm_calls', 0)} LLM calls"
    )
    by_agent = u.get("by_agent") or {}
    if len(by_agent) > 1:
        parts = [
            f"{name} {vals.get('total_tokens', 0)}"
            for name, vals in sorted(by_agent.items())
        ]
        line += " (" + ", ".join(parts) + ")"
    return line


def last_token_usage() -> dict[str, Any]:
    return _LAST_USAGE.get() or empty_token_usage()


def begin_usage_span() -> bool:
    """Push a collector for nested specialist token totals. Returns True if root turn."""
    stack = _USAGE_STACK.get()
    is_root = stack is None
    if stack is None:
        stack = []
        _USAGE_STACK.set(stack)
    stack.append([])
    return is_root


def finish_usage_span(
    local_usage: dict[str, Any],
    *,
    agent_name: str,
    run: Any = None,
    reply: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    """Attach tokens to this LangSmith span and roll them up to the parent agent."""
    stack = _USAGE_STACK.get() or []
    child_usages = stack.pop() if stack else []
    if not stack:
        _USAGE_STACK.set(None)

    local = {
        "input_tokens": _as_int(local_usage.get("input_tokens")),
        "output_tokens": _as_int(local_usage.get("output_tokens")),
        "total_tokens": _as_int(local_usage.get("total_tokens")),
        "llm_calls": _as_int(local_usage.get("llm_calls")),
    }
    by_agent: dict[str, Any] = {agent_name: dict(local)}
    combined = dict(local)
    for child in child_usages:
        combined = _add_token_usage(combined, child)
        for name, part in (child.get("by_agent") or {}).items():
            prev = by_agent.get(name) or {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
            }
            by_agent[name] = _add_token_usage(prev, part)
    combined["by_agent"] = by_agent

    usage_meta = {
        "input_tokens": combined["input_tokens"],
        "output_tokens": combined["output_tokens"],
        "total_tokens": combined["total_tokens"],
    }
    token_line = format_token_usage(combined)
    model_name = (model or os.environ.get("OLLAMA_MODEL") or "").strip() or None

    if run is not None:
        extra_meta: dict[str, Any] = {
            "tokens": combined,
            "token_summary": token_line,
            "ls_provider": "ollama",
        }
        if model_name:
            extra_meta["ls_model_name"] = model_name
        try:
            run.set(
                usage_metadata=usage_meta,
                metadata=extra_meta,
                outputs={
                    "reply": (reply or "")[:4000],
                    "usage_metadata": usage_meta,
                    "token_summary": token_line,
                    "tokens": combined,
                },
            )
        except Exception:
            try:
                run.extra.setdefault("metadata", {}).update(extra_meta)
                run.extra["metadata"]["usage_metadata"] = usage_meta
                run.outputs = {
                    "reply": (reply or "")[:4000],
                    "usage_metadata": usage_meta,
                    "token_summary": token_line,
                    "tokens": combined,
                }
            except Exception:
                pass

    if stack:
        stack[-1].append(combined)
    _LAST_USAGE.set(combined)
    return combined


try:
    from langsmith import traceable as traceable
except ImportError:  # pragma: no cover

    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap


@contextmanager
def agent_trace(
    name: str,
    *,
    inputs: dict[str, Any],
    tags: list[str],
    metadata: dict[str, Any],
) -> Iterator[Any]:
    """Named parent span around one user turn (orchestrator or specialist)."""
    configure_langsmith()
    try:
        from langsmith import trace
    except ImportError:
        yield None
        return
    with trace(
        name=name,
        run_type="chain",
        inputs=inputs,
        tags=tags,
        metadata=metadata,
    ) as run:
        yield run


configure_langsmith()
