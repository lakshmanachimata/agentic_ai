"""Typed trip context shared across orchestrator → specialist hops.

LangGraph message threads are *not* shared with nested specialists (each
delegation uses a fresh thread). This module holds structured fields
(``origin``, ``duration_s``, ``start_time``, …) keyed by the orchestrator
session so calendar/restaurants can read what travel just wrote, instead of
copying durations through free-text.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Any

from route_common import format_distance, format_duration

_SCOPE: ContextVar[str | None] = ContextVar("agentic_ai_trip_scope", default=None)
_TRIPS: dict[str, TripState] = {}
_DEFAULT_KEY = "__default__"

_DURATION_LINE = re.compile(
    r"(?:Estimated time|Total):\s*([^,\n]+)",
    re.IGNORECASE,
)


@dataclass
class TripState:
    origin: str = ""
    destination: str = ""
    mode: str = ""
    start_time: str = ""
    duration_s: float | None = None
    distance_m: float | None = None
    origin_label: str = ""
    dest_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_route(self) -> bool:
        return bool((self.origin or self.origin_label) and (self.destination or self.dest_label))

    def is_empty(self) -> bool:
        return not (
            self.origin
            or self.destination
            or self.origin_label
            or self.dest_label
            or self.start_time
            or self.duration_s
            or self.distance_m
        )

    def as_prompt_block(self) -> str:
        """Compact typed block injected into specialist prompts."""
        if self.is_empty():
            return ""
        lines: list[str] = []
        if self.origin:
            lines.append(f"origin: {self.origin}")
        if self.origin_label and self.origin_label != self.origin:
            lines.append(f"origin_label: {self.origin_label}")
        if self.destination:
            lines.append(f"destination: {self.destination}")
        if self.dest_label and self.dest_label != self.destination:
            lines.append(f"dest_label: {self.dest_label}")
        if self.mode:
            lines.append(f"mode: {self.mode}")
        if self.start_time:
            lines.append(f"start_time: {self.start_time}")
        if self.duration_s is not None and self.duration_s > 0:
            lines.append(
                f"duration_s: {int(round(self.duration_s))} "
                f"({format_duration(self.duration_s)})"
            )
        if self.distance_m is not None and self.distance_m > 0:
            lines.append(f"distance_m: {int(round(self.distance_m))} ({format_distance(self.distance_m)})")
        return "\n".join(lines)

    def for_trace(self) -> dict[str, Any]:
        """Compact payload for LangSmith metadata / inputs / outputs."""
        if self.is_empty():
            return {}
        out: dict[str, Any] = {}
        for key in (
            "origin",
            "destination",
            "mode",
            "start_time",
            "origin_label",
            "dest_label",
        ):
            val = getattr(self, key)
            if val:
                out[key] = val
        if self.duration_s is not None and self.duration_s > 0:
            out["duration_s"] = int(round(self.duration_s))
            out["duration_human"] = format_duration(self.duration_s)
        if self.distance_m is not None and self.distance_m > 0:
            out["distance_m"] = int(round(self.distance_m))
            out["distance_human"] = format_distance(self.distance_m)
        return out


def snapshot_for_trace() -> dict[str, Any]:
    """Trip fields + ``trip_id`` for LangSmith (empty dict if nothing to show)."""
    payload = get_trip().for_trace()
    if not payload:
        return {}
    tid = current_trip_id()
    if tid:
        payload = {**payload, "trip_id": tid}
    return payload


def _key() -> str:
    return _SCOPE.get() or _DEFAULT_KEY


def current_trip_id() -> str | None:
    return _SCOPE.get()


def begin_trip_scope(thread_id: str) -> Token[str | None]:
    """Bind this turn (and nested specialist calls) to ``thread_id``'s trip."""
    return _SCOPE.set(thread_id)


def reset_trip_scope(token: Token[str | None]) -> None:
    _SCOPE.reset(token)


def get_trip() -> TripState:
    return _TRIPS.setdefault(_key(), TripState())


def clear_trip(thread_id: str | None = None) -> None:
    key = thread_id if thread_id is not None else _key()
    _TRIPS.pop(key, None)


def clear_all_trips() -> None:
    _TRIPS.clear()
    _SCOPE.set(None)


def update_trip(**fields: Any) -> TripState:
    """Merge non-empty fields into the current trip.

    Changing origin/destination/mode without a new ``duration_s`` clears the
    previous duration so calendar cannot reuse a stale drive time.
    """
    trip = get_trip()
    route_keys = ("origin", "destination", "mode")
    route_changed = False
    for name in route_keys:
        if name not in fields:
            continue
        raw = fields[name]
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        old = str(getattr(trip, name) or "").strip()
        if old and old.lower() != text.lower():
            route_changed = True
        setattr(trip, name, text)

    if route_changed and "duration_s" not in fields:
        trip.duration_s = None
        trip.distance_m = None

    for f in fields:
        if f in route_keys:
            continue
        if not hasattr(trip, f):
            continue
        value = fields[f]
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            setattr(trip, f, text)
            continue
        setattr(trip, f, value)
    return trip


def fill_route_args(
    origin: str = "",
    destination: str = "",
    mode: str = "",
    start_time: str = "",
) -> tuple[str, str, str, str]:
    """Prefer explicit tool args; fall back to typed trip state."""
    trip = get_trip()
    origin_out = (origin or "").strip() or trip.origin or trip.origin_label
    dest_out = (destination or "").strip() or trip.destination or trip.dest_label
    mode_out = (mode or "").strip() or trip.mode or "driving"
    start_out = (start_time or "").strip() or trip.start_time
    return origin_out, dest_out, mode_out, start_out


def harvest_travel_metrics(text: str) -> TripState:
    """Best-effort parse of travel-agent prose when tools did not write duration."""
    trip = get_trip()
    if trip.duration_s:
        return trip
    m = _DURATION_LINE.search(text or "")
    if not m:
        return trip
    chunk = m.group(1).strip().lower()
    hours = minutes = seconds = 0
    hm = re.search(r"(\d+)\s*hrs?", chunk)
    mm = re.search(r"(\d+)\s*min", chunk)
    sm = re.search(r"(\d+)\s*sec", chunk)
    if hm:
        hours = int(hm.group(1))
    if mm:
        minutes = int(mm.group(1))
    if sm and not hours and not minutes:
        seconds = int(sm.group(1))
    total = hours * 3600 + minutes * 60 + seconds
    if total > 0:
        trip.duration_s = float(total)
    return trip


def compose_specialist_query(query: str, *, specialist: str) -> str:
    """Attach typed trip facts so nested ReAct agents do not re-guess them."""
    q = (query or "").strip()
    block = get_trip().as_prompt_block()
    if not block:
        return q
    extra = (
        "Typed trip state (authoritative; do not invent origin, times, or duration):\n"
        f"{block}"
    )
    if specialist == "calendar":
        extra += (
            "\nFor a road trip call create_travel_calendar with origin, destination, "
            "and start_time from this state. Leave duration empty — OSRM or duration_s "
            "is already known. Never invent 4h/6h."
        )
    elif specialist == "travel":
        extra += "\nUse these origin/destination/mode/start_time values when calling tools."
    elif specialist == "restaurants":
        extra += (
            "\nIf the user wants food along this trip, use origin/destination/start_time "
            "from this state."
        )
    if q:
        return f"{q}\n\n{extra}"
    return extra
