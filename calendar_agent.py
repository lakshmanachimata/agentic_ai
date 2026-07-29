"""
LangChain agent (ReAct-style graph): travel / calendar events.

Always creates:
  1) An ``.ics`` (iCalendar) file under ``calendar_events/`` — import into
     Google Calendar via Settings → Import, or open the file on many devices.
  2) A Google Calendar "Add event" TEMPLATE link (no API key / OAuth).

Optionally, if Google Calendar OAuth credentials are set up and the optional
packages are installed, also inserts the event into the user's calendar via API:
  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
  Place OAuth desktop client JSON as ``credentials.json`` in this project folder.
  First run opens a browser to authorize; token is saved as ``token.json``.

Requires Ollama running locally with the model pulled:
  ollama pull qwen2.5:7b

Interactive:
  python calendar_agent.py

One-off:
  python calendar_agent.py "Create a calendar event: Drive Paris to Lyon tomorrow 9 AM, 4 hours"
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, run_interactive
from route_common import parse_start_time

EVENTS_DIR = Path(__file__).resolve().parent / "calendar_events"
CREDENTIALS_PATH = Path(__file__).resolve().parent / "credentials.json"
TOKEN_PATH = Path(__file__).resolve().parent / "token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _escape_ics_text(value: str) -> str:
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _fold_ics_line(line: str) -> str:
    """RFC 5545 line folding at 75 octets (approx ASCII-safe)."""
    if len(line) <= 75:
        return line
    parts = [line[:75]]
    rest = line[75:]
    while rest:
        parts.append(" " + rest[:74])
        rest = rest[74:]
    return "\r\n".join(parts)


def _fmt_ics_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def _fmt_gcal_local(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def _parse_event_start(start_time: str) -> datetime:
    """Parse start time; supports today/tomorrow prefixes then route_common rules."""
    raw = (start_time or "").strip()
    lowered = raw.lower()
    day_offset = 0
    if lowered.startswith("tomorrow"):
        day_offset = 1
        raw = raw[8:].lstrip(" ,:-")
    elif lowered.startswith("today"):
        raw = raw[5:].lstrip(" ,:-")
    base = parse_start_time(raw)
    if day_offset:
        base = base + timedelta(days=day_offset)
    return base


def _parse_duration_minutes(duration: str, default: int = 60) -> int:
    raw = (duration or "").strip().lower()
    if not raw:
        return default
    if raw.isdigit():
        return max(1, int(raw))
    total = 0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)", raw):
        n = float(amount)
        if unit.startswith("h"):
            total += int(round(n * 60))
        else:
            total += int(round(n))
    if total > 0:
        return total
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return default


def _safe_filename(title: str) -> str:
    base = re.sub(r"[^\w\-.]+", "_", (title or "event").strip())[:60].strip("_")
    return base or "event"


def _google_template_url(
    title: str,
    start: datetime,
    end: datetime,
    location: str = "",
    description: str = "",
) -> str:
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{_fmt_gcal_local(start)}/{_fmt_gcal_local(end)}",
    }
    if location:
        params["location"] = location
    if description:
        params["details"] = description
    return "https://calendar.google.com/calendar/render?" + urlencode(params, quote_via=quote)


def _build_ics(
    *,
    title: str,
    start: datetime,
    end: datetime,
    location: str = "",
    description: str = "",
    uid: str | None = None,
) -> str:
    event_uid = uid or f"{uuid.uuid4().hex}@agentic_ai"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//agentic_ai//travel-calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{event_uid}",
        f"DTSTAMP:{stamp}",
        f"DTSTART:{_fmt_ics_local(start)}",
        f"DTEND:{_fmt_ics_local(end)}",
        f"SUMMARY:{_escape_ics_text(title)}",
    ]
    if location:
        lines.append(f"LOCATION:{_escape_ics_text(location)}")
    if description:
        lines.append(f"DESCRIPTION:{_escape_ics_text(description)}")
    lines.extend(["END:VEVENT", "END:VCALENDAR"])
    return "\r\n".join(_fold_ics_line(line) for line in lines) + "\r\n"


def _write_ics(title: str, ics_body: str) -> Path:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVENTS_DIR / f"{stamp}_{_safe_filename(title)}.ics"
    path.write_text(ics_body, encoding="utf-8")
    return path


def _try_google_api_insert(
    *,
    title: str,
    start: datetime,
    end: datetime,
    location: str = "",
    description: str = "",
) -> str | None:
    """Insert via Google Calendar API if creds + packages exist; else None."""
    enabled = os.environ.get("GOOGLE_CALENDAR_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled and not CREDENTIALS_PATH.is_file() and not TOKEN_PATH.is_file():
        return None

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        return (
            "Google Calendar API skipped: install optional packages with\n"
            "  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
        )

    if not CREDENTIALS_PATH.is_file() and not TOKEN_PATH.is_file():
        return (
            "Google Calendar API skipped: place OAuth desktop client JSON as "
            f"{CREDENTIALS_PATH.name} (or set GOOGLE_CALENDAR_ENABLED only after setup)."
        )

    try:
        creds: Any = None
        if TOKEN_PATH.is_file():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CREDENTIALS_PATH.is_file():
                    return (
                        "Google Calendar API skipped: need credentials.json to authorize "
                        "(Google Cloud → OAuth client → Desktop app)."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
                creds = flow.run_local_server(port=0)
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        local_tz = datetime.now().astimezone().tzinfo
        start_aware = start.replace(tzinfo=local_tz) if start.tzinfo is None else start
        end_aware = end.replace(tzinfo=local_tz) if end.tzinfo is None else end
        body = {
            "summary": title,
            "location": location or None,
            "description": description or None,
            "start": {"dateTime": start_aware.isoformat()},
            "end": {"dateTime": end_aware.isoformat()},
        }
        tz_name = os.environ.get("TZ") or getattr(local_tz, "key", None)
        if tz_name:
            body["start"]["timeZone"] = tz_name
            body["end"]["timeZone"] = tz_name

        created = (
            service.events()
            .insert(calendarId="primary", body=body)
            .execute()
        )
        link = created.get("htmlLink") or "(no htmlLink returned)"
        eid = created.get("id", "?")
        return f"Google Calendar API: created event id={eid}\nOpen: {link}"
    except Exception as e:
        return f"Google Calendar API error: {e}"


def _create_event_bundle(
    *,
    title: str,
    start: datetime,
    end: datetime,
    location: str = "",
    description: str = "",
) -> str:
    ics = _build_ics(
        title=title,
        start=start,
        end=end,
        location=location,
        description=description,
    )
    path = _write_ics(title, ics)
    gcal_url = _google_template_url(title, start, end, location, description)
    api_note = _try_google_api_insert(
        title=title,
        start=start,
        end=end,
        location=location,
        description=description,
    )

    lines = [
        f"Title: {title}",
        f"When: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} (local)",
        f"Location: {location or '(none)'}",
        f"ICS file (import into Google Calendar): {path}",
        "  Google Calendar → Settings → Import & export → Import → choose the .ics file",
        f"Quick-add link (opens Google Calendar form): {gcal_url}",
    ]
    if description:
        lines.insert(3, f"Notes: {description[:500]}")
    if api_note:
        lines.append(api_note)
    else:
        lines.append(
            "API push: not used (no credentials.json/token.json). "
            "ICS + quick-add link still work without Google API setup."
        )
    return "\n".join(lines)


@tool
def create_calendar_event(
    title: str,
    start_time: str,
    duration: str = "1 hour",
    location: str = "",
    description: str = "",
    end_time: str = "",
) -> str:
    """Create one calendar event as an .ics file plus a Google Calendar add link.

    Also tries Google Calendar API insert when credentials.json / token.json exist.

    Args:
        title: Event title (e.g. 'Drive Paris to Lyon').
        start_time: Start — '9:00 AM', '2026-07-30 09:00', '14:30', etc.
        duration: Length if end_time empty — '90', '90 min', '2 hours' (default 1 hour).
        location: Optional place / address.
        description: Optional notes (route, passengers, packing list).
        end_time: Optional explicit end; overrides duration when set.
    """
    title = (title or "").strip()
    if not title:
        return "Error: title is required."

    start = _parse_event_start(start_time)
    if (end_time or "").strip():
        end = _parse_event_start(end_time)
        if end <= start:
            # Prefer duration when end would be before start (e.g. same-day clock confusion).
            end = start + timedelta(minutes=_parse_duration_minutes(duration, 60))
    else:
        end = start + timedelta(minutes=_parse_duration_minutes(duration, 60))

    return _create_event_bundle(
        title=title,
        start=start,
        end=end,
        location=(location or "").strip(),
        description=(description or "").strip(),
    )


@tool
def create_travel_calendar(
    origin: str,
    destination: str,
    start_time: str = "",
    duration: str = "4 hours",
    mode: str = "driving",
    notes: str = "",
) -> str:
    """Create a trip calendar event (departure → arrival) as .ics + Google add link.

    Use when the user wants to put travel on their calendar: leave origin at
    start_time, arrive at destination after duration (or known drive time).

    Args:
        origin: Start place.
        destination: End place.
        start_time: Departure time (default today 08:00 if empty).
        duration: Expected trip length — '4 hours', '90 min', etc.
        mode: driving / walking / cycling (for title/description only).
        notes: Extra details to store on the event.
    """
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    if not origin or not destination:
        return "Error: origin and destination are required."

    start = _parse_event_start(start_time)
    minutes = _parse_duration_minutes(duration, 240)
    end = start + timedelta(minutes=minutes)
    mode_label = (mode or "driving").strip() or "driving"
    title = f"{mode_label.title()}: {origin} → {destination}"
    description_parts = [
        f"Travel ({mode_label}) from {origin} to {destination}.",
        f"Estimated duration: {minutes} minutes.",
        "Times are local to the machine that created this file.",
    ]
    if (notes or "").strip():
        description_parts.append(notes.strip())
    description = "\n".join(description_parts)

    return _create_event_bundle(
        title=title,
        start=start,
        end=end,
        location=f"{origin} → {destination}",
        description=description,
    )


@tool
def list_saved_calendar_files() -> str:
    """List recently created .ics files in the calendar_events folder."""
    if not EVENTS_DIR.is_dir():
        return "No calendar_events folder yet — create an event first."
    files = sorted(EVENTS_DIR.glob("*.ics"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "calendar_events/ exists but has no .ics files yet."
    lines = [f"Saved calendar files ({len(files)}):"]
    for path in files[:20]:
        lines.append(f"- {path}")
    return "\n".join(lines)


def build_agent():
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
    )
    return create_agent(
        llm,
        tools=[create_calendar_event, create_travel_calendar, list_saved_calendar_files],
        system_prompt=(
            "You are a travel calendar assistant. When the user wants to add something "
            "to their calendar (trip, departure, meeting related to travel, reminder), "
            "always call a tool — never invent file paths.\n"
            "- Single event with title/time → create_calendar_event.\n"
            "- Trip between two places → create_travel_calendar (include origin, destination, "
            "start_time, and duration or travel time if known).\n"
            "- Asking what was saved → list_saved_calendar_files.\n"
            "After tools return, explain clearly: (1) path to the .ics file, "
            "(2) how to import it in Google Calendar, (3) the quick-add link, "
            "(4) whether Google API push happened. "
            "Use prior turns for places/times (e.g. 'add that trip to my calendar')."
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
        "Travel calendar agent",
        "ask to create calendar events or travel itinerary .ics files for Google Calendar.",
        graph,
    )


if __name__ == "__main__":
    main()
