"""
LangChain agent (ReAct-style graph): travel / calendar events.

Always creates:
  1) An ``.ics`` (iCalendar) file under ``calendar_events/`` — import into
     Google Calendar via Settings → Import, or open the file on many devices.
  2) A Google Calendar "Add event" TEMPLATE link (no API key / OAuth), and
     opens that link in the default browser so you can Save the event manually.

Optionally, if Google Calendar OAuth credentials are set up and the optional
packages are installed, also inserts the event into the user's calendar via API:
  pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
  Place OAuth desktop client JSON as ``credentials.json`` in this project folder.
  First run opens a browser to authorize; token is saved as ``token.json``.

Requires Ollama running locally with the model pulled:
  ollama pull qwen3.5:latest

Interactive:
  python calendar_agent.py

One-off:
  python calendar_agent.py "Create a calendar event: Drive Paris to Lyon tomorrow 9 AM, 4 hours"
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent_common import invoke_agent, make_chat_ollama, run_interactive
from route_common import fetch_osrm_route, format_duration, parse_start_time
from tracing_common import traceable

import httpx

EVENTS_DIR = Path(__file__).resolve().parent / "calendar_events"
LATEST_INVITE_PATH = EVENTS_DIR / "latest_invite.json"
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
    """Parse start time; supports today/tomorrow and absolute datetimes."""
    raw = (start_time or "").strip()
    if not raw:
        return parse_start_time("")

    lowered = raw.lower()
    day_offset = 0
    if re.search(r"\bday after tomorrow\b", lowered):
        day_offset = 2
        raw = re.sub(r"\bday after tomorrow\b", "", raw, flags=re.I)
    elif re.search(r"\btomorrow\b", lowered):
        day_offset = 1
        raw = re.sub(r"\btomorrow\b", "", raw, flags=re.I)
    elif re.search(r"\btoday\b", lowered):
        day_offset = 0
        raw = re.sub(r"\btoday\b", "", raw, flags=re.I)

    raw = re.sub(r"\bat\b", " ", raw, flags=re.I)
    raw = re.sub(r"\s+", " ", raw).strip(" ,:-")

    # Absolute date already present — do not add relative day_offset on top.
    if re.search(r"\d{4}-\d{2}-\d{2}", raw):
        return parse_start_time(raw)

    base = parse_start_time(raw)
    if day_offset:
        base = base + timedelta(days=day_offset)
    return base


def _parse_duration_minutes(duration: str, default: int = 60) -> int:
    """Parse lengths like '8 hours 47 minutes', '8h47m', '527', '8:47' (h:mm)."""
    raw = (duration or "").strip().lower()
    if not raw:
        return default
    if raw.isdigit():
        return max(1, int(raw))

    # Trailing "h:mm" or "hh:mm" as a duration (not a clock time of day)
    m_colon = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\s*(?:h|hr|hrs|hours))?$", raw)
    if m_colon:
        return max(1, int(m_colon.group(1)) * 60 + int(m_colon.group(2)))

    # Compact forms first so "8h47m" is not parsed as minutes-only ("47m").
    m_compact = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*h(?:ours?)?\s*(\d+)?\s*m(?:in(?:utes?)?)?",
        raw,
    )
    if m_compact:
        hours = float(m_compact.group(1))
        mins = int(m_compact.group(2) or 0)
        return max(1, int(round(hours * 60)) + mins)

    total = 0
    for amount, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes)\b",
        raw,
    ):
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


@traceable(name="calendar.osrm_duration", run_type="retriever")
def _osrm_duration_seconds(origin: str, destination: str, mode: str) -> tuple[float | None, str]:
    """Return (seconds, note) from OSRM, or (None, error)."""
    try:
        with httpx.Client() as client:
            route = fetch_osrm_route(client, origin, destination, mode or "driving")
    except Exception as e:
        return None, f"Routing lookup failed: {e}"

    if isinstance(route, str):
        return None, route
    return float(route.duration_s), (
        f"OSRM {mode or 'driving'}: {format_duration(route.duration_s)} "
        f"({route.distance_m / 1000:.1f} km)"
    )


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


@traceable(name="calendar.open_gcal_url", run_type="tool")
def _open_calendar_url(url: str) -> str:
    """Open the Google Calendar add-event URL in the default browser.

    On macOS this is equivalent to ``open <url>``; elsewhere uses ``webbrowser``
    (or ``xdg-open`` / Windows start via the stdlib).
    Skipped when ``AGENTIC_AI_GUI=1`` (Streamlit UI opens the link on button click).
    """
    url = (url or "").strip()
    if not url:
        return "Browser: no URL to open."
    if os.environ.get("AGENTIC_AI_GUI", "").strip() in ("1", "true", "yes", "on"):
        return (
            "Browser: skipped auto-open (GUI mode). "
            "Use the Add to Calendar button in the UI."
        )
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", url], check=False)
            return "Browser: opened Google Calendar add-event page (macOS open)."
        opened = webbrowser.open(url, new=2)
        if opened:
            return "Browser: opened Google Calendar add-event page."
        return (
            "Browser: could not auto-open; paste this URL manually:\n" + url
        )
    except Exception as e:
        return f"Browser: failed to open ({e}). Paste this URL manually:\n{url}"


@traceable(name="calendar.create_event_bundle", run_type="tool")
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
    browser_note = _open_calendar_url(gcal_url)
    api_note = _try_google_api_insert(
        title=title,
        start=start,
        end=end,
        location=location,
        description=description,
    )

    invite = {
        "title": title,
        "when": f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} (local)",
        "start": start.strftime("%Y-%m-%d %H:%M"),
        "end": end.strftime("%Y-%m-%d %H:%M"),
        "location": location or "",
        "notes": (description or "")[:500],
        "ics_path": str(path),
        "gcal_url": gcal_url,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_INVITE_PATH.write_text(json.dumps(invite, indent=2), encoding="utf-8")
    except OSError:
        pass

    lines = [
        f"Title: {title}",
        f"When: {invite['when']}",
        f"Location: {location or '(none)'}",
        f"ICS file (import into Google Calendar): {path}",
        "  Google Calendar → Settings → Import & export → Import → choose the .ics file",
        f"Quick-add link: {gcal_url}",
        browser_note,
        "Click Save in the browser tab to add the event to your Google Calendar.",
    ]
    if description:
        lines.insert(3, f"Notes: {description[:500]}")
    if api_note:
        lines.append(api_note)
    else:
        lines.append(
            "API push: not used (no credentials.json/token.json). "
            "Use the opened browser form (or the .ics file) to save manually."
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
    Opens the Google add-event page in the browser for the user to Save.

    Args:
        title: Event title (e.g. 'Drive Mumbai to Hyderabad').
        start_time: REQUIRED clock + day. Prefer 'tomorrow 10:00 AM' or
            '2026-07-30 10:00'. Never omit the day word if the user said tomorrow.
        duration: Length if end_time empty — e.g. '8 hours 47 minutes', '527 min'.
        location: Optional place / address.
        description: Optional notes.
        end_time: Optional explicit end (same date rules as start_time).
    """
    title = (title or "").strip()
    if not title:
        return "Error: title is required."
    if not (start_time or "").strip():
        return (
            "Error: start_time is required. Pass e.g. 'tomorrow 10:00 AM' or "
            "'2026-07-30 10:00' (local)."
        )

    start = _parse_event_start(start_time)
    if (end_time or "").strip():
        end = _parse_event_start(end_time)
        if end <= start:
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
    start_time: str,
    duration: str = "",
    mode: str = "driving",
    notes: str = "",
) -> str:
    """Create a road-trip calendar event (departure → arrival) as .ics + Google link.

    Duration is taken from live OSRM routing (authoritative for drive/walk/bike).
    Only if routing fails is the optional duration argument used.

    Args:
        origin: Start place (e.g. Mumbai).
        destination: End place (e.g. Hyderabad).
        start_time: REQUIRED. Must include day: 'tomorrow 10:00 AM' or
            '2026-07-30 10:00'. Do not pass only '10 AM' if the user said tomorrow.
        duration: Optional fallback only — e.g. '8 hours 47 minutes'. Prefer leaving
            empty so OSRM sets the real length.
        mode: driving (default), walking, or cycling.
        notes: Extra details for the event description.
    """
    origin = (origin or "").strip()
    destination = (destination or "").strip()
    if not origin or not destination:
        return "Error: origin and destination are required."
    if not (start_time or "").strip():
        return (
            "Error: start_time is required. Include the day, e.g. "
            "'tomorrow 10:00 AM' or '2026-07-30 10:00'."
        )

    start = _parse_event_start(start_time)
    mode_label = (mode or "driving").strip() or "driving"

    seconds, route_note = _osrm_duration_seconds(origin, destination, mode_label)
    if seconds is not None and seconds > 0:
        minutes = max(1, int(round(seconds / 60.0)))
        duration_source = route_note
    else:
        minutes = _parse_duration_minutes(duration, 0)
        if minutes <= 0:
            return (
                f"Error: could not resolve trip duration via routing ({route_note}). "
                "Pass duration explicitly, e.g. duration='8 hours 47 minutes'."
            )
        duration_source = f"fallback duration '{duration}' (OSRM unavailable: {route_note})"

    end = start + timedelta(minutes=minutes)
    title = f"{mode_label.title()}: {origin} → {destination}"
    description_parts = [
        f"Travel ({mode_label}) from {origin} to {destination}.",
        f"Departure: {start.strftime('%Y-%m-%d %H:%M')} (local).",
        f"Arrival: {end.strftime('%Y-%m-%d %H:%M')} (local).",
        f"Duration used: {minutes} minutes — {duration_source}.",
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


def build_agent(
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
):
    llm = make_chat_ollama(model=model, temperature=temperature, top_k=top_k)
    return create_agent(
        llm,
        tools=[create_calendar_event, create_travel_calendar, list_saved_calendar_files],
        system_prompt=(
            "You are a travel calendar assistant. When the user wants to add something "
            "to their calendar (trip, departure, meeting related to travel, reminder), "
            "always call a tool — never invent file paths or trip durations.\n"
            "- Road trip A→B → create_travel_calendar. REQUIRED start_time must include "
            "the day ('tomorrow 10:00 AM' or 'YYYY-MM-DD HH:MM'). Leave duration empty "
            "so OSRM sets the real drive time; do not invent 4h/6h.\n"
            "- Single non-route event → create_calendar_event with accurate duration.\n"
            "- Asking what was saved → list_saved_calendar_files.\n"
            "After tools return, explain clearly: (1) path to the .ics file, "
            "(2) that the Google Calendar add page was opened in the browser — "
            "user should click Save (GUI: use the Add to Calendar button instead), "
            "(3) the exact local start/end datetimes from the tool, "
            "(4) whether Google API push happened. "
            "Use prior turns for places/times (e.g. 'add that trip to my calendar')."
        ),
        checkpointer=MemorySaver(),
    )


def run_query(graph: Any, question: str, *, thread_id: str | None = None) -> str:
    return invoke_agent(graph, question, thread_id=thread_id, agent_name="calendar")


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
        agent_name="calendar",
    )


if __name__ == "__main__":
    main()
