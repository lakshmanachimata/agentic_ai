from __future__ import annotations

from datetime import datetime, timedelta

import calendar_agent


def test_escape_and_fold_ics():
    assert calendar_agent._escape_ics_text("a;b,c\n") == "a\\;b\\,c\\n"
    short = calendar_agent._fold_ics_line("SUMMARY:Hi")
    assert short == "SUMMARY:Hi"
    long_line = "X" * 90
    folded = calendar_agent._fold_ics_line(long_line)
    assert "\r\n " in folded
    assert folded.split("\r\n")[0] == "X" * 75


def test_parse_duration_minutes_variants():
    assert calendar_agent._parse_duration_minutes("") == 60
    assert calendar_agent._parse_duration_minutes("90") == 90
    assert calendar_agent._parse_duration_minutes("8:47") == 8 * 60 + 47
    assert calendar_agent._parse_duration_minutes("8 hours 47 minutes") == 8 * 60 + 47
    assert calendar_agent._parse_duration_minutes("2 hours") == 120
    assert calendar_agent._parse_duration_minutes("8h47m") == 8 * 60 + 47
    assert calendar_agent._parse_duration_minutes("8h 47m") == 8 * 60 + 47
    assert calendar_agent._parse_duration_minutes("not-a-duration") == 60


def test_parse_event_start_relative_and_absolute():
    now = datetime.now()
    tomorrow = calendar_agent._parse_event_start("tomorrow 10:00 AM")
    assert tomorrow.hour == 10
    assert tomorrow.date() == (now + timedelta(days=1)).date()

    later = calendar_agent._parse_event_start("day after tomorrow 9:00")
    assert later.date() == (now + timedelta(days=2)).date()

    abs_dt = calendar_agent._parse_event_start("2026-08-21 14:30")
    assert abs_dt == datetime(2026, 8, 21, 14, 30)


def test_safe_filename_and_gcal_url():
    assert calendar_agent._safe_filename("Drive: Paris → Lyon!") == "Drive_Paris_Lyon"
    assert calendar_agent._safe_filename("@@@") == "event"
    start = datetime(2026, 8, 21, 10, 0)
    end = datetime(2026, 8, 21, 18, 47)
    url = calendar_agent._google_template_url(
        "Trip", start, end, location="Mumbai", description="notes"
    )
    assert url.startswith("https://calendar.google.com/calendar/render?")
    assert "action=TEMPLATE" in url
    assert "text=Trip" in url


def test_build_ics_contains_event():
    start = datetime(2026, 8, 21, 10, 0)
    end = datetime(2026, 8, 21, 11, 0)
    body = calendar_agent._build_ics(
        title="Standup",
        start=start,
        end=end,
        location="Office",
        description="weekly",
        uid="abc@agentic_ai",
    )
    assert "BEGIN:VEVENT" in body
    assert "SUMMARY:Standup" in body
    assert "LOCATION:Office" in body
    assert "UID:abc@agentic_ai" in body


def test_create_calendar_event_validation():
    assert "title is required" in calendar_agent.create_calendar_event.invoke(
        {"title": "  ", "start_time": "tomorrow 10 AM"}
    )
    assert "start_time is required" in calendar_agent.create_calendar_event.invoke(
        {"title": "Meet", "start_time": ""}
    )


def test_create_calendar_event_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(calendar_agent, "LATEST_INVITE_PATH", tmp_path / "latest_invite.json")
    monkeypatch.setattr(calendar_agent, "_open_calendar_url", lambda url: "Browser: skipped")
    monkeypatch.setattr(calendar_agent, "_try_google_api_insert", lambda **_k: None)

    out = calendar_agent.create_calendar_event.invoke(
        {
            "title": "Standup",
            "start_time": "2026-08-21 09:00",
            "duration": "30 min",
            "location": "Zoom",
        }
    )
    assert "Standup" in out
    assert "2026-08-21 09:00" in out
    ics = list(tmp_path.glob("*.ics"))
    assert len(ics) == 1
    assert (tmp_path / "latest_invite.json").is_file()


def test_create_travel_calendar_errors():
    assert "origin and destination" in calendar_agent.create_travel_calendar.invoke(
        {"origin": "", "destination": "Hyderabad", "start_time": "tomorrow 10 AM"}
    )
    assert "start_time is required" in calendar_agent.create_travel_calendar.invoke(
        {"origin": "Mumbai", "destination": "Hyderabad", "start_time": ""}
    )


def test_open_calendar_url_gui_skip(monkeypatch):
    monkeypatch.setenv("AGENTIC_AI_GUI", "1")
    note = calendar_agent._open_calendar_url("https://calendar.google.com/example")
    assert "skipped auto-open" in note
    assert "no URL" in calendar_agent._open_calendar_url("  ")


def test_list_saved_calendar_files(tmp_path, monkeypatch):
    missing = tmp_path / "absent"
    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", missing)
    assert "No calendar_events folder" in calendar_agent.list_saved_calendar_files.invoke({})
    missing.mkdir()
    msg = calendar_agent.list_saved_calendar_files.invoke({})
    assert "no .ics" in msg
    (missing / "a.ics").write_text("BEGIN:VCALENDAR\n", encoding="utf-8")
    listed = calendar_agent.list_saved_calendar_files.invoke({})
    assert "a.ics" in listed
