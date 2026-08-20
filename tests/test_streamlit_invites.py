from __future__ import annotations

import json

import streamlit_app
from streamlit_app import CalendarInvite, parse_calendar_invites, _field, open_url


def test_field_and_parse_invite_from_text():
    text = (
        "Title: Drive: Mumbai → Hyderabad\n"
        "When: 2026-08-21 10:00 → 2026-08-21 18:47 (local)\n"
        "Location: Mumbai → Hyderabad\n"
        "Notes: OSRM driving\n"
        "ICS file (import into Google Calendar): /tmp/trip.ics\n"
        "Quick-add link: https://calendar.google.com/calendar/render?action=TEMPLATE&text=Trip\n"
    )
    assert _field(text, "Title").startswith("Drive")
    invites = parse_calendar_invites(text)
    assert len(invites) == 1
    inv = invites[0]
    assert inv.title.startswith("Drive")
    assert "TEMPLATE" in inv.gcal_url
    assert inv.ics_path.endswith("trip.ics")


def test_parse_invites_url_only_and_dedupe():
    url = "https://calendar.google.com/calendar/render?action=TEMPLATE&text=A"
    invites = parse_calendar_invites(f"Please open {url}")
    assert invites
    assert invites[0].gcal_url.startswith("https://calendar.google.com")

    doubled = parse_calendar_invites(f"Title: Same\nQuick-add link: {url}\nTitle: Same\nQuick-add link: {url}\n")
    assert len(doubled) == 1


def test_load_latest_invite_file(tmp_path, monkeypatch):
    path = tmp_path / "latest_invite.json"
    monkeypatch.setattr(streamlit_app, "LATEST_INVITE_PATH", path)
    assert streamlit_app.load_latest_invite_file() is None
    path.write_text(
        json.dumps(
            {
                "title": "Trip",
                "when": "tomorrow",
                "location": "Mumbai",
                "notes": "drive",
                "ics_path": "/tmp/a.ics",
                "gcal_url": "https://calendar.google.com/calendar/render?action=TEMPLATE",
            }
        ),
        encoding="utf-8",
    )
    inv = streamlit_app.load_latest_invite_file()
    assert inv is not None
    assert inv.title == "Trip"
    from_file = parse_calendar_invites("", include_latest_file=True)
    assert from_file[0].title == "Trip"


def test_open_url_empty():
    ok, msg = open_url("  ")
    assert ok is False
    assert "No Google Calendar URL" in msg


def test_calendar_invite_defaults():
    inv = CalendarInvite()
    assert inv.title == ""
    assert inv.gcal_url == ""
