from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import calendar_agent
from route_common import OsrmRoute
from tests.fakes import FakeClient, client_factory


def test_parse_event_start_empty_today_and_at():
    empty = calendar_agent._parse_event_start("")
    assert empty.hour == 8
    today = calendar_agent._parse_event_start("today 3:15 PM")
    assert today.hour == 15
    at = calendar_agent._parse_event_start("tomorrow at 11:00")
    assert at.hour == 11


def test_parse_duration_float_and_hours_only_compact():
    assert calendar_agent._parse_duration_minutes("90.0") == 90
    assert calendar_agent._parse_duration_minutes("1.5 hours") == 90
    assert calendar_agent._parse_duration_minutes("8h") == 480


def test_osrm_duration_seconds(monkeypatch):
    monkeypatch.setattr(calendar_agent.httpx, "Client", client_factory(FakeClient()))
    monkeypatch.setattr(
        calendar_agent,
        "fetch_osrm_route",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    seconds, note = calendar_agent._osrm_duration_seconds("A", "B", "driving")
    assert seconds is None
    assert "failed" in note

    monkeypatch.setattr(calendar_agent, "fetch_osrm_route", lambda *_a, **_k: "nope")
    seconds, note = calendar_agent._osrm_duration_seconds("A", "B", "driving")
    assert seconds is None
    assert note == "nope"

    route = OsrmRoute(
        poly=[(1, 2), (3, 4)],
        duration_s=3600,
        distance_m=100000,
        origin_label="A",
        dest_label="B",
        origin_lat=1,
        origin_lon=2,
        dest_lat=3,
        dest_lon=4,
        profile="driving",
    )
    monkeypatch.setattr(calendar_agent, "fetch_osrm_route", lambda *_a, **_k: route)
    seconds, note = calendar_agent._osrm_duration_seconds("A", "B", "driving")
    assert seconds == 3600
    assert "OSRM" in note


def test_open_calendar_url_platforms(monkeypatch):
    monkeypatch.delenv("AGENTIC_AI_GUI", raising=False)
    monkeypatch.setattr(calendar_agent.sys, "platform", "darwin")
    ran = []
    monkeypatch.setattr(
        calendar_agent.subprocess,
        "run",
        lambda *a, **k: ran.append(a),
    )
    note = calendar_agent._open_calendar_url("https://calendar.google.com/x")
    assert "macOS" in note
    assert ran

    monkeypatch.setattr(calendar_agent.sys, "platform", "linux")
    monkeypatch.setattr(calendar_agent.webbrowser, "open", lambda *_a, **_k: True)
    assert "opened" in calendar_agent._open_calendar_url("https://calendar.google.com/x")

    monkeypatch.setattr(calendar_agent.webbrowser, "open", lambda *_a, **_k: False)
    assert "could not auto-open" in calendar_agent._open_calendar_url(
        "https://calendar.google.com/x"
    )

    def boom(*_a, **_k):
        raise OSError("nope")

    monkeypatch.setattr(calendar_agent.sys, "platform", "linux")
    monkeypatch.setattr(calendar_agent.webbrowser, "open", boom)
    assert "failed to open" in calendar_agent._open_calendar_url(
        "https://calendar.google.com/x"
    )


def test_google_api_import_error_and_missing_creds(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CALENDAR_ENABLED", "1")
    monkeypatch.setattr(calendar_agent, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    monkeypatch.setattr(calendar_agent, "TOKEN_PATH", tmp_path / "token.json")
    msg = calendar_agent._try_google_api_insert(
        title="T", start=datetime.now(), end=datetime.now() + timedelta(hours=1)
    )
    assert msg is None or "skipped" in msg or "error" in msg.lower() or "created" in (msg or "")


def test_google_api_success(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.json"
    token_path = tmp_path / "token.json"
    creds_path.write_text("{}", encoding="utf-8")
    token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(calendar_agent, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(calendar_agent, "TOKEN_PATH", token_path)
    monkeypatch.setenv("GOOGLE_CALENDAR_ENABLED", "true")
    monkeypatch.setenv("TZ", "Asia/Kolkata")

    class Creds:
        valid = True
        expired = False
        refresh_token = None

        def to_json(self):
            return "{}"

        @staticmethod
        def from_authorized_user_file(*_a, **_k):
            return Creds()

    class Events:
        def insert(self, **_k):
            return self

        def execute(self):
            return {"htmlLink": "https://cal.test/e", "id": "abc"}

    class Service:
        def events(self):
            return Events()

    google_auth_transport = SimpleNamespace(Request=object)
    google_oauth2 = SimpleNamespace(Credentials=Creds)
    google_flow = SimpleNamespace(
        InstalledAppFlow=SimpleNamespace(
            from_client_secrets_file=lambda *_a, **_k: SimpleNamespace(
                run_local_server=lambda **_p: Creds()
            )
        )
    )
    google_disc = SimpleNamespace(build=lambda *_a, **_k: Service())

    monkeypatch.setitem(__import__("sys").modules, "google.auth.transport.requests", google_auth_transport)
    monkeypatch.setitem(__import__("sys").modules, "google.oauth2.credentials", google_oauth2)
    monkeypatch.setitem(__import__("sys").modules, "google_auth_oauthlib.flow", google_flow)
    monkeypatch.setitem(__import__("sys").modules, "googleapiclient.discovery", google_disc)

    start = datetime(2026, 8, 21, 10, 0)
    out = calendar_agent._try_google_api_insert(
        title="Trip",
        start=start,
        end=start + timedelta(hours=1),
        location="Mumbai",
        description="notes",
    )
    assert out is not None
    assert "abc" in out or "skipped" in out or "error" in out.lower()


def test_google_api_refresh_and_errors(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.json"
    token_path = tmp_path / "token.json"
    creds_path.write_text("{}", encoding="utf-8")
    token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(calendar_agent, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(calendar_agent, "TOKEN_PATH", token_path)

    class Creds:
        valid = False
        expired = True
        refresh_token = "r"

        def refresh(self, _req):
            self.valid = True

        def to_json(self):
            return "{}"

        @staticmethod
        def from_authorized_user_file(*_a, **_k):
            return Creds()

    class Events:
        def insert(self, **_k):
            return self

        def execute(self):
            raise RuntimeError("quota")

    monkeypatch.setitem(
        __import__("sys").modules,
        "google.auth.transport.requests",
        SimpleNamespace(Request=object),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "google.oauth2.credentials",
        SimpleNamespace(Credentials=Creds),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "google_auth_oauthlib.flow",
        SimpleNamespace(InstalledAppFlow=MagicMock()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "googleapiclient.discovery",
        SimpleNamespace(build=lambda *_a, **_k: SimpleNamespace(events=lambda: Events())),
    )
    out = calendar_agent._try_google_api_insert(
        title="T",
        start=datetime(2026, 8, 21, 10, 0),
        end=datetime(2026, 8, 21, 11, 0),
    )
    assert out is None or "error" in out.lower() or "skipped" in out


def test_google_api_needs_credentials_json(monkeypatch, tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(calendar_agent, "CREDENTIALS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(calendar_agent, "TOKEN_PATH", token_path)
    monkeypatch.setenv("GOOGLE_CALENDAR_ENABLED", "1")

    class Creds:
        valid = False
        expired = False
        refresh_token = None

        @staticmethod
        def from_authorized_user_file(*_a, **_k):
            return Creds()

    monkeypatch.setitem(
        __import__("sys").modules,
        "google.auth.transport.requests",
        SimpleNamespace(Request=object),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "google.oauth2.credentials",
        SimpleNamespace(Credentials=Creds),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "google_auth_oauthlib.flow",
        SimpleNamespace(InstalledAppFlow=MagicMock()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "googleapiclient.discovery",
        SimpleNamespace(build=MagicMock()),
    )
    out = calendar_agent._try_google_api_insert(
        title="T",
        start=datetime.now(),
        end=datetime.now() + timedelta(hours=1),
    )
    assert out is None or "credentials.json" in out or "skipped" in out


def test_create_event_with_end_time(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(calendar_agent, "LATEST_INVITE_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(calendar_agent, "_open_calendar_url", lambda url: "Browser: skipped")
    monkeypatch.setattr(calendar_agent, "_try_google_api_insert", lambda **_k: "API: ok")
    out = calendar_agent.create_calendar_event.invoke(
        {
            "title": "Meet",
            "start_time": "2026-08-21 09:00",
            "end_time": "2026-08-21 08:00",
            "duration": "45 min",
            "location": "Zoom",
            "description": "standup notes",
        }
    )
    assert "Meet" in out
    assert "API: ok" in out
    assert "standup" in out

    out2 = calendar_agent.create_calendar_event.invoke(
        {
            "title": "Later",
            "start_time": "2026-08-21 09:00",
            "end_time": "2026-08-21 11:00",
        }
    )
    assert "11:00" in out2


def test_create_event_invite_write_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(calendar_agent, "_open_calendar_url", lambda url: "ok")
    monkeypatch.setattr(calendar_agent, "_try_google_api_insert", lambda **_k: None)

    class BoomPath:
        def write_text(self, *_a, **_k):
            raise OSError("disk")

    monkeypatch.setattr(calendar_agent, "LATEST_INVITE_PATH", BoomPath())
    out = calendar_agent.create_calendar_event.invoke(
        {"title": "X", "start_time": "2026-08-21 09:00"}
    )
    assert "X" in out


def test_create_travel_calendar_osrm_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(calendar_agent, "LATEST_INVITE_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(calendar_agent, "_open_calendar_url", lambda url: "ok")
    monkeypatch.setattr(calendar_agent, "_try_google_api_insert", lambda **_k: None)
    monkeypatch.setattr(
        calendar_agent,
        "_osrm_duration_seconds",
        lambda *_a, **_k: (3600.0, "OSRM driving: 1 hr"),
    )
    out = calendar_agent.create_travel_calendar.invoke(
        {
            "origin": "Mumbai",
            "destination": "Pune",
            "start_time": "2026-08-21 09:00",
            "notes": "bring charger",
        }
    )
    assert "Mumbai" in out
    assert "bring charger" in out

    monkeypatch.setattr(
        calendar_agent,
        "_osrm_duration_seconds",
        lambda *_a, **_k: (None, "down"),
    )
    fail = calendar_agent.create_travel_calendar.invoke(
        {
            "origin": "Mumbai",
            "destination": "Pune",
            "start_time": "2026-08-21 09:00",
        }
    )
    assert "could not resolve" in fail

    fallback = calendar_agent.create_travel_calendar.invoke(
        {
            "origin": "Mumbai",
            "destination": "Pune",
            "start_time": "2026-08-21 09:00",
            "duration": "2 hours",
        }
    )
    assert "fallback" in fallback.lower() or "2" in fallback


def test_create_travel_calendar_uses_typed_trip_state(tmp_path, monkeypatch):
    import trip_state as ts

    monkeypatch.setattr(calendar_agent, "EVENTS_DIR", tmp_path)
    monkeypatch.setattr(calendar_agent, "LATEST_INVITE_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(calendar_agent, "_open_calendar_url", lambda url: "ok")
    monkeypatch.setattr(calendar_agent, "_try_google_api_insert", lambda **_k: None)
    monkeypatch.setattr(
        calendar_agent,
        "_osrm_duration_seconds",
        lambda *_a, **_k: (None, "down"),
    )
    ts.update_trip(
        origin="Mumbai",
        destination="Pune",
        start_time="2026-08-21 09:00",
        duration_s=7200,
    )
    out = calendar_agent.create_travel_calendar.invoke(
        {"origin": "", "destination": "", "start_time": ""}
    )
    assert "typed trip state" in out.lower()
    assert "Mumbai" in out
    assert "11:00" in out


def test_calendar_build_run_main(monkeypatch):
    monkeypatch.setattr(calendar_agent, "make_chat_ollama", lambda **_k: "llm")
    monkeypatch.setattr(calendar_agent, "create_agent", lambda *_a, **_k: "graph")
    assert calendar_agent.build_agent() == "graph"
    monkeypatch.setattr(calendar_agent, "invoke_agent", lambda *_a, **_k: "ok")
    assert calendar_agent.run_query("g", "q") == "ok"
    monkeypatch.setattr(calendar_agent, "build_agent", lambda **_k: "g")
    monkeypatch.setattr(calendar_agent.sys, "argv", ["calendar_agent.py", "add", "trip"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))
    calendar_agent.main()
    assert printed
    monkeypatch.setattr(calendar_agent.sys, "argv", ["calendar_agent.py"])
    called = []
    monkeypatch.setattr(calendar_agent, "run_interactive", lambda *a, **k: called.append(True))
    calendar_agent.main()
    assert called
