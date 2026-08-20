"""HTTP/client doubles used by unit tests (no live network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx


class FakeResponse:
    def __init__(
        self,
        payload: Any = None,
        status_code: int = 200,
        *,
        json_error: bool = False,
        http_error: bool = False,
    ):
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error
        self._http_error = http_error
        self.request = httpx.Request("GET", "https://example.test")

    def raise_for_status(self) -> None:
        if self._http_error or self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> Any:
        if self._json_error:
            raise json.JSONDecodeError("bad", "doc", 0)
        return self._payload


class FakeClient:
    """httpx.Client stand-in with scripted get/post handlers."""

    def __init__(self, get=None, post=None):
        self._get = get
        self._post = post
        self.gets: list[tuple[str, dict[str, Any]]] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_a) -> bool:
        return False

    def get(self, url: str, **kwargs):
        self.gets.append((url, kwargs))
        if self._get is None:
            raise AssertionError(f"unexpected GET {url}")
        return self._get(url, **kwargs)

    def post(self, url: str, **kwargs):
        self.posts.append((url, kwargs))
        if self._post is None:
            raise AssertionError(f"unexpected POST {url}")
        return self._post(url, **kwargs)


def client_factory(inner: FakeClient):
    """Return a callable matching ``httpx.Client(**kwargs)``."""

    def _make(**_k) -> FakeClient:
        return inner

    return _make


def nominatim_hit(query: str, lat: str = "19.07", lon: str = "72.87") -> FakeResponse:
    return FakeResponse(
        [{"lat": lat, "lon": lon, "display_name": f"{query}, India"}]
    )


MUMBAI_HYD_OSRM = {
    "code": "Ok",
    "routes": [
        {
            "duration": 31620,
            "distance": 704000,
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [72.87, 19.07],
                    [73.85, 18.52],
                    [78.48, 17.38],
                ],
            },
        }
    ],
}


def osrm_ok_handler(url: str, **kwargs):
    if "nominatim" in url:
        q = kwargs.get("params", {}).get("q") or kwargs.get("params", {}).get("lat", "x")
        if "lat" in (kwargs.get("params") or {}) and "q" not in (kwargs.get("params") or {}):
            return FakeResponse(
                {
                    "address": {"city": "Pune"},
                    "display_name": "Pune, Maharashtra, India",
                }
            )
        return nominatim_hit(str(q))
    if "overpass" in url:
        return FakeResponse({"elements": []})
    return FakeResponse(MUMBAI_HYD_OSRM)


def wttr_payload() -> dict[str, Any]:
    return {
        "current_condition": [
            {
                "temp_C": "22",
                "FeelsLikeC": "21",
                "windspeedKmph": "10",
                "winddir16Point": "NW",
                "humidity": "55",
                "pressure": "1012",
                "weatherDesc": [{"value": "Sunny"}],
            }
        ],
        "nearest_area": [
            {
                "areaName": [{"value": "Paris"}],
                "region": [{"value": "Ile-de-France"}],
                "country": [{"value": "France"}],
            }
        ],
        "weather": [
            {
                "date": "2026-08-21",
                "maxtempC": "26",
                "mintempC": "16",
                "hourly": [
                    {
                        "time": "900",
                        "tempC": "20",
                        "chanceofrain": "10",
                        "weatherDesc": [{"value": "Partly cloudy"}],
                    },
                    {
                        "time": "1200",
                        "tempC": "24",
                        "chanceofrain": "5",
                        "weatherDesc": [{"value": "Sunny"}],
                    },
                ],
            }
        ],
    }


class SessionState:
    """Minimal Streamlit session_state stand-in."""

    def __init__(self, **values):
        self._data = dict(values)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, key):
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def markdown(self, *a, **k):
        return None

    def caption(self, *a, **k):
        return None

    def write(self, *a, **k):
        return None

    def code(self, *a, **k):
        return None

    def button(self, *a, **k):
        return False

    def link_button(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def success(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None


class FakeStreamlit:
    """Record Streamlit calls and drive sidebar/chat for ``main()`` tests."""

    def __init__(
        self,
        *,
        chat_prompt: str | None = None,
        selected_client: str = "orchestrator",
        clear_chat: bool = False,
        add_calendar: bool = False,
    ):
        self.session_state = SessionState()
        self.chat_prompt = chat_prompt
        self.selected_client = selected_client
        self.clear_chat = clear_chat
        self.add_calendar = add_calendar
        self.rerun_called = False
        self.titles: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.sidebar = _Ctx()

        def cache_resource(*_a, **_k):
            def deco(fn):
                fn.clear = lambda: None
                return fn

            return deco

        self.cache_resource = cache_resource

    def set_page_config(self, **_k):
        return None

    def title(self, text):
        self.titles.append(text)

    def caption(self, *a, **k):
        return None

    def header(self, *a, **k):
        return None

    def subheader(self, *a, **k):
        return None

    def divider(self):
        return None

    def markdown(self, *a, **k):
        return None

    def write(self, *a, **k):
        return None

    def code(self, *a, **k):
        return None

    def success(self, msg):
        self.successes.append(str(msg))

    def error(self, msg):
        self.errors.append(str(msg))

    def warning(self, *a, **k):
        return None

    def link_button(self, *a, **k):
        return None

    def chat_message(self, *_a, **_k):
        return _Ctx()

    def spinner(self, *_a, **_k):
        return _Ctx()

    def container(self, **_k):
        return _Ctx()

    def columns(self, *_a, **_k):
        return (_Ctx(), _Ctx())

    def selectbox(self, label, options, **kwargs):
        key = kwargs.get("key")
        if key:
            current = self.session_state.get(key, options[0])
            self.session_state[key] = current
            return current
        return self.selected_client

    def slider(self, *_a, **kwargs):
        key = kwargs.get("key")
        if key and key in self.session_state:
            return self.session_state[key]
        return kwargs.get("min_value", 0)

    def button(self, label, **_k):
        if "Clear chat" in str(label):
            return self.clear_chat
        if "Add to Calendar" in str(label):
            return self.add_calendar
        return False

    def chat_input(self, *_a, **_k):
        return self.chat_prompt

    def rerun(self):
        self.rerun_called = True
        raise RuntimeError("streamlit-rerun")
