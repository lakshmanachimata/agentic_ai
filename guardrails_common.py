"""Open-source Guardrails AI checks for user turns and specialist hops.

Uses Apache-2.0 ``guardrails-ai`` with **local custom validators** (no Hub
ML models, no remote inference) so tests and CLI stay offline.

- User / orchestrator: abusive language and prompt-injection.
- Specialist hops: the same, plus domain mismatch (wrong prompt between agents).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from guardrails import Guard, OnFailAction, settings
from guardrails.classes.rc import RC
from guardrails.validators import (
    FailResult,
    PassResult,
    Validator,
    register_validator,
)

SPECIALISTS = frozenset({"weather", "travel", "restaurants", "calendar"})

REFUSAL_ABUSE = (
    "I can't help with abusive or harassing language. "
    "Please rephrase a weather, travel, dining, or calendar question."
)
REFUSAL_INJECTION = (
    "That looks like an attempt to override instructions. "
    "Please ask a normal weather, travel, dining, or calendar question."
)
REFUSAL_HOP = (
    "That request does not belong to the {specialist} specialist. "
    "Ask a {specialist}-related question, or use the orchestrator so it can route correctly."
)

_TRIP_SPLIT = re.compile(r"\n*Typed trip state \(authoritative", re.IGNORECASE)

_ABUSE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bf+u+c+k+(?:ing|er|ed)?\b",
        r"\bshit\b",
        r"\bassholes?\b",
        r"\bbitch(?:es)?\b",
        r"\bbastards?\b",
        r"\bkill yourself\b",
        r"\bkys\b",
        r"\bgo die\b",
        r"\bslur\b",
        r"\b(hate|kill) you\b",
    )
)

_INJECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (?:all )?(?:previous|prior|above) (?:instructions|prompts|rules)",
        r"disregard (?:your|the) (?:system|previous) (?:prompt|instructions)",
        r"you are now (?:dan|jailbroken|unrestricted)",
        r"\bjailbreak\b",
        r"developer mode (?:enabled|on)",
        r"override (?:the )?(?:system|safety)",
        r"reveal (?:your )?(?:system prompt|hidden instructions)",
        r"new instructions\s*:",
        r"pretend you (?:are|have) no (?:rules|limits|restrictions)",
    )
)

_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "weather": (
        "weather",
        "forecast",
        "temperature",
        "humidity",
        "rainfall",
        "rain",
        "snow",
        "windy",
        "climate",
        "degrees",
        "wttr",
    ),
    "travel": (
        "travel time",
        "towns between",
        "driving",
        "drive",
        "route",
        "journey",
        "cycling",
        "walking",
        "distance",
        "osrm",
        "eta",
    ),
    "restaurants": (
        "restaurants",
        "restaurant",
        "dining",
        "cuisine",
        "bistro",
        "lunch",
        "dinner",
        "sushi",
        "pizza",
        "cafe",
        "café",
        "food",
        "eat",
    ),
    "calendar": (
        "create_travel_calendar",
        "google calendar",
        "add that trip",
        "add this trip",
        "add to calendar",
        "calendar",
        ".ics",
        "ics file",
        "invite",
        "schedule the trip",
    ),
}


def _configure_guardrails() -> None:
    """Keep Guardrails local: no Hub metrics, no OTEL export."""
    settings.disable_tracing = True
    settings.rc = RC(enable_metrics=False, use_remote_inferencing=False)


_configure_guardrails()


def guardrails_enabled() -> bool:
    flag = os.environ.get("AGENTIC_AI_GUARDRAILS", "true").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _user_portion(text: str) -> str:
    return _TRIP_SPLIT.split(text or "", 1)[0].strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def matched_domains(text: str) -> set[str]:
    """Which specialist domains the (user-portion) text looks like."""
    body = _user_portion(text).lower()
    if not body:
        return set()
    found: set[str] = set()
    for name, phrases in _DOMAIN_HINTS.items():
        if any(_contains_phrase(body, p) for p in phrases):
            found.add(name)
    return found


@register_validator(name="agentic-ai/abusive-language", data_type="string")
class AbusiveLanguage(Validator):
    def __init__(self, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, **kwargs)

    def _validate(self, value: Any, metadata: dict[str, Any]) -> FailResult | PassResult:
        text = str(value or "")
        for pat in _ABUSE_PATTERNS:
            if pat.search(text):
                return FailResult(error_message="abuse: abusive or harassing language")
        return PassResult()


@register_validator(name="agentic-ai/prompt-injection", data_type="string")
class PromptInjection(Validator):
    def __init__(self, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, **kwargs)

    def _validate(self, value: Any, metadata: dict[str, Any]) -> FailResult | PassResult:
        text = str(value or "")
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                return FailResult(error_message="injection: instruction-override / jailbreak")
        return PassResult()


@register_validator(name="agentic-ai/specialist-hop", data_type="string")
class SpecialistHop(Validator):
    """Reject a hop whose wording clearly belongs to a different specialist."""

    def __init__(self, on_fail: Any = None, **kwargs: Any) -> None:
        super().__init__(on_fail=on_fail, **kwargs)

    def _validate(self, value: Any, metadata: dict[str, Any]) -> FailResult | PassResult:
        specialist = str((metadata or {}).get("specialist") or "").strip().lower()
        if specialist not in SPECIALISTS:
            return PassResult()
        domains = matched_domains(str(value or ""))
        if not domains or specialist in domains:
            return PassResult()
        return FailResult(
            error_message=(
                f"wrong_specialist: {specialist} received a "
                f"{sorted(domains)[0]} request"
            )
        )


@lru_cache(maxsize=1)
def _safety_guard() -> Guard:
    return Guard(name="user-safety").use(
        AbusiveLanguage(on_fail=OnFailAction.NOOP),
        PromptInjection(on_fail=OnFailAction.NOOP),
    )


@lru_cache(maxsize=1)
def _hop_guard() -> Guard:
    return Guard(name="specialist-hop").use(
        AbusiveLanguage(on_fail=OnFailAction.NOOP),
        PromptInjection(on_fail=OnFailAction.NOOP),
        SpecialistHop(on_fail=OnFailAction.NOOP),
    )


def _failure_code(outcome: Any) -> str:
    for summary in getattr(outcome, "validation_summaries", None) or []:
        if getattr(summary, "validator_status", "") != "fail":
            continue
        reason = str(getattr(summary, "failure_reason", "") or "")
        return reason.split(":", 1)[0].strip() or "blocked"
    return "blocked"


def _refusal_for_code(code: str, *, specialist: str = "") -> str:
    if code == "abuse":
        return REFUSAL_ABUSE
    if code == "injection":
        return REFUSAL_INJECTION
    if code == "wrong_specialist":
        return REFUSAL_HOP.format(specialist=specialist or "that")
    return REFUSAL_ABUSE


def _run_guard(guard: Guard, text: str, metadata: dict[str, Any] | None = None) -> str | None:
    try:
        outcome = guard.validate(text, metadata=metadata or {})
    except Exception:
        return None
    if getattr(outcome, "validation_passed", True):
        return None
    return _failure_code(outcome)


def screen_user_prompt(text: str) -> str | None:
    """Return a refusal for abusive / jailbreak user text, else None."""
    if not guardrails_enabled():
        return None
    q = (text or "").strip()
    if not q:
        return None
    code = _run_guard(_safety_guard(), q)
    if not code:
        return None
    return _refusal_for_code(code)


def screen_specialist_hop(text: str, specialist: str) -> str | None:
    """Refuse abuse, injection, or a hop sent to the wrong specialist."""
    if not guardrails_enabled():
        return None
    q = (text or "").strip()
    if not q:
        return None
    spec = (specialist or "").strip().lower()
    code = _run_guard(_hop_guard(), q, metadata={"specialist": spec})
    if not code:
        return None
    return _refusal_for_code(code, specialist=spec)


def screen_prompt(text: str, *, agent_name: str = "agent") -> str | None:
    """Chokepoint used by ``invoke_agent``.

    Orchestrator / generic agents: safety only. Named specialists: safety + hop.
    """
    name = (agent_name or "agent").strip().lower()
    if name in SPECIALISTS:
        return screen_specialist_hop(text, name)
    return screen_user_prompt(text)
