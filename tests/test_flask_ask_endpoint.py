"""Flask-level integration tests for the /ask endpoint's
SEMANTIC_INTERPRETER_ENABLED-gated routing, using Flask's real test client
(real routes, real auth gate) rather than calling modules.ai_engine directly.

Importing app.py has real side effects (starts a background data-sync
thread on import unless disabled), so this is the first test file in the
suite to import it, and it sets the required guard env vars first --
DISABLE_BACKGROUND_SCHEDULER plus test-only APP_USER/APP_PASS for the
HTTP Basic Auth gate -- before that import happens.
"""

import base64
import json
import os

os.environ.setdefault("DISABLE_BACKGROUND_SCHEDULER", "1")
os.environ.setdefault("APP_USER", "test-user")
os.environ.setdefault("APP_PASS", "test-pass")

from unittest.mock import patch  # noqa: E402

import config  # noqa: E402
from app import app as flask_app  # noqa: E402
from modules import ai_engine  # noqa: E402
from modules.intent_schema import PORTFOLIO_FILTER  # noqa: E402

_AUTH_HEADERS = {
    "Authorization": "Basic " + base64.b64encode(b"test-user:test-pass").decode(),
}


def _fake_llm_client(raw: dict):
    payload = json.dumps(raw, ensure_ascii=False)

    class Function:
        pass
    fn = Function()
    fn.arguments = payload

    class Call:
        pass
    call = Call()
    call.function = fn

    class Message:
        pass
    message = Message()
    message.tool_calls = [call]

    class Choice:
        pass
    choice = Choice()
    choice.message = message

    class Completions:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"choices": [choice]})()

    return type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()


def _status_filter_raw() -> dict:
    return {
        "intent": PORTFOLIO_FILTER, "scope": "portfolio",
        "entities": {
            "projects": [], "departments": [], "status": ["الجارية"], "programs": [],
            "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
        },
        "operations": {"filter": True, "group_by": None, "sort": None, "limit": None, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.95, "requires_clarification": False,
    }


def _client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


# ── Auth gate sanity ──────────────────────────────────────────────────────

def test_ask_without_auth_is_rejected():
    resp = _client().post("/ask", json={"query": "المشاريع الجارية"})
    assert resp.status_code == 401


def test_ask_requires_a_query():
    resp = _client().post("/ask", json={"query": ""}, headers=_AUTH_HEADERS)
    assert resp.status_code == 400


# ── Flag off (default): behavior is exactly today's ──────────────────────

def test_ask_with_flag_off_uses_legacy_path(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    assert config.SEMANTIC_INTERPRETER_ENABLED is False
    resp = _client().post("/ask", json={"query": "المشاريع الجارية"}, headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "answer" in body and "query_type" in body


# ── Flag disabled by default -- explicit, dedicated confirmation ────────

def test_semantic_interpreter_flag_is_disabled_by_default():
    """No env var set anywhere in this process -> config must read the
    documented default (off). This is the guarantee the deploy checklist
    depends on: an unconfigured/fresh environment never runs the new path."""
    assert os.environ.get("SEMANTIC_INTERPRETER_ENABLED") is None
    assert config.SEMANTIC_INTERPRETER_ENABLED is False


# ── Real persistent-session behavior through the Flask test client ──────
# Flask's test client keeps its own cookie jar for its lifetime, so reusing
# one client instance across requests is the real session-cookie mechanism
# the browser/production caller uses -- not a mock of it.

def _timeout(**kwargs):
    raise TimeoutError("forced -- keeps classification on the deterministic fallback path")


_TIMEOUT_LLM_CLIENT = type("Client", (), {
    "chat": type("Chat", (), {"completions": type("Completions", (), {"create": staticmethod(_timeout)})()})()
})()


def test_persistent_session_context_survives_across_requests_flag_off(seeded_db, today, monkeypatch):
    """A bare project name with no other classifiable signal falls through
    modules.understanding's own deterministic fast paths to its LLM
    fallback -- mock that too (not just modules.semantic_interpreter's),
    otherwise this test's classification depends on live network behavior
    and is exactly as flaky as the pre-existing "Manager C" test was
    (fixed earlier this session by adding a missing field alias for a
    different query shape; here the query has no field-alias escape hatch,
    so the fallback path is mocked deterministic instead)."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    client = _client()
    with patch("modules.understanding._get_openai", return_value=_TIMEOUT_LLM_CLIENT):
        r1 = client.post("/ask", json={"query": "الباحث الاجتماعي الثاني"}, headers=_AUTH_HEADERS)
        assert r1.status_code == 200
        r2 = client.post("/ask", json={"query": "من مديره؟"}, headers=_AUTH_HEADERS)
    assert r2.status_code == 200
    # "من مديره؟" only resolves without re-naming the project if turn 1's
    # session state (active project) actually persisted across requests.
    assert "Manager A" in r2.get_json()["answer"]


def test_persistent_session_context_survives_across_requests_flag_on(seeded_db, today, monkeypatch):
    """Same session-continuity property with the flag on -- project_summary
    isn't a v2-handled intent, so this also exercises defer-to-legacy
    across a live session, not just a single request. Both the new
    interpreter's and the legacy understanding layer's LLM calls are
    mocked to fail, for the same determinism reason as the flag-off
    version above."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    client = _client()
    with patch("modules.semantic_interpreter._get_openai", return_value=_TIMEOUT_LLM_CLIENT), \
         patch("modules.understanding._get_openai", return_value=_TIMEOUT_LLM_CLIENT):
        r1 = client.post("/ask", json={"query": "الباحث الاجتماعي الثاني"}, headers=_AUTH_HEADERS)
        assert r1.status_code == 200
        r2 = client.post("/ask", json={"query": "من مديره؟"}, headers=_AUTH_HEADERS)
    assert r2.status_code == 200
    assert "Manager A" in r2.get_json()["answer"]


# ── Flag on: routes through the new path, same response shape ───────────

def test_ask_with_flag_on_routes_through_new_path(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    client_obj = _fake_llm_client(_status_filter_raw())
    with patch("modules.semantic_interpreter._get_openai", return_value=client_obj):
        resp = _client().post("/ask", json={"query": "المشاريع الجارية"}, headers=_AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["query_type"] == PORTFOLIO_FILTER
    assert "answer" in body


def test_ask_flag_on_falls_back_gracefully_on_llm_timeout(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)

    def _raise(**kwargs):
        raise TimeoutError("simulated Azure OpenAI timeout")

    fake_client = type("Client", (), {
        "chat": type("Chat", (), {"completions": type("Completions", (), {"create": staticmethod(_raise)})()})()
    })()
    with patch("modules.semantic_interpreter._get_openai", return_value=fake_client):
        resp = _client().post("/ask", json={"query": "المشاريع الجارية"}, headers=_AUTH_HEADERS)
    # The endpoint must still return a normal 200 with a real answer, not an
    # error page or a 500 -- the fallback to the legacy path must be silent
    # to the HTTP caller.
    assert resp.status_code == 200
    body = resp.get_json()
    assert "answer" in body and body["answer"]
