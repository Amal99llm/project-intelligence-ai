"""Regression coverage for current-turn metric precedence on project follow-ups."""

import base64
from unittest.mock import patch

import pytest

import config
from app import app as flask_app
from modules import ai_engine, session_context
from modules.semantic_dictionary import detect_requested_fields


BACKLOG_WORDINGS = (
    "كم باقي إيراد", "الإيراد المتبقي", "المتبقي من المشروع",
    "كم باقي يتحقق", "الأعمال المتبقية", "قيمة الأعمال المتبقية",
    "الباك لوق", "Backlog",
)


def _timeout(**kwargs):
    raise TimeoutError("force deterministic interpretation")


_TIMEOUT_CLIENT = type("Client", (), {
    "chat": type("Chat", (), {
        "completions": type("Completions", (), {"create": staticmethod(_timeout)})(),
    })(),
})()


def _assert_backlog(answer: str) -> None:
    assert "Backlog" in answer or "الأعمال المتبقية" in answer
    assert "200,000.00" in answer
    assert "نهاية المشروع" not in answer


@pytest.mark.parametrize("wording", BACKLOG_WORDINGS)
def test_required_backlog_synonyms_are_canonical_current_turn_metrics(wording):
    assert [field.canonical for field in detect_requested_fields(wording)] == ["backlog"]


@pytest.mark.parametrize("enabled", [False, True])
def test_ai_engine_metric_changes_keep_active_project(seeded_db, today, monkeypatch, enabled):
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", enabled)
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    session_id = f"metric-precedence-{enabled}"

    with patch("modules.semantic_interpreter._get_openai", return_value=_TIMEOUT_CLIENT), \
         patch("modules.understanding._get_openai", return_value=_TIMEOUT_CLIENT), \
         patch("modules.response_composer._get_openai", return_value=_TIMEOUT_CLIENT):
        ai_engine.answer("اعطني ملخص الباحث", session_id=session_id)
        ai_engine.answer("وربحه؟", session_id=session_id)
        ai_engine.answer("وهامشه؟", session_id=session_id)
        ai_engine.answer("متى ينتهي؟", session_id=session_id)
        backlog = ai_engine.answer("كم باقي إيراد؟", session_id=session_id)["answer"]
        correction = ai_engine.answer("مو النهاية، الإيرادات", session_id=session_id)["answer"]
        manager = ai_engine.answer("ومديره؟", session_id=session_id)["answer"]
        contract = ai_engine.answer("كم قيمة العقد؟", session_id=session_id)["answer"]
        ai_engine.answer("ارجع للمحفظة", session_id=session_id)
        portfolio_count = ai_engine.answer("كم مشروع عندنا؟", session_id=session_id)["answer"]
        ai_engine.answer("ارجع للباحث", session_id=session_id)
        status = ai_engine.answer("وش وضعه؟", session_id=session_id)["answer"]

    _assert_backlog(backlog)
    assert "إيرادات" in correction and "نهاية المشروع" not in correction
    assert "Manager A" in manager
    assert "1.2 مليون" in contract
    assert "5" in portfolio_count
    assert status
    state = session_context.get_context(session_id)
    assert state["active_project_code"] == "PRJ-001"


_AUTH_HEADERS = {
    "Authorization": "Basic " + base64.b64encode(b"test-user:test-pass").decode(),
}


@pytest.mark.parametrize("enabled", [False, True])
def test_flask_ask_persistent_session_metric_precedence(seeded_db, today, monkeypatch, enabled):
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", enabled)
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setenv("APP_USER", "test-user")
    monkeypatch.setenv("APP_PASS", "test-pass")
    flask_app.config.update(TESTING=True)
    client = flask_app.test_client()

    with patch("modules.semantic_interpreter._get_openai", return_value=_TIMEOUT_CLIENT), \
         patch("modules.understanding._get_openai", return_value=_TIMEOUT_CLIENT), \
         patch("modules.response_composer._get_openai", return_value=_TIMEOUT_CLIENT):
        for query in ("اعطني ملخص الباحث", "متى ينتهي؟"):
            assert client.post("/ask", json={"query": query}, headers=_AUTH_HEADERS).status_code == 200
        response = client.post("/ask", json={"query": "كم باقي إيراد؟"}, headers=_AUTH_HEADERS)
        manager = client.post("/ask", json={"query": "ومديره؟"}, headers=_AUTH_HEADERS)

    assert response.status_code == 200
    _assert_backlog(response.get_json()["answer"])
    assert "Manager A" in manager.get_json()["answer"]
