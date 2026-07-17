from modules import session_context
from modules.ai_engine import _answer_inner

import pytest


@pytest.fixture()
def no_ai(monkeypatch):
    def unavailable():
        raise RuntimeError("AI disabled in deterministic architecture test")
    monkeypatch.setattr("modules.understanding._get_openai", unavailable)
    monkeypatch.setattr("modules.response_composer._get_openai", unavailable)


def _turn(session_id, query, today):
    state = session_context.get_context(session_id)
    text, kind, _, updates = _answer_inner(query, today, state)
    session_context.update_context(session_id, **updates)
    return text, kind, session_context.get_context(session_id)


def test_canonical_state_commit_is_deeply_isolated():
    pending = {"candidates": [{"project_code": "PRJ-001", "display_name": "One"}],
               "original_query": "original", "kind": "lookup"}
    session_context.update_context(
        "canonical-state", active_project_code="PRJ-001",
        active_project_display_name="One", pending_project_confirmation=pending,
        last_user_intent="project_summary", last_requested_metric="net_profit",
    )
    pending["candidates"][0]["project_code"] = "MUTATED"
    state = session_context.get_context("canonical-state")
    assert state["active_project_id"] == "PRJ-001"
    assert state["active_project_name"] == "One"
    assert state["pending_original_request"] == "original"
    assert state["pending_disambiguation_options"][0]["project_code"] == "PRJ-001"
    assert state["last_intent"] == "project_summary"
    assert state["last_metrics"] == ["net_profit"]


def test_compound_named_request_executes_every_metric_and_switches_context(seeded_db, today, no_ai):
    session_context.update_context(
        "compound-plan", active_project_code="PRJ-001", last_project_code="PRJ-001",
    )
    answer, _, state = _turn(
        "compound-plan",
        "For Al Noor Energy Project give me the profit, margin, and end date",
        today,
    )
    assert state["active_project_id"] == "PRJ-002"
    assert set(state["last_metrics"]) >= {"pl", "profit_pct", "effective_end_date"}
    assert "150,000" in answer or "-150" in answer
    assert "%" in answer
    assert "2026-" not in answer
