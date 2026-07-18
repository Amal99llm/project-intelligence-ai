from modules import ai_engine, session_context
from modules.conversation_state import (
    CANONICAL_FIELDS, set_active_project, set_comparison_context,
    set_ranked_project, transition_for_turn,
)


def _ask(session, query):
    return ai_engine.answer(query, session_id=session)


def test_transition_helpers_define_complete_canonical_contract():
    ranked = set_ranked_project("P1", "One", "rank", "portfolio_ranking", "profit", "highest")
    assert ranked["last_selected_project_id"] == "P1"
    assert ranked["last_ranked_project_name"] == "One"
    assert ranked["last_scope"] == "project"
    compared = set_comparison_context(["P1", "P2"], ["One", "Two"], "compare", "risk")
    assert compared["last_scope"] == "comparison"
    assert compared["last_compared_project_names"] == ["One", "Two"]
    assert set(CANONICAL_FIELDS) <= set(session_context._empty_context())


def test_none_does_not_overwrite_valid_canonical_reference():
    current = set_active_project("P1", "One", "summary", "project_summary")
    transition = transition_for_turn(current, "small_talk", {"last_scope": None,
                                                              "active_project_code": None})
    assert "last_scope" not in transition
    assert "last_selected_project_id" not in transition


def test_comparison_side_selection_switches_to_project_but_preserves_pair(
        seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    session_context.update_context(
        "pair-side", **set_comparison_context(
            ["PRJ-001", "PRJ-003"],
            ["الباحث الاجتماعي الثاني", "برنامج التطوير الأول"], "compare"))

    answer = _ask("pair-side", "من مدير الثاني؟")
    state = session_context.get_context("pair-side")

    assert "Manager C" in answer["answer"]
    assert state["last_scope"] == "project"
    assert state["last_selected_project_id"] == "PRJ-003"
    assert state["last_compared_project_ids"] == ["PRJ-001", "PRJ-003"]


def test_explicit_portfolio_scope_overrides_active_project(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    session_context.update_context("scope-switch", **set_active_project(
        "PRJ-001", "الباحث الاجتماعي الثاني", "summary", "project_summary"))

    result = _ask("scope-switch", "لا، أقصد المحفظة: كم الإيرادات؟")

    assert result["query_type"] == "portfolio_kpi"
    assert session_context.get_context("scope-switch")["last_scope"] == "portfolio"


def test_sessions_are_isolated_and_reset_is_atomic():
    session_context.update_context("session-a", **set_ranked_project(
        "A", "Alpha", "rank", "portfolio_ranking", "profit", "highest"))
    session_context.update_context("session-b", **set_comparison_context(
        ["B", "C"], ["Beta", "Charlie"], "compare"))

    assert session_context.get_context("session-a")["last_compared_project_ids"] == []
    assert session_context.get_context("session-b")["last_selected_project_id"] is None
    session_context.reset_conversation_context("session-a")
    reset = session_context.get_context("session-a")
    assert reset["last_selected_project_id"] is None
    assert reset["last_ranked_project_id"] is None
    assert session_context.get_context("session-b")["last_compared_project_ids"] == ["B", "C"]
