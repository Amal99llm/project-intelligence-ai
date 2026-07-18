from modules import ai_engine, session_context
from modules.database import BacklogProject, get_session


def _ask(session, text):
    return ai_engine.answer(text, session_id=session)


def test_ranked_project_compares_to_short_explicit_name(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    _ask("context-compare-1", "وش أعلى مشروع ربحية؟")

    result = _ask("context-compare-1", "قارن بينه وبين الباحث")
    state = session_context.get_context("context-compare-1")

    assert result["query_type"] == "project_comparison"
    assert state["last_compared_project_ids"] == ["PRJ-003", "PRJ-001"]
    assert state["last_compared_project_names"] == [
        "برنامج التطوير الأول", "الباحث الاجتماعي الثاني"]


def test_contextual_pair_survives_winner_followup(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    _ask("context-compare-2", "وش أعلى مشروع ربحية؟")
    _ask("context-compare-2", "قارن بينه وبين الباحث الاجتماعي الثاني")
    pair = session_context.get_context("context-compare-2")["last_compared_project_ids"]
    assert pair == ["PRJ-003", "PRJ-001"]

    winner = _ask("context-compare-2", "أيهم أعلى ربحية؟")

    assert winner["query_type"] == "project_comparison"
    assert session_context.get_context("context-compare-2")["last_compared_project_ids"] == pair


def test_qarenah_ma_uses_ranked_project_as_first_side(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    with get_session() as db:
        db.query(BacklogProject).filter_by(project_code="PRJ-002").update(
            {"project_name_ar": "مشروع العقار"})
        db.commit()
    _ask("context-compare-3", "وش أعلى مشروع ربحية؟")

    _ask("context-compare-3", "قارنه مع مشروع العقار")

    assert session_context.get_context("context-compare-3")["last_compared_project_ids"] == [
        "PRJ-003", "PRJ-002"]


def test_selected_comparison_project_can_start_new_contextual_pair(
        seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    with get_session() as db:
        db.query(BacklogProject).filter_by(project_code="PRJ-002").update(
            {"project_name_ar": "مشروع العقار الأول"})
        db.query(BacklogProject).filter_by(project_code="PRJ-003").update(
            {"project_name_ar": "مشروع العقار الثاني"})
        db.add(BacklogProject(project_code="PRJ-ZAMZAM", project_name_ar="مشروع زمزم",
                              status="Ongoing", total_revenue=100, total_cost=80))
        db.commit()

    _ask("context-compare-4", "قارن بين الباحث والعقار")
    _ask("context-compare-4", "1")
    selected = session_context.get_context("context-compare-4")["last_selected_project_id"]
    _ask("context-compare-4", "قارن بينه وبين زمزم")

    assert session_context.get_context("context-compare-4")["last_compared_project_ids"] == [
        selected, "PRJ-ZAMZAM"]
