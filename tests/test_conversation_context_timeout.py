"""Focused regressions for deterministic conversational references."""

from modules import ai_engine, session_context, understanding
from modules.database import BacklogProject, get_session


def _ask(session_id, text):
    return ai_engine.answer(text, session_id=session_id)


def _force_timeout(monkeypatch):
    def timed_out():
        raise TimeoutError("Request timed out.")
    monkeypatch.setattr(understanding, "_get_openai", timed_out)


def test_portfolio_count_followup_counts_53_ongoing(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    with get_session() as db:
        for index in range(49):
            db.add(BacklogProject(project_code=f"ACTIVE-{index:03d}",
                                  project_name_ar=f"مشروع جاري {index}", status="Ongoing"))
        for index in range(82):
            db.add(BacklogProject(project_code=f"DONE-{index:03d}",
                                  project_name_ar=f"مشروع مكتمل {index}", status="Completed"))
        db.commit()

    first = _ask("portfolio-count", "كم مشروع عندنا؟")
    second = _ask("portfolio-count", "منهم كم جاري؟")

    assert "136" in first["answer"]
    assert "53" in second["answer"]
    assert session_context.get_context("portfolio-count")["last_scope"] == "portfolio"


def test_ranked_project_is_reused_by_details_manager_and_backlog(
        seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    session_id = "ranked-followups"

    ranked = _ask(session_id, "وش أعلى مشروع ربحية؟")
    details = _ask(session_id, "طيب اعطني تفاصيله")
    manager = _ask(session_id, "ومن مديره؟")
    backlog = _ask(session_id, "وكم الباكلوج حقه؟")
    state = session_context.get_context(session_id)

    assert "برنامج التطوير الأول" in ranked["answer"]
    assert "برنامج التطوير الأول" in details["answer"]
    assert "Manager C" in manager["answer"]
    assert "300" in backlog["answer"]
    assert state["last_ranked_project_id"] == "PRJ-003"
    assert state["last_selected_project_id"] == "PRJ-003"


def test_comparison_pair_survives_disambiguation_and_followups(
        seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    with get_session() as db:
        db.query(BacklogProject).filter_by(project_code="PRJ-002").update(
            {"project_name_ar": "مشروع العقار الأول", "risk": 10})
        db.query(BacklogProject).filter_by(project_code="PRJ-003").update(
            {"project_name_ar": "مشروع العقار الثاني", "risk": 90})
        db.commit()
    session_id = "comparison-followups"

    _ask(session_id, "قارن بين مشروع الباحث والعقار")
    _ask(session_id, "1")
    pair = session_context.get_context(session_id)["last_compared_project_ids"]
    profit = _ask(session_id, "أيهم أعلى ربحية؟")
    risk = _ask(session_id, "وأيهما مخاطره أعلى؟")

    assert len(pair) == 2
    assert session_context.get_context(session_id)["last_compared_project_ids"] == pair
    assert profit["query_type"] == "project_comparison"
    assert risk["query_type"] == "project_comparison"


def test_failed_lookup_does_not_create_pronoun_context(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    session_id = "missing-project"

    _ask(session_id, "اعطني ملخص عن مشروع ناسا")
    followup = _ask(session_id, "ملخص عنه")

    assert "ما عندي مشروع محدد سابقًا" in followup["answer"]
    assert session_context.get_context(session_id)["last_selected_project_id"] is None


def test_all_context_followups_survive_llm_timeout(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    _force_timeout(monkeypatch)

    _ask("timeout-portfolio", "كم مشروع عندنا؟")
    assert "4" in _ask("timeout-portfolio", "منهم كم جاري؟")["answer"]

    _ask("timeout-ranked", "وش أعلى مشروع ربحية؟")
    assert "برنامج التطوير الأول" in _ask("timeout-ranked", "طيب اعطني تفاصيله")["answer"]
    assert "Manager C" in _ask("timeout-ranked", "ومن مديره؟")["answer"]
    assert "300" in _ask("timeout-ranked", "وكم الباكلوج حقه؟")["answer"]

    _ask("timeout-comparison", "قارن بين الباحث والنور")
    pair = session_context.get_context("timeout-comparison")["last_compared_project_ids"]
    assert len(pair) == 2
    _ask("timeout-comparison", "أيهم أعلى ربحية؟")
    _ask("timeout-comparison", "وأيهما مخاطره أعلى؟")
    assert session_context.get_context("timeout-comparison")["last_compared_project_ids"] == pair
