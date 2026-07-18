"""Permanent regressions copied from the failed production conversation."""

import pytest

from modules.ai_engine import _answer_inner
from modules.database import BacklogProject, get_session
from modules.session_context import get_context, update_context
from modules.semantic_dictionary import detect_requested_field


@pytest.fixture()
def no_ai(monkeypatch):
    def unavailable():
        raise RuntimeError("AI disabled in deterministic regression test")

    monkeypatch.setattr("modules.understanding._get_openai", unavailable)
    monkeypatch.setattr("modules.response_composer._get_openai", unavailable)


def _send(session_id: str, query: str, today) -> tuple[str, str]:
    text, kind, _, updates = _answer_inner(query, today, get_context(session_id))
    committed = {"last_intent": kind, "last_operation": kind}
    committed.update(updates)
    update_context(session_id, **committed)
    return text, kind


def _name_researcher(seeded_db, program="برنامج الأبحاث الاجتماعية") -> None:
    with get_session() as db:
        row = db.query(BacklogProject).filter_by(project_code="PRJ-001").one()
        row.project_name_ar = "الباحث الاجتماعي الثاني"
        row.program = program
        db.commit()


def test_comparison_with_nonexistent_second_project(seeded_db, today, no_ai):
    _name_researcher(seeded_db)
    _send("missing-comparison", "اعطني ملخص مشروع الباحث", today)
    response, kind = _send(
        "missing-comparison", "قارن الباحث مع مشروع لا وجود له إطلاقًا", today
    )
    assert kind == "project_comparison"
    assert "لم أتمكن" in response or "أكثر من مشروع" in response
    assert "جاري" not in response


def test_confirmation_preserves_original_intent(seeded_db, today, no_ai):
    _name_researcher(seeded_db)
    prompt, _ = _send("confirmation-intent", "اعطني ملخص عن الباخث", today)
    assert "هل تقصد" in prompt
    response, kind = _send("confirmation-intent", "اي", today)
    assert kind == "project_summary"
    assert "التقدم" in response and "قيمة العقد" in response
    assert "غير مدعوم" not in response


def test_program_field_returns_program_not_project_code(seeded_db, today, no_ai):
    _name_researcher(seeded_db, program="PRJ-001")
    _send("program-field", "اعطني ملخص مشروع الباحث", today)
    response, kind = _send("program-field", "وش البرنامج؟", today)
    assert kind == "project_kpi"
    assert "PRJ-001" not in response
    assert "غير متوفر" in response


def test_remaining_duration_followup(seeded_db, today, no_ai):
    for wording in ("كم باقي له؟", "كم باقي؟", "المدة الباقية"):
        assert detect_requested_field(wording).canonical == "days_remaining"
    _name_researcher(seeded_db)
    _send("remaining-duration", "اعطني ملخص مشروع الباحث", today)
    _send("remaining-duration", "متى ينتهي؟", today)
    response, kind = _send("remaining-duration", "كم باقي له؟", today)
    assert kind == "project_kpi"
    assert "تقصد أي مشروع" not in response
    assert "باقي على نهاية المشروع" in response


def test_margin_between_filter_is_portfolio_scope(seeded_db, today, no_ai):
    _name_researcher(seeded_db)
    _send("margin-between", "اعطني ملخص مشروع الباحث", today)
    response, kind = _send(
        "margin-between", "وش المشاريع اللي هامشها بين ٥٪ و١٠٪؟", today
    )
    assert kind == "portfolio_filter"
    assert "45.44" not in response
