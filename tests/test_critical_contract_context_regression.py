from modules.ai_engine import _answer_inner, _has_explicit_project_reference
from modules.database import BacklogProject, get_session
from modules.session_context import get_context, update_context
from modules.semantic_dictionary import (
    compose_canonical_fields, detect_requested_field, detect_requested_fields,
    extract_field_concepts,
)


def _send(session_id, query, today):
    text, kind, _, updates = _answer_inner(query, today, get_context(session_id))
    update_context(session_id, last_intent=kind, last_operation=kind, **updates)
    return text, get_context(session_id)


def test_field_semantics_compose_metric_meaning_before_entity_resolution():
    field = detect_requested_field("كم العقد الأساسي؟")
    assert field and field.canonical == "contract_value"
    assert {"contract", "base"}.issubset(extract_field_concepts("وش كان العقد قبل التعديلات؟"))
    assert compose_canonical_fields("وش كان العقد قبل التعديلات؟")[0] == "contract_value"
    assert not _has_explicit_project_reference("كم العقد الأساسي؟", [field])
    assert _has_explicit_project_reference("كم قيمة عقد الباحث؟", detect_requested_fields("كم قيمة عقد الباحث؟"))


def test_semantic_composition_handles_unlisted_phrasings():
    cases = {
        "what was the contract before amendments?": "contract_value",
        "current contract amount including amendments": "total_contract_value",
        "budgeted margin كم؟": "planned_pm_pct",
        "actual profit كم؟": "pl",
        "expected cost at completion": "etc_cost",
        "prior years revenue": "previous_years_rev",
        "current revenue": "revenue_current",
        "كم باقي على مدة العقد؟": "days_remaining",
    }
    for query, expected in cases.items():
        assert expected in compose_canonical_fields(query), query
        assert detect_requested_field(query).canonical == expected, query


def test_contract_followup_context_regression(seeded_db, today, monkeypatch):
    def unavailable():
        raise RuntimeError("AI disabled in deterministic regression test")
    monkeypatch.setattr("modules.understanding._get_openai", unavailable)
    with get_session() as db:
        row = db.query(BacklogProject).filter_by(project_code="PRJ-001").one()
        row.project_name_ar = "الباحث الاجتماعي الثاني"
        row.contract_value = 1_000_000_000
        row.amendment_crs = 120_000_000
        row.total_contract_value = 1_120_000_000
        db.commit()

    first, ctx = _send("critical-contract", "اعطني ملخص مشروع الباحث", today)
    assert ctx["active_project_code"] == "PRJ-001"
    assert "الباحث الاجتماعي الثاني" in first

    total, ctx = _send("critical-contract", "كم قيمة العقد؟", today)
    assert "تقصد أي مشروع" not in total
    assert "1.12 مليار" in total
    assert ctx["active_project_code"] == "PRJ-001"

    base, ctx = _send("critical-contract", "كم العقد الأساسي؟", today)
    assert "هل تقصد مشروع" not in base
    assert "1 مليار" in base
    assert ctx["active_project_code"] == "PRJ-001"
