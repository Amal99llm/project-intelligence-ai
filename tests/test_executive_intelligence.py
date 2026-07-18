from modules.ai_engine import _answer_inner
from modules.executive_intelligence import (
    AnalysisType, classify_executive_request, evaluate_project_attention,
)


def test_executive_classifier_produces_controlled_requests():
    cases = {
        "وش أهم ثلاث مشاريع تحتاج متابعة؟": AnalysisType.MANAGEMENT_PRIORITIES,
        "لو عندي اجتماع بعد خمس دقائق، وش أهم الأشياء اللي لازم أعرفها؟": AnalysisType.EXECUTIVE_MEETING_BRIEF,
        "وين أكبر المخاطر؟": AnalysisType.RISK_ANALYSIS,
        "وش العقود اللي بتنتهي قريب؟": AnalysisType.CONTRACT_EXPIRATION,
        "أي العقود تحتاج تجديد؟": AnalysisType.RENEWAL_CANDIDATES,
    }
    for query, expected in cases.items():
        request = classify_executive_request(query)
        assert request is not None
        assert expected in request.analyses


def test_attention_is_multi_signal_explainable_and_ignores_contract_size(today):
    assessment = evaluate_project_attention({
        "project_code": "RISKY", "project_name_ar": "مشروع متعثر", "status": "Ongoing",
        "days_remaining": -20, "net_profit": -100_000, "profit_pct": -5,
        "risk": 2_000_000, "total_contract_value": 10,
    }, today)
    assert assessment.score >= 80
    assert len(assessment.reasons) >= 3
    assert all(isinstance(reason, str) for reason in assessment.reasons)


def test_management_priority_never_falls_into_project_lookup(seeded_db, today):
    text, kind, _, updates = _answer_inner("وش أهم ثلاث مشاريع تحتاج متابعة؟", today, {})
    assert kind == "executive_analysis"
    assert "تستحق المتابعة" in text
    assert "لم أتمكن من العثور" not in text
    assert updates["last_result_scope"] == "portfolio"


def test_risk_analysis_uses_risk_not_contract_value(seeded_db, today):
    text, kind, _, _ = _answer_inner("وين أكبر المخاطر؟", today, {})
    assert kind == "executive_analysis"
    assert "المخاطر المالية" in text or "قيم مخاطر" in text
    assert "قيمة العقد" not in text


def test_meeting_brief_contains_all_executive_sections(seeded_db, today):
    text, kind, _, _ = _answer_inner(
        "لو عندي اجتماع بعد خمس دقائق، وش أهم الأشياء اللي لازم أعرفها؟", today, {}
    )
    assert kind == "executive_analysis"
    for expected in ("الملخص التنفيذي", "تستحق المتابعة", "العقود النشطة", "المخاطر", "التوصية"):
        assert expected in text


def test_expiration_states_default_window_and_excludes_completed(seeded_db, today):
    text, kind, _, _ = _answer_inner("وش العقود اللي بتنتهي قريب؟", today, {})
    assert kind == "executive_analysis"
    assert "خلال 90 يومًا" in text
    assert "PRJ-005" not in text


def test_renewal_candidates_are_cautious_and_evidenced(seeded_db, today):
    text, kind, _, _ = _answer_inner("أي العقود تحتاج تجديد؟", today, {})
    assert kind == "executive_analysis"
    assert "ليست توصية مؤكدة بالتجديد" in text
    assert "المشروع ما زال نشطًا" in text
    assert "نسبة الإنجاز" in text or "الأعمال المتبقية" in text


def test_compound_executive_request_keeps_every_section(seeded_db, today):
    query = "أنا داخل اجتماع مع الإدارة. عطني ملخص المحفظة، أهم ثلاث مشاريع تحتاج متابعة، وأقرب ثلاثة عقود للانتهاء."
    text, kind, _, _ = _answer_inner(query, today, {})
    assert kind == "executive_analysis"
    assert "الملخص التنفيذي" in text
    assert "تستحق المتابعة" in text
    assert "العقود النشطة" in text
