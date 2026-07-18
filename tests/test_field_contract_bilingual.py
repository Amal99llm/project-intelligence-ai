from datetime import date

from modules.contract_semantics import analyze_contract_request, render_contract_answer
from modules.semantic_dictionary import detect_requested_field
from modules.project_repository import _to_date


def test_field_registry_resolves_arabic_english_and_mixed_business_fields():
    cases = {
        "who is the officer?": "officer_name", "which segment?": "segment",
        "which program is it under?": "program", "what category?": "category",
        "what is the project type?": "project_type", "accounts receivable كم؟": "ar",
        "كم Total Planned Cost؟": "total_planned_cost", "كم Planned Margin؟": "planned_pm_pct",
        "كم ETC Revenue؟": "etc_revenue",
    }
    for query, expected in cases.items():
        field = detect_requested_field(query)
        assert field and field.canonical == expected, query


def test_english_contract_semantics_are_metric_specific():
    cases = {
        "What is the base contract value?": ("get", "contract_value"),
        "How much are the contract amendments?": ("get", "amendment_crs"),
        "When does it end?": ("get", "effective_end_date"),
        "How much time is left?": ("remaining", None),
        "What is the contract duration?": ("duration", None),
    }
    for query, expected in cases.items():
        request = analyze_contract_request(query)
        assert request and request.operation == expected[0], query
        if expected[1]: assert expected[1] in request.metrics


def test_missing_end_date_is_not_rendered_as_zero_days():
    project = {"project_name_en": "Example", "effective_end_date": None, "days_remaining": None, "_response_language": "en"}
    answer = render_contract_answer(project, analyze_contract_request("How much time is left?"), date(2026, 7, 15))
    assert answer == "No valid end date is recorded for this project."
    assert "0 days" not in answer


def test_all_invalid_date_shapes_remain_missing():
    for value in (None, float("nan"), "", "invalid text", 0):
        assert _to_date(value) is None
