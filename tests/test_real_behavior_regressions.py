from datetime import date
import pytest

from modules import query_builder, query_executor, response_formatter
from modules.ai_engine import _answer_inner, _try_resolve_pending
from modules.session_context import get_context, update_context
from modules.followup_gate import check as followup_check
from modules.semantic_dictionary import detect_requested_fields
from modules.project_entity_resolver import resolve_project
from modules.contract_semantics import parse_future_period_days
from modules.understanding import PORTFOLIO_FILTER, PORTFOLIO_RANKING, understand


def test_profitability_defaults_to_pl_but_margin_is_percentage():
    assert query_builder.build_query("أعلى مشروع ربحية؟", "2026-07-17")["sort"] == {
        "column": "pl", "direction": "DESC"
    }
    assert query_builder.build_query("أعلى هامش ربح؟", "2026-07-17")["sort"] == {
        "column": "profit_pct", "direction": "DESC"
    }


def test_backlog_ranking_variants_do_not_use_project_lookup():
    for query in ("أعلى Backlog؟", "أعلى أعمال متبقية", "أكبر باك لوق", "وش المشروع اللي باقي فيه أكثر؟"):
        routed = understand(query, {})
        assert routed.intent == PORTFOLIO_RANKING, query
        spec = query_builder.build_query(query, "2026-07-17")
        assert spec["sort"] == {"column": "backlog", "direction": "DESC"}
        assert spec["limit"] == 1


def test_margin_filter_formats_percentages_and_total_count():
    query = "وش المشاريع اللي هامشها أقل من 5%؟"
    assert understand(query, {}).intent == PORTFOLIO_FILTER
    spec = query_builder.build_query(query, "2026-07-17")
    projects = [
        {"project_code": "A", "project_name_ar": "أ", "profit_pct": -61.47212618334063},
        {"project_code": "B", "project_name_ar": "ب", "profit_pct": 0.0},
        {"project_code": "C", "project_name_ar": "ج", "profit_pct": 12.0},
    ]
    result = query_executor.execute(spec, projects=projects)
    answer = response_formatter.format_answer(query, {"intent": PORTFOLIO_FILTER, **result})
    assert "يوجد 2 مشروع" in answer
    assert "هامش الربح: -61.47%" in answer
    assert "هامش الربح: 0.00%" in answer
    assert "مطابق" not in answer


def test_temporal_filters_use_fixed_date_and_effective_fields():
    ending = "وش المشاريع اللي تنتهي خلال شهر؟"
    started = "وش المشاريع اللي بدأت هالسنة؟"
    for query in (ending, started):
        assert understand(query, {}).intent == PORTFOLIO_FILTER
    end_spec = query_builder.build_query(ending, "2026-07-17")
    assert end_spec["filters"] == [{"column": "days_remaining", "op": "between", "value": 0, "value2": 30}]
    start_spec = query_builder.build_query(started, "2026-07-17")
    start_filter = start_spec["filters"][0]
    assert start_filter["column"] == "start_date"
    assert start_filter["value"] == date(2026, 1, 1)
    assert start_filter["value2"] == date(2026, 12, 31)


def test_multi_metric_questions_keep_all_requested_fields():
    cases = {
        "متى تاريخ البداية والنهاية؟": {"start_date", "effective_end_date"},
        "كم ربحه وهامشه؟": {"pl", "profit_pct"},
        "مين مديره ومتى ينتهي؟": {"project_manager", "effective_end_date"},
        "وش حالته ونسبة إنجازه؟": {"status", "progress_completed"},
    }
    for query, expected in cases.items():
        assert expected.issubset({field.canonical for field in detect_requested_fields(query)}), query


def test_natural_metric_templates_and_arabic_dates():
    project = {
        "total_cost": 259_140_000, "pl": 215_860_000, "profit_pct": 45.4,
        "risk": 40_540_000, "start_date": date(2024, 1, 7),
        "effective_end_date": date(2029, 4, 17),
    }
    answer = response_formatter.format_project_metrics(
        project, ["total_cost", "pl", "profit_pct", "risk"]
    )
    assert "بلغت تكاليف المشروع" in answer
    assert "يحقق المشروع صافي ربح" in answer
    assert "هامش الربح الحالي" in answer
    assert "قيمة المخاطر المسجلة" in answer
    dates = response_formatter.format_project_metrics(project, ["start_date", "effective_end_date"])
    assert "7 يناير 2024" in dates and "17 أبريل 2029" in dates


def test_delay_answer_is_direct_and_contains_date_or_days():
    on_time = {"days_remaining": 100, "effective_end_date": date(2029, 4, 17), "status": "Ongoing"}
    delayed = {"days_remaining": -198, "effective_end_date": date(2026, 1, 1), "status": "Ongoing"}
    assert response_formatter.format_delay_status(on_time).startswith("لا،")
    assert "17 أبريل 2029" in response_formatter.format_delay_status(on_time)
    assert response_formatter.format_delay_status(delayed).startswith("نعم،")
    assert "198 يوم" in response_formatter.format_delay_status(delayed)


def test_explicit_project_name_beats_active_project_gate():
    ctx = {"active_project_code": "OLD", "last_project_code": "OLD"}
    assert not followup_check("متى ينتهي عقد مشروع العقار", ctx).fires


def test_disambiguation_keeps_original_options_for_and_second():
    options = [
        {"project_code": "A", "display_name": "مشروع أ"},
        {"project_code": "B", "display_name": "مشروع ب"},
    ]
    ctx = {
        "pending_project_confirmation": {"candidates": options, "kind": "lookup"},
        "selected_disambiguation_index": 0,
    }
    code, _, state = _try_resolve_pending("والثاني", ctx)
    assert code == "B"
    assert state["candidates"] == options


def _turn(session_id, query, today):
    text, kind, source, updates = _answer_inner(query, today, get_context(session_id))
    if updates:
        update_context(session_id, **updates)
    return text, kind, get_context(session_id)


@pytest.fixture()
def no_ai(monkeypatch):
    def unavailable():
        raise RuntimeError("AI disabled in deterministic regression test")
    monkeypatch.setattr("modules.understanding._get_openai", unavailable)
    monkeypatch.setattr("modules.response_composer._get_openai", unavailable)


def test_named_project_switch_then_followup_stays_on_new_project(seeded_db, today, no_ai):
    _, _, first = _turn("switch", "ملخص مشروع Social Researcher II", today)
    assert first["active_project_code"] == "PRJ-001"

    end_answer, _, second = _turn("switch", "متى ينتهي عقد مشروع Al Noor Energy Project", today)
    assert second["active_project_code"] == "PRJ-002"
    assert "25 يوليو 2026" in end_answer

    profit_answer, _, third = _turn("switch", "كم ربحه؟", today)
    assert third["active_project_code"] == "PRJ-002"
    assert "150,000" in profit_answer or "-150" in profit_answer


def test_disambiguation_first_then_and_second_uses_same_options(seeded_db, today, no_ai):
    prompt, _, initial = _turn("ambiguity", "Development Program", today)
    options = initial["last_disambiguation_options"]
    assert len(options) >= 2
    assert "1." in prompt and "2." in prompt

    _, _, selected_first = _turn("ambiguity", "1", today)
    assert selected_first["active_project_code"] == options[0]["project_code"]
    assert selected_first["last_disambiguation_options"] == options

    _, _, selected_second = _turn("ambiguity", "والثاني", today)
    assert selected_second["active_project_code"] == options[1]["project_code"]
    assert selected_second["last_disambiguation_options"] == options


def test_project_name_alone_clears_previous_metric_behavior(seeded_db, today, no_ai):
    update_context("name-only", active_project_code="PRJ-001", last_project_code="PRJ-001",
                   last_requested_metric="segment")
    answer, kind, ctx = _turn("name-only", "Al Noor Energy Project", today)
    assert ctx["active_project_code"] == "PRJ-002"
    assert kind == "project_summary"
    assert "Segment" not in answer and "segment:" not in answer.lower()


def test_multi_field_and_delay_answers_run_through_active_context(seeded_db, today, no_ai):
    update_context("details", active_project_code="PRJ-003", last_project_code="PRJ-003")
    dates, _, _ = _turn("details", "متى تاريخ البداية والنهاية؟", today)
    assert "بدأ المشروع في" in dates and "ومن المخطط أن ينتهي" in dates
    assert "2026-" not in dates

    delay, _, _ = _turn("details", "هل المشروع متأخر؟", today)
    assert delay.startswith(("نعم،", "لا،"))
    assert "2027" in delay or "يوم" in delay


def test_saudi_typo_project_resolution_requests_confirmation():
    projects = [{
        "project_code": "R", "project_name_ar": "الباحث الاجتماعي الثاني",
        "project_name_en": "Social Researcher II",
    }]
    resolution = resolve_project("اعطيني ملخص عن الباخث", projects)
    assert resolution.status in {"confirmation", "matched"}
    assert resolution.candidates[0].project_code == "R"


@pytest.mark.parametrize("reply", ["نعم", "ايوه", "إيه", "يب", "صح", "صحيح", "أكيد", "هو", "هذا"])
def test_saudi_confirmation_variants_execute_pending_request(reply):
    ctx = {"pending_project_confirmation": {
        "candidates": [{"project_code": "R", "display_name": "الباحث الاجتماعي الثاني"}],
        "kind": "lookup", "original_query": "اعطني ملخص الباحث",
    }}
    assert _try_resolve_pending(reply, ctx)[0] == "R"


def test_contract_dialogue_sequence_uses_active_project(seeded_db, today, no_ai):
    update_context("contract", active_project_code="PRJ-001", last_project_code="PRJ-001",
                   last_result_type="project_summary")
    questions = [
        ("كم قيمة العقد؟", "قيمة العقد الإجمالية"),
        ("كم العقد الأساسي؟", "قيمة العقد الأساسية"),
        ("كم قيمة التعديلات؟", "قيمة تعديلات العقد"),
        ("بعد التعديلات كم صار؟", "بعد التعديلات"),
        ("متى ينتهي؟", "ينتهي المشروع في"),
        ("كم باقي له؟", "باقي على نهاية المشروع"),
        ("كم مدة عقده؟", "مدة المشروع المخططة"),
        ("هل عقده قرب ينتهي؟", "العقد"),
        ("هل صار تعديل على تاريخ النهاية؟", "تاريخ النهاية"),
    ]
    for question, expected in questions:
        answer, _, ctx = _turn("contract", question, today)
        assert ctx["active_project_code"] == "PRJ-001"
        assert expected in answer, (question, answer)


def test_social_message_beats_pending_project_choice(seeded_db, today, no_ai):
    pending = {"candidates": [
        {"project_code": "PRJ-001", "display_name": "أ"},
        {"project_code": "PRJ-002", "display_name": "ب"},
    ], "kind": "lookup"}
    update_context("bye", active_project_code="PRJ-001", pending_project_confirmation=pending)
    answer, kind, ctx = _turn("bye", "في أمان الله", today)
    assert kind == "small_talk"
    assert "أمان الله" in answer or "مع السلامة" in answer or "اللقاء" in answer
    assert ctx["pending_project_confirmation"] == pending


def test_recent_project_navigation_switches_back(seeded_db, today, no_ai):
    update_context("nav", active_project_code="PRJ-001", last_result_type="project_summary")
    update_context("nav", active_project_code="PRJ-002", last_result_type="project_summary")
    answer, _, ctx = _turn("nav", "طيب السابق", today)
    assert ctx["active_project_code"] == "PRJ-001"
    assert "Social Researcher II" in answer or "الباحث" in answer


def test_colloquial_future_period_parser_and_query_builder():
    cases = {
        "وش المشاريع اللي تنتهي خلال أسبوع؟": 7,
        "وش المشاريع اللي تنتهي الأسبوعين الجاية؟": 14,
        "وش المشاريع اللي تنتهي خلال 15 يوم؟": 15,
        "وش المشاريع اللي تنتهي خلال ثلاث شهور؟": 90,
    }
    for query, expected in cases.items():
        assert parse_future_period_days(query) == expected
        spec = query_builder.build_query(query, "2026-07-17")
        assert spec["filters"][0]["value2"] == expected


def test_contextual_pronouns_and_colloquial_financial_metrics(seeded_db, today, no_ai):
    update_context("pronouns", active_project_code="PRJ-001", last_project_code="PRJ-001",
                   last_result_type="project_summary")
    summary, kind, ctx = _turn("pronouns", "هذا المشروع", today)
    assert kind == "project_summary" and ctx["active_project_code"] == "PRJ-001"
    assert "PRJ-001" in summary or "Social Researcher II" in summary or "الباحث" in summary
    assert "total_cost" in {field.canonical for field in detect_requested_fields("كم صرفنا فيه؟")}
    assert "total_revenue" in {field.canonical for field in detect_requested_fields("وش دخلنا منه؟")}
