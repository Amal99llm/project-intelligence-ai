from unittest.mock import patch

import config
from modules import query_builder, query_executor, response_formatter
from modules.semantic_dictionary import detect_small_talk
from modules.understanding import (
    PORTFOLIO_FILTER,
    PORTFOLIO_RANKING,
    PORTFOLIO_SUMMARY,
    _llm_understand,
    understand,
)


def test_saudi_portfolio_summary_variants_do_not_need_llm():
    for query in ("وش عندنا؟", "وش وضع المحفظة؟", "عطني الزبدة", "وش أخبار المشاريع؟"):
        result = understand(query, {})
        assert result.intent == PORTFOLIO_SUMMARY, query
        assert result.method != "llm"


def test_saudi_contract_ranking_variants():
    for query in ("وش أكبر مشروع؟", "أعلى عقد؟", "أكبر قيمة؟", "مين أكبر مشروع عندنا؟"):
        result = understand(query, {})
        assert result.intent == PORTFOLIO_RANKING, query
        spec = query_builder.build_query(query, "2026-07-17")
        assert spec["sort"] == {"column": "total_contract_value", "direction": "DESC"}


def test_ongoing_count_is_a_portfolio_filter():
    result = understand("كم مشروع شغال؟", {})
    assert result.intent == PORTFOLIO_FILTER
    spec = query_builder.build_query("كم مشروع شغال؟", "2026-07-17")
    assert spec["filters"] == [{"column": "status", "op": "==", "value": "Ongoing"}]
    assert spec["aggregation"] == {"func": "COUNT", "column": None}


def test_compound_ongoing_and_low_margin_filter():
    query = "وش المشاريع الجارية اللي هامشها أقل من 5%؟"
    assert understand(query, {}).intent == PORTFOLIO_FILTER
    spec = query_builder.build_query(query, "2026-07-17")
    assert {item["column"] for item in spec["filters"]} == {"status", "profit_pct"}
    assert next(item for item in spec["filters"] if item["column"] == "profit_pct")["value"] == 5.0


def test_missing_ai_key_fails_fast_without_constructing_client():
    query_builder._openai_client = None
    with patch.object(config, "AZURE_OPENAI_KEY", ""):
        with patch("openai.AzureOpenAI") as constructor:
            try:
                query_builder.build_query("صياغة غامضة تماماً", "2026-07-17")
            except query_builder.QueryBuildError:
                pass
            else:
                raise AssertionError("missing configuration must stop query building")
            constructor.assert_not_called()


def test_invalid_llm_contract_is_rejected_to_safe_fallback():
    class Function:
        arguments = '{"intent":"invented","scope":"other","is_followup":false,"confidence":2}'

    class Call:
        function = Function()

    class Message:
        tool_calls = [Call()]

    class Choice:
        message = Message()

    class Completions:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"choices": [Choice()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    with patch("modules.understanding._get_openai", return_value=client):
        result = _llm_understand("؟؟؟", {}, "")
    assert result.method == "fallback"
    assert result.intent != "invented"


def test_farewell_is_not_shadowed_by_greeting():
    assert detect_small_talk("مع السلامة") == "bye"


def test_status_counts_share_portfolio_count_route_and_wording():
    cases = (
        ("كم المشاريع الجارية؟", "Ongoing", "جاريًا"),
        ("كم المشاريع المكتملة؟", ["Completed", "Closed"], "مكتملًا"),
    )
    for query, expected_status, expected_word in cases:
        routed = understand(query, {})
        assert routed.intent == PORTFOLIO_FILTER
        assert routed.method == "structural"
        spec = query_builder.build_query(query, "2026-07-17")
        assert spec["aggregation"]["func"] == "COUNT"
        assert spec["filters"][0]["value"] == expected_status
        payload = {
            "intent": routed.intent,
            "spec": spec,
            "aggregation": spec["aggregation"],
            "aggregation_result": 53,
            "rows": [],
        }
        answer = response_formatter.format_answer(query, payload)
        assert expected_word in answer
        assert "مطابق" not in answer


def test_ranking_direction_metric_and_formatter_are_consistent():
    cases = (
        ("وش أصغر مشروع؟", "total_contract_value", "ASC"),
        ("وش أكبر مشروع؟", "total_contract_value", "DESC"),
        ("أعلى مشروع ربحية؟", "pl", "DESC"),
        ("أقل مشروع ربحية؟", "pl", "ASC"),
    )
    projects = [
        {"project_code": "A", "project_name_ar": "الأول", "total_contract_value": 100, "profit_pct": 5, "pl": 10},
        {"project_code": "B", "project_name_ar": "الثاني", "total_contract_value": 200, "profit_pct": 20, "pl": 30},
    ]
    for query, metric, direction in cases:
        routed = understand(query, {})
        assert routed.intent == PORTFOLIO_RANKING
        spec = query_builder.build_query(query, "2026-07-17")
        assert spec["sort"] == {"column": metric, "direction": direction}
        result = query_executor.execute(spec, projects=projects)
        payload = {"intent": routed.intent, **result}
        answer = response_formatter.format_answer(query, payload)
        assert "مطابق" not in answer
        assert ("الأعلى" if direction == "DESC" else "الأقل") in answer
