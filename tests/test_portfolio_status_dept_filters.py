"""Regression coverage for the Arabic status/department portfolio-filter bug.

"عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة" (give me all active
projects in the Specialized Projects department) was being misrouted:
"النشطة" (active) and "المتخصصة" (specialized) were not recognized as a
status/department pair, so the query fell through to an ungrounded LLM
value-guessing path that matched zero rows -- surfacing a generic "give me
the project name" message that reads as a project-name-lookup failure.

These tests pin the centralized STATUS_ALIASES/DEPT_ALIASES vocabulary
(modules.semantic_dictionary), the deterministic filter spec it feeds
(modules.query_builder), the routing rule that must reach it before
project-name resolution (modules.understanding / is_explicit_portfolio_filter),
the enum safety net (modules.query_schema), the context-aware zero-match
wording (modules.response_formatter), and that plain project-name lookups
(including genuinely ambiguous ones) still resolve correctly and are never
guessed at.
"""

from modules import query_builder, query_executor, query_schema, response_formatter
from modules.project_entity_resolver import resolve_project
from modules.understanding import PORTFOLIO_FILTER, understand

TODAY = "2026-07-19"


# ── Status vocabulary ───────────────────────────────────────────────────────

def test_single_status_aliases_route_to_portfolio_filter():
    cases = (
        ("المشاريع النشطة", "Ongoing"),
        ("المشاريع الجارية", "Ongoing"),
        ("المشاريع المنتهية", "Completed"),
        ("المشاريع المتوقفة", "On-hold"),
    )
    for query, expected_status in cases:
        routed = understand(query, {})
        assert routed.intent == PORTFOLIO_FILTER, query
        assert routed.method == "structural", query
        spec = query_builder.build_query(query, TODAY)
        assert spec["filters"] == [{"column": "status", "op": "==", "value": expected_status}], query


# ── Department vocabulary + compound status/department filters ─────────────

def test_status_and_department_compound_filters():
    cases = (
        ("عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة", "Ongoing", "BPO-Specialized Pr"),
        ("مشاريع إدارة التفتيش الجارية", "Ongoing", "BPO - Inspection"),
        ("المشاريع الصحية المنتهية", "Completed", "BPO - Health"),
    )
    for query, expected_status, expected_dept in cases:
        routed = understand(query, {})
        assert routed.intent == PORTFOLIO_FILTER, query

        # The routing rule: a status/department filter must be caught before
        # project-name resolution ever runs (this is what ai_engine checks
        # ahead of build_plan/resolve_project).
        assert query_builder.is_explicit_portfolio_filter(query, TODAY), query

        spec = query_builder.build_query(query, TODAY)
        by_column = {f["column"]: f["value"] for f in spec["filters"]}
        assert by_column == {"status": expected_status, "dept": expected_dept}, query


def test_exact_reported_bug_query_end_to_end():
    """The literal query from the bug report, verified through execution and
    response formatting, not just spec-building."""
    query = "عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة"
    spec = query_builder.build_query(query, TODAY)
    projects = [
        {"project_code": "P1", "project_name_ar": "مشروع تفتيش هيئة العلا",
         "status": "Ongoing", "dept": "BPO-Specialized Pr"},
        {"project_code": "P2", "project_name_ar": "مشروع خارج القسم",
         "status": "Ongoing", "dept": "BPO - Inspection"},
        {"project_code": "P3", "project_name_ar": "مشروع متخصص متوقف",
         "status": "On-hold", "dept": "BPO-Specialized Pr"},
    ]
    result = query_executor.execute(spec, projects=projects)
    assert [row["project_code"] for row in result["rows"]] == ["P1"]
    payload = {"intent": PORTFOLIO_FILTER, "spec": spec, **result}
    answer = response_formatter.format_answer(query, payload)
    assert "تفتيش هيئة العلا" in answer
    assert "اسم المشروع" not in answer


# ── Enum safety net ──────────────────────────────────────────────────────

def test_query_schema_rejects_unrecognized_status_literal():
    """Guards the failure mode that caused the original bug: an LLM (or any
    caller) inventing a literal, non-canonical status value must be rejected
    outright instead of silently executing a zero-row filter."""
    try:
        query_schema.validate_query_spec({
            "filters": [{"column": "status", "op": "==", "value": "نشط"}],
        })
    except query_schema.QueryValidationError:
        pass
    else:
        raise AssertionError("a non-canonical status literal must be rejected")


def test_query_schema_accepts_canonical_status_literal():
    cleaned = query_schema.validate_query_spec({
        "filters": [{"column": "status", "op": "==", "value": "Ongoing"}],
    })
    assert cleaned["filters"] == [{"column": "status", "op": "==", "value": "Ongoing"}]


# ── Zero-match wording must stay context-aware, not project-name-shaped ────

def test_zero_match_filter_response_does_not_ask_for_project_name():
    spec = {
        "filters": [
            {"column": "status", "op": "==", "value": "On-hold"},
            {"column": "dept", "op": "==", "value": "BPO - Health"},
        ],
    }
    payload = {"intent": PORTFOLIO_FILTER, "spec": spec, "rows": [], "aggregation": None}
    answer = response_formatter.format_answer("المشاريع الصحية المتوقفة", payload)
    assert "اسم المشروع" not in answer
    assert "لا توجد مشاريع" in answer
    assert "متوقف" in answer
    assert "الصحة" in answer


# ── Project-name lookups: still data-driven, never guessed on ambiguity ────

_HAJJ_CARD_PROJECTS = [
    {"project_code": "PJ-IT-HAJJSI", "project_name_ar": "بطاقة الحج الذكية 1442",
     "project_name_en": "HAJ_Smart Hajj ID 1442"},
    {"project_code": "PJ-IT-HAJJ43", "project_name_ar": "بطاقة الحج الذكية 1443",
     "project_name_en": "WBHAJ_Smart Hajj ID 1443"},
]

_MAKKAH_ROAD_PROJECTS = [
    {"project_code": "PM-GP-MKKH", "project_name_ar": "طريق مكة 2018",
     "project_name_en": "HAJ_Makkah Road 2018-Main"},
    {"project_code": "PJ-GP-MKKH22", "project_name_ar": "طريق مكة 2022",
     "project_name_en": "HAJ_Makkah Road 2022-Main"},
    {"project_code": "PJ-GP-MKKH23", "project_name_ar": "طريق مكة 2023",
     "project_name_en": "HAJ-Makkah Road 2023-Main"},
]

_BAHITH_PROJECTS = [
    {"project_code": "PJ-GP-HRSDSR", "project_name_ar": "الباحث الاجتماعي الثاني",
     "project_name_en": "HRSD-Social Case Surveying-Wave 2-Main"},
    {"project_code": "PRJ-OTHER", "project_name_ar": "برنامج التطوير الأول",
     "project_name_en": "Development Program I"},
]


def test_ambiguous_short_project_name_is_not_guessed_hajj_card():
    res = resolve_project("بطاقة الحج", _HAJJ_CARD_PROJECTS)
    assert res.status == "ambiguous"
    assert {c.project_code for c in res.candidates} == {"PJ-IT-HAJJSI", "PJ-IT-HAJJ43"}


def test_ambiguous_short_project_name_is_not_guessed_makkah_road():
    res = resolve_project("طريق مكة", _MAKKAH_ROAD_PROJECTS)
    assert res.status == "ambiguous"
    assert len(res.candidates) == 3


def test_unique_short_project_name_resolves_baheth():
    res = resolve_project("الباحث", _BAHITH_PROJECTS)
    assert res.status == "matched"
    assert res.canonical_project_code == "PJ-GP-HRSDSR"


def test_bug_query_phrase_no_longer_falls_through_to_project_resolver():
    """Before the fix, the routing gap meant the compound status+department
    query fell all the way through to project-name resolution and returned
    no_match. It must never reach the resolver at all now (is_explicit_
    portfolio_filter short-circuits ai_engine before that point) -- this
    pins the entity-extraction side too: even if it did reach the resolver,
    the phrase is now stripped of "عطني"/"كل"/plural "المشاريع" noise.
    """
    from modules.project_entity_resolver import extract_project_phrase
    phrase = extract_project_phrase("عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة")
    assert "عطني" not in phrase.split()
    assert "كل" not in phrase.split()
    assert "المشاريع" not in phrase.split()
