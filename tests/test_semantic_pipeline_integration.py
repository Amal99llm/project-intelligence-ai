"""Integration tests for the new semantic interpretation pipeline
(modules.semantic_interpreter -> modules.entity_resolvers ->
modules.query_compiler -> modules.query_executor -> modules.verification ->
modules.response_formatter/response_layer), and for modules.ai_engine's
SEMANTIC_INTERPRETER_ENABLED-gated integration point.

Two kinds of coverage:
  1. Full function-composition tests exercising the whole new layer stack
     against inline project dicts (no DB, LLM mocked) -- proves the layers
     actually fit together, not just each in isolation.
  2. Real, DB-backed tests through modules.ai_engine.answer() with the flag
     forced on via monkeypatch (config default stays off everywhere else),
     using the same seeded_db/today fixtures as the rest of the suite.
"""

import json
from datetime import date, timedelta
from unittest.mock import patch

import config
from modules import ai_engine, query_compiler, query_executor, response_formatter, session_context
from modules.database import BacklogProject, get_session
from modules.intent_schema import CONTROLLED_FALLBACK, GROUPED_ANALYTICS, PORTFOLIO_FILTER, PROJECT_SUMMARY, SMALL_TALK
from modules.semantic_interpreter import interpret


# ── Helpers ───────────────────────────────────────────────────────────────

def _fake_llm_client(raw: dict):
    payload = json.dumps(raw, ensure_ascii=False)

    class Function:
        pass
    fn = Function()
    fn.arguments = payload

    class Call:
        pass
    call = Call()
    call.function = fn

    class Message:
        pass
    message = Message()
    message.tool_calls = [call]

    class Choice:
        pass
    choice = Choice()
    choice.message = message

    class Completions:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"choices": [choice]})()

    return type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()


def _bug_report_raw() -> dict:
    return {
        "intent": PORTFOLIO_FILTER,
        "scope": "portfolio",
        "entities": {
            "projects": [], "departments": ["إدارة المشاريع المتخصصة"], "status": ["نشطة"],
            "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
        },
        "operations": {"filter": True, "group_by": None, "sort": None, "limit": None, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.96,
        "requires_clarification": False,
    }


_PROJECTS = [
    {"project_code": "P1", "project_name_ar": "مشروع تفتيش هيئة العلا", "project_name_en": "Inspection",
     "status": "Ongoing", "dept": "BPO-Specialized Pr"},
    {"project_code": "P2", "project_name_ar": "مشروع خارج القسم", "project_name_en": "Outside",
     "status": "Ongoing", "dept": "BPO - Inspection"},
    {"project_code": "P3", "project_name_ar": "مشروع متخصص متوقف", "project_name_en": "Paused",
     "status": "On-hold", "dept": "BPO-Specialized Pr"},
]


# ── 1. Function-composition: the whole new stack fits together ──────────

def test_full_stack_resolves_and_executes_the_reported_bug_query():
    """Composition test for interpret -> resolve -> compile -> execute ->
    format against inline projects. modules.verification always re-derives
    from a fresh DB read by design (see modules/verification.py), so it is
    exercised separately, against the real DB, in
    test_flag_on_department_filter_against_real_db_reproduces_bug_fix below."""
    client = _fake_llm_client(_bug_report_raw())
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        interpretation = interpret(
            "عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة", {}, projects=_PROJECTS,
        )
    assert interpretation.intent == PORTFOLIO_FILTER
    assert not interpretation.requires_clarification
    assert interpretation.entities.status == ["Ongoing"]
    assert interpretation.entities.departments == ["BPO-Specialized Pr"]

    spec = query_compiler.compile_query_spec(interpretation)
    result = query_executor.execute(spec, projects=_PROJECTS)
    assert [row["project_code"] for row in result["rows"]] == ["P1"]

    payload = {"intent": PORTFOLIO_FILTER, "spec": spec, **result}
    answer = response_formatter.format_answer(
        "عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة", payload,
    )
    assert "تفتيش هيئة العلا" in answer
    assert "اسم المشروع" not in answer


def test_full_stack_grouped_analytics():
    from modules import grouped_analytics
    raw = {
        "intent": GROUPED_ANALYTICS, "scope": "portfolio",
        "entities": {
            "projects": [], "departments": [], "status": [], "programs": [],
            "business_units": [], "segments": [], "managers": [], "metrics": ["إجمالي الإيرادات"], "dates": [],
        },
        "operations": {"filter": False, "group_by": "dept", "sort": "desc", "limit": 1, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.9, "requires_clarification": False,
    }
    projects = [
        {"project_code": "P1", "dept": "BPO - Inspection", "total_revenue": 100},
        {"project_code": "P2", "dept": "BPO-Specialized Pr", "total_revenue": 500},
    ]
    client = _fake_llm_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        interpretation = interpret("وش أعلى إدارة من ناحية الإيرادات؟", {}, projects=projects)
    entities = query_compiler.compile_grouped_entities(interpretation)
    result = grouped_analytics.execute_grouped(projects, entities)
    text = grouped_analytics.format_grouped(result)
    assert "BPO-Specialized Pr" in text


# ── 2. Real DB-backed tests through ai_engine.answer() ───────────────────

def _seed_extra_project(**overrides):
    defaults = dict(
        project_name_en="", project_name_ar="", bu="Digital", segment="Public",
        status="Ongoing", progress_completed=0.5,
        start_date=date(2026, 1, 1), end_date=date(2026, 12, 31),
        contract_value=0, amendment_crs=0, total_contract_value=0,
        previous_years_rev=0, revenue_current=0, other_income=0, total_revenue=0,
        previous_years_cost=0, cost_of_revenue=0, other_cost=0, total_cost=0,
        backlog=0, pl=0, planned_profit=0, planned_pm_pct=0, variance=0,
        pm_pct_up_to_2025=0, gp_2026=0, pm_pct_2026=0, risk=0,
        acc_rev=0, pb=0, adv=0, ar=0, contract_assets=0, ap=0, acc_exp=0,
        contract_liabilities=0, deferred_cost=0, open_po=0, ecl_ar=0,
        ecl_acc_rev=0, etc_cost=0, etc_revenue=0,
    )
    defaults.update(overrides)
    with get_session() as session:
        session.add(BacklogProject(**defaults))
        session.commit()


def test_flag_off_by_default_uses_legacy_path(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    assert config.SEMANTIC_INTERPRETER_ENABLED is False
    result = ai_engine.answer("المشاريع الجارية", session_id="v2-off-test")
    assert "answer" in result
    # Whatever the answer is, it must come from the untouched legacy path --
    # confirmed by the flag genuinely being off, not by inspecting internals.


def test_flag_on_status_filter_against_real_seeded_db(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    raw = {
        "intent": PORTFOLIO_FILTER, "scope": "portfolio",
        "entities": {
            "projects": [], "departments": [], "status": ["الجارية"], "programs": [],
            "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
        },
        "operations": {"filter": True, "group_by": None, "sort": None, "limit": None, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.95, "requires_clarification": False,
    }
    client = _fake_llm_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = ai_engine.answer("المشاريع الجارية", session_id="v2-status-test")
    assert result["query_type"] == PORTFOLIO_FILTER
    # 4 of the 5 seeded TEST_PROJECTS default to status="Ongoing" (PRJ-005
    # is the only Completed one) -- proves the DB was actually filtered.
    assert "4" in result["answer"] or "أربع" in result["answer"]


def test_flag_on_department_filter_against_real_db_reproduces_bug_fix(seeded_db, today, monkeypatch):
    """The original reported bug, run through the new path against a real
    (temp) database with an extra department-bearing project seeded in."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    _seed_extra_project(
        project_code="PRJ-DEPT-1", project_name_ar="مشروع تفتيش تجريبي",
        status="Ongoing", dept="BPO-Specialized Pr",
    )
    client = _fake_llm_client(_bug_report_raw())
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = ai_engine.answer(
            "عطني كل المشاريع النشطة في إدارة المشاريع المتخصصة", session_id="v2-dept-test",
        )
    assert result["query_type"] == PORTFOLIO_FILTER
    assert "تفتيش تجريبي" in result["answer"]
    assert "اسم المشروع" not in result["answer"]


def test_flag_on_grouped_analytics_translates_dept_label_never_leaks_raw_literal(seeded_db, today, monkeypatch):
    """modules.grouped_analytics.format_grouped echoes the raw winning
    bucket value verbatim -- for `dept` that's a truncated DB literal like
    "BPO-Specialized Pr", which must never reach the user untranslated.
    Regression coverage for the leak the response_layer guard caught
    during manual testing against the real model."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    _seed_extra_project(
        project_code="PRJ-GROUP-1", project_name_ar="مشروع تجريبي للتجميع",
        status="Ongoing", dept="BPO-Specialized Pr", total_revenue=900_000,
    )
    raw = {
        "intent": GROUPED_ANALYTICS, "scope": "portfolio",
        "entities": {
            "projects": [], "departments": [], "status": [], "programs": [],
            "business_units": [], "segments": [], "managers": [], "metrics": ["الإيرادات"], "dates": [],
        },
        "operations": {"filter": False, "group_by": "dept", "sort": "desc", "limit": 1, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.9, "requires_clarification": False,
    }
    client = _fake_llm_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = ai_engine.answer("وش أعلى إدارة من ناحية الإيرادات؟", session_id="v2-group-leak-test")
    assert result["query_type"] == GROUPED_ANALYTICS
    assert "BPO-Specialized Pr" not in result["answer"]
    assert "إدارة المشاريع المتخصصة" in result["answer"]


def test_unresolved_entity_on_a_v2_unhandled_intent_defers_never_clarifies(seeded_db, today, monkeypatch):
    """Regression: a query classified as an intent this path doesn't own
    (e.g. project_summary) must ALWAYS defer to the legacy path, even when
    the model also tagged an entity that fails to resolve. Before the fix,
    the clarification check ran before the intent-ownership check, so a
    project-name query the model mistagged (e.g. as a "manager" entity)
    got intercepted into a dead-end clarification instead of falling
    through to the legacy resolver -- which, as proven by
    test_flag_off_by_default_uses_legacy_path-style calls, resolves real
    project names like "الباحث الاجتماعي الثاني" correctly on its own."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    raw = {
        "intent": PROJECT_SUMMARY, "scope": "project",
        "entities": {
            "projects": [], "departments": [], "status": [], "programs": [],
            "business_units": [], "segments": [],
            "managers": ["الباحث الاجتماعي الثاني"],  # model mistagged the project name
            "metrics": [], "dates": [],
        },
        "operations": {"filter": False, "group_by": None, "sort": None, "limit": None, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.9, "requires_clarification": False,
    }
    client = _fake_llm_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = ai_engine.answer("الباحث الاجتماعي الثاني", session_id="v2-unresolved-defer-test")
    assert result["query_type"] != CONTROLLED_FALLBACK
    assert "ما قدرت أتأكد" not in result["answer"]
    # Check the deterministic, verified outcome (session state resolved to
    # the real project) rather than the free-form composed wording, which
    # is LLM-phrased and not guaranteed to restate the name verbatim.
    state = session_context.get_context("v2-unresolved-defer-test")
    assert state.get("last_selected_project_id") == "PRJ-001" or state.get("active_project_code") == "PRJ-001"


def test_pending_clarification_defers_to_legacy_path(seeded_db, today, monkeypatch):
    """A disambiguation reply in flight must never be intercepted by the
    new path -- it isn't rebuilt to understand pending state this session."""
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)
    session_context.update_context("v2-pending-test", pending_project_confirmation={
        "candidates": [
            {"project_code": "PRJ-001", "display_name": "الباحث الاجتماعي الثاني"},
            {"project_code": "PRJ-002", "display_name": "مشروع النور للطاقة"},
        ],
        "kind": "lookup", "original_query": "مشروع",
    })

    def _must_not_be_called(**kwargs):
        raise AssertionError("the semantic interpreter must not run while a clarification is pending")

    with patch("modules.semantic_interpreter._get_openai", side_effect=_must_not_be_called):
        result = ai_engine.answer("الأول", session_id="v2-pending-test")
    assert "answer" in result


def test_llm_failure_falls_back_to_legacy_path(seeded_db, today, monkeypatch):
    monkeypatch.setattr(ai_engine, "riyadh_today", lambda: today)
    monkeypatch.setattr(config, "SEMANTIC_INTERPRETER_ENABLED", True)

    def _raise(**kwargs):
        raise TimeoutError("simulated Azure OpenAI timeout")

    fake_client = type("Client", (), {
        "chat": type("Chat", (), {"completions": type("Completions", (), {"create": staticmethod(_raise)})()})()
    })()
    with patch("modules.semantic_interpreter._get_openai", return_value=fake_client):
        result = ai_engine.answer("المشاريع الجارية", session_id="v2-fallback-test")
    # Must still get a real answer from the legacy path, not an error/crash.
    assert "answer" in result
    assert result["answer"]
