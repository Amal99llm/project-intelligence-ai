"""Synthetic-only tests for the clean trusted-tool chat engine."""
from datetime import date, timedelta
import config
from modules.database import BacklogProject, get_session
from modules.chat_service import answer
from modules import session_context
from modules.project_tools import search_projects, get_project_fields, filter_projects, aggregate_portfolio, compare_projects
from modules import chat_service
from modules.response_guard import validate
from types import SimpleNamespace

def _add(**values):
    defaults = dict(project_code="SYN-1", project_name_ar="الباحث الاجتماعي الثاني", project_name_en="Social Researcher II",
                    status="جاري", dept="التحول الرقمي", backlog=642_690_000, revenue_current=10_000,
                    total_revenue=20_000, total_contract_value=700_000_000, profit_pct=45.4,
                    project_manager="Synthetic Manager", end_date=date(2026, 12, 31))
    defaults.update(values)
    with get_session() as db: db.add(BacklogProject(**defaults)); db.commit()

def test_trusted_tools_and_null_are_canonical(seeded_db):
    _add(project_code="SYN-N", project_name_ar="مشروع تجريبي", backlog=None, project_manager=None)
    assert search_projects("مشروع تجريبي")[0]["project_id"] == "SYN-N"
    assert get_project_fields("SYN-N", ["project_manager"])["fields"]["project_manager"] is None
    assert get_project_fields("SYN-N", ["backlog"])["fields"]["backlog"] == 0
    assert filter_projects({"department": "التحول"})
    assert aggregate_portfolio("backlog", "sum")["value"] == 650_000
    assert len(compare_projects(["PRJ-001", "SYN-N"], ["backlog"])) == 2

def test_metric_override_correction_and_followup(seeded_db):
    _add(project_name_ar="الباحث التجريبي الثاني"); sid = "v2-followup"; today = date(2026, 7, 20)
    first = answer("متى ينتهي عقد الباحث التجريبي الثاني", sid, today)["answer"]
    assert "31/12/2026" in first
    backlog = answer("كم باقي إيراد في المشروع؟", sid, today)["answer"]
    assert "642,690,000" in backlog and "31/12/2026" not in backlog
    corrected = answer("مو النهاية، الإيرادات المتبقية", sid, today)["answer"]
    assert "تصحيحًا" in corrected and "642,690,000" in corrected
    manager = answer("ومديره؟", sid, today)["answer"]
    assert "Synthetic Manager" in manager

def test_ambiguity_number_selection_explicit_project_override(seeded_db):
    _add(project_code="SYN-R1", project_name_ar="تشغيل الرقابة على القطاع العقاري")
    _add(project_code="SYN-R2", project_name_ar="رقمنة أرشيف الثروة العقارية")
    _add(project_code="SYN-LAB", project_name_ar="مراكز الاختبارات الرقمية بـ8 مدن", end_date=date(2027, 1, 15))
    sid = "v2-ambiguity"
    choices = answer("والعقار", sid)["answer"]
    assert "1." in choices and "2." in choices
    selected = answer("1", sid)["answer"]
    assert "تم اختيار" in selected
    switched = answer("كم باقي وينتهي مراكز الاختبارات الرقمية بـ8 مدن؟", sid, date(2026, 7, 20))["answer"]
    assert "مراكز الاختبارات" in switched and "15/01/2027" in switched
    assert session_context.get_context(sid)["active_project_id"] == "SYN-LAB"

def test_no_hallucinated_project_or_number(seeded_db):
    response = answer("كم إيراد مشروع غير موجود إطلاقا", "v2-missing")["answer"]
    assert "حدد لي المشروع" in response
    assert not any(ch.isdigit() for ch in response)

def test_return_to_portfolio_clears_active_project_but_remembers_it(seeded_db):
    # Regression test for the portfolio-contamination bug: leaving a project
    # must NOT leave active_project_id set, or every following portfolio
    # question (counts, filters, KPIs) silently gets scoped to it.
    # last_active_project_id is the mechanism for "ارجع للمشروع" recall,
    # not active_project_id.
    _add(project_name_ar="الباحث التجريبي الثاني"); sid = "v2-scope"
    answer("الباحث التجريبي الثاني", sid)
    assert "للمحفظة" in answer("ارجع للمحفظة", sid)["answer"]
    ctx = session_context.get_context(sid)
    assert ctx["active_project_id"] is None
    assert ctx["active_scope"] == "portfolio"
    assert ctx["last_active_project_id"] == "SYN-1"


def test_portfolio_count_after_leaving_active_project_is_not_scoped_to_it(seeded_db):
    _add(project_code="SYN-1", project_name_ar="الباحث التجريبي الثاني", status="Ongoing")
    _add(project_code="SYN-OTHER", project_name_ar="مشروع تجريبي آخر", status="Ongoing")
    sid = "v2-portfolio-count"
    answer("الباحث التجريبي الثاني", sid)
    answer("ارجع للمحفظة", sid)
    result = aggregate_portfolio("status", "count", filters={"status": "Ongoing"})
    assert result["value"] == 2


def test_status_only_filter_with_arabic_alias(seeded_db):
    _add(project_code="SYN-ON", project_name_ar="مشروع نشط تجريبي", status="Ongoing")
    _add(project_code="SYN-OFF", project_name_ar="مشروع منتهي تجريبي", status="Completed")
    matches = filter_projects({"status": "نشطة"})
    ids = {row["project_id"] for row in matches}
    assert "SYN-ON" in ids and "SYN-OFF" not in ids


def test_status_and_department_filter_with_arabic_aliases(seeded_db):
    _add(project_code="SYN-DEP1", project_name_ar="مشروع متخصص تجريبي",
         status="Ongoing", dept="BPO-Specialized Pr")
    _add(project_code="SYN-DEP2", project_name_ar="مشروع غير متخصص تجريبي",
         status="Ongoing", dept="BPO - Inspection")
    matches = filter_projects({"status": "جاري", "department": "المشاريع المتخصصة"})
    ids = {row["project_id"] for row in matches}
    assert "SYN-DEP1" in ids and "SYN-DEP2" not in ids


def test_composite_situation_question_returns_more_than_status(seeded_db, monkeypatch):
    _add(project_name_ar="مشروع الوضع التجريبي", status="Ongoing", profit_pct=18.0, risk=0)
    sid = "v2-situation"
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "")  # exercise the deterministic composite path
    answer("مشروع الوضع التجريبي", sid)
    response = answer("وش وضعه؟", sid)["answer"]
    assert "642,690" not in response  # sanity: not the raw un-formatted number
    assert "حالة المشروع" in response
    assert "قيمة الأعمال المتبقية" in response
    assert "هامش الربح" in response
    assert "status" not in response and "profit_pct" not in response


def test_no_internal_field_names_in_date_answer(seeded_db):
    _add(project_name_ar="مشروع التاريخ التجريبي", end_date=date(2027, 1, 7))
    sid = "v2-date-leak"
    answer("مشروع التاريخ التجريبي", sid)
    response = answer("متى ينتهي عقده", sid)["answer"]
    assert "end_date" not in response
    assert "تاريخ الانتهاء" not in response  # internal label; a natural sentence is expected instead
    assert "07/01/2027" in response


def test_million_currency_formatting(seeded_db):
    _add(project_name_ar="مشروع التنسيق التجريبي", backlog=55_282_379.76)
    sid = "v2-format"
    answer("مشروع التنسيق التجريبي", sid)
    response = answer("كم باقي إيراد في المشروع؟", sid)["answer"]
    assert "55.28 مليون ريال" in response
    assert "55282379" not in response.replace(",", "")


def test_correction_word_does_not_false_trigger_inside_project_name(seeded_db):
    # "مش" (a CORRECTIONS token) is a literal substring of "المشروع" itself;
    # this must not spuriously prefix every answer with "تصحيحًا:".
    _add(project_name_ar="مشروع تجريبي عادي")
    sid = "v2-no-false-correction"
    answer("مشروع تجريبي عادي", sid)
    response = answer("كم باقي إيراد في المشروع؟", sid)["answer"]
    assert "تصحيحًا" not in response


def _tool_response(name, arguments):
    function = SimpleNamespace(name=name, arguments=__import__("json").dumps(arguments, ensure_ascii=False))
    call = SimpleNamespace(function=function)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))])


def _text_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=text))])


class _SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses); self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception): raise item
        return item


def test_azure_selects_exactly_one_tool_and_composes_verified_answer(seeded_db, monkeypatch):
    _add(project_name_ar="مشروع اللهجة التجريبي")
    sid = "azure-tool"
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "synthetic-key")
    # Establish the active project without involving Azure.
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "")
    answer("مشروع اللهجة التجريبي", sid)
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "synthetic-key")
    fake = _SequenceClient([
        _tool_response("get_project_fields", {"project_identifier": "SYN-1", "canonical_fields": ["backlog"]}),
        _text_response("باقي الأعمال في مشروع «مشروع اللهجة التجريبي» 642,690,000 ريال."),
    ])
    monkeypatch.setattr(chat_service, "_client", fake)
    response = answer("كم باقي فلوس بالمشروع؟", sid)["answer"]
    assert "642,690,000 ريال" in response
    selection = fake.requests[0]
    assert selection["tool_choice"] == "required"
    assert selection["parallel_tool_calls"] is False
    assert {tool["function"]["name"] for tool in selection["tools"]} == {
        "search_projects", "get_project_fields", "filter_projects", "aggregate_portfolio",
        "compare_projects", "get_contract_context"}
    composer_payload = fake.requests[1]["messages"][1]["content"]
    assert "642690000" in composer_payload and "previous_metric" in composer_payload


def test_azure_timeout_and_invalid_call_use_deterministic_fallback(seeded_db, monkeypatch):
    _add(project_name_ar="مشروع المهلة التجريبي")
    sid = "azure-timeout"
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "")
    answer("مشروع المهلة التجريبي", sid)
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "synthetic-key")
    timeout_client = _SequenceClient([TimeoutError("synthetic timeout")])
    monkeypatch.setattr(chat_service, "_client", timeout_client)
    assert "642,690,000 ريال" in answer("وش باقي من شغل العقد؟", sid)["answer"]
    invalid_client = _SequenceClient([_tool_response("get_project_fields", {
        "project_identifier": "MADE-UP", "canonical_fields": ["backlog"]})])
    monkeypatch.setattr(chat_service, "_client", invalid_client)
    assert "642,690,000 ريال" in answer("مو الموعد، قصدي الباقي المالي", sid)["answer"]


def test_guard_rejects_new_numbers_units_internal_names_and_old_metric():
    result = {"project_name": "مشروع آمن", "fields": {"backlog": 5000}}
    assert validate("باقي الأعمال في مشروع آمن 5,000 ريال.", result, ["backlog"])
    assert not validate("باقي الأعمال في مشروع آمن 6,000 ريال.", result, ["backlog"])
    assert not validate("باقي الأعمال في مشروع آمن 5,000 دولار.", result, ["backlog"])
    assert not validate("backlog في مشروع آمن 5,000 ريال.", result, ["backlog"])
    assert not validate("ينتهي مشروع آمن بقيمة 5,000 ريال.", result, ["backlog"])


def test_unseen_saudi_wording_and_project_navigation_fallback(seeded_db, monkeypatch):
    _add(project_code="SYN-REAL", project_name_ar="المشروع العقاري التجريبي", end_date=date(2026, 12, 31))
    _add(project_code="SYN-TEST", project_name_ar="مشروع الاختبارات الرقمية", end_date=date(2027, 1, 15))
    monkeypatch.setattr(config, "AZURE_OPENAI_KEY", "")
    sid = "unseen-wording"
    answer("المشروع العقاري التجريبي", sid)
    assert "642,690,000" in answer("كم باقي فلوس بالمشروع؟", sid)["answer"]
    assert "642,690,000" in answer("وش باقي من شغل العقد؟", sid)["answer"]
    assert "642,690,000" in answer("مو الموعد، قصدي الباقي المالي", sid)["answer"]
    assert "31/12/2026" in answer("ذا المشروع متى يخلص؟", sid)["answer"]
    revenue_cost = answer("كم جاب وكم كلف؟", sid)["answer"]
    assert "10,000 ريال" in revenue_cost and "0 ريال" in revenue_cost
    assert "الاختبارات" in answer("حولني لمشروع الاختبارات", sid)["answer"]
    assert "العقاري" in answer("ارجع للعقاري", sid)["answer"]
    assert "Synthetic Manager" in answer("مين ماسكه؟", sid)["answer"]
    summary = answer("وش وضعه باختصار؟", sid)["answer"]
    assert "حالة المشروع" in summary and "project_name_ar" not in summary