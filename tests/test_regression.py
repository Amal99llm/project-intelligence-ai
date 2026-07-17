"""
Regression test suite — Phase 1 & 2 implementation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from modules.followup_gate import check as gate_check
from modules.session_context import get_context, update_context, _reset_all_for_tests
from modules.semantic_dictionary import normalize_text


@pytest.fixture(autouse=True)
def reset_sessions():
    _reset_all_for_tests()
    yield
    _reset_all_for_tests()


def _ctx_project(code="HRSD-001", name="الباحث الاجتماعي الثاني", depth=1, answered=None):
    return {
        "active_project_code": code, "last_project_code": code,
        "active_project_display_name": name, "last_project_display_name": name,
        "active_project_depth": depth, "conversation_phase": "opening",
        "last_answered_fields": answered or [], "last_result_scope": "project",
        "last_result_type": "project_summary", "pending_project_confirmation": None,
        "last_list_project_codes": None, "last_project_list": [],
    }


def _ctx_empty():
    return {
        "active_project_code": None, "last_project_code": None,
        "active_project_depth": 0, "conversation_phase": "opening",
        "last_answered_fields": [], "last_result_scope": None,
        "last_result_type": None, "pending_project_confirmation": None,
        "last_list_project_codes": None, "last_project_list": [],
    }


def _ctx_list(codes=None, intent="losing_projects"):
    return {
        "active_project_code": None, "last_project_code": None,
        "last_result_scope": "list", "last_result_type": "portfolio_filter",
        "last_list_project_codes": codes or ["P1","P2","P3"],
        "last_project_list": codes or ["P1","P2","P3"],
        "last_list_intent": intent, "active_project_depth": 0,
        "conversation_phase": "opening", "last_answered_fields": [],
        "pending_project_confirmation": None,
    }


def _ctx_comparison(codes=None):
    first = (codes or ["P1","P2"])[0]
    return {
        "active_project_code": first, "last_project_code": first,
        "last_result_scope": "comparison", "last_result_type": "comparison",
        "last_comparison": {"codes": codes or ["P1","P2"], "field": None},
        "active_project_depth": 1, "conversation_phase": "opening",
        "last_answered_fields": [], "pending_project_confirmation": None,
        "last_list_project_codes": None, "last_project_list": [],
    }


# ════ GROUP 1: Gate fires correctly ════════════════════════════════════

class TestGateFires:

    def test_waw_takalif(self):
        g = gate_check("وتكاليفه؟", _ctx_project())
        assert g.fires
        assert g.field == "total_cost"
        assert g.confidence >= 0.9

    def test_waw_ribh(self):
        g = gate_check("وربحه؟", _ctx_project())
        assert g.fires
        # pl IS the profit P&L field — both pl and net_profit are correct
        assert g.field in ("net_profit", "pl", "profit_pct")

    def test_waw_eiradat(self):
        g = gate_check("وايراداته؟", _ctx_project())
        assert g.fires
        assert g.field == "total_revenue"

    def test_waw_meta_yantahi(self):
        g = gate_check("ومتى ينتهي؟", _ctx_project())
        assert g.fires

    def test_hal_yahtaj_mutabaa(self):
        g = gate_check("هل يحتاج متابعة؟", _ctx_project())
        assert g.fires
        assert g.intent == "assessment"

    def test_wash_wadauh(self):
        g = gate_check("وش وضعه؟", _ctx_project())
        assert g.fires
        assert g.intent == "assessment"

    def test_takalif_no_waw(self):
        g = gate_check("تكاليفه", _ctx_project())
        assert g.fires
        assert g.field == "total_cost"

    def test_hamshuh(self):
        g = gate_check("وهامشه؟", _ctx_project())
        assert g.fires

    def test_halatuh(self):
        g = gate_check("حالته", _ctx_project())
        assert g.fires

    def test_yasthil_intibah(self):
        g = gate_check("يستاهل انتباه", _ctx_project())
        assert g.fires
        assert g.intent == "assessment"


# ════ GROUP 2: Gate must NOT fire ══════════════════════════════════════

class TestGateNoFire:

    def test_no_project_ctx(self):
        g = gate_check("وتكاليفه؟", _ctx_empty())
        assert not g.fires

    def test_ijmali_revenues(self):
        g = gate_check("كم إجمالي الإيرادات؟", _ctx_project())
        assert not g.fires

    def test_li_jamee_almasharee(self):
        g = gate_check("الايرادات لجميع المشاريع", _ctx_project())
        assert not g.fires

    def test_comparison_no_fire(self):
        g = gate_check("قارن بين مشروعين", _ctx_project())
        assert not g.fires

    def test_new_project_long_name(self):
        g = gate_check("اعطني ملخص مشروع الباحث الاجتماعي الثاني", _ctx_project())
        assert not g.fires

    def test_losing_projects_no_fire(self):
        g = gate_check("أي المشاريع خسرانة؟", _ctx_project())
        assert not g.fires

    def test_list_leysh_no_fire(self):
        # After list, ليش is a list followup, not project followup
        g = gate_check("ليش؟", _ctx_list())
        assert not g.fires


# ════ GROUP 3: Session Context v2 ══════════════════════════════════════

class TestSessionContextV2:

    def test_v2_fields_exist(self):
        ctx = get_context("new-session")
        for field in ["active_project_code","active_project_depth","last_answered_fields",
                      "conversation_phase","last_result_type","last_project_list",
                      "last_comparison","last_executive_result","last_requested_metric"]:
            assert field in ctx, f"missing field: {field}"

    def test_active_syncs_legacy(self):
        update_context("s1", active_project_code="X", active_project_display_name="XN")
        ctx = get_context("s1")
        assert ctx["last_project_code"] == "X"
        assert ctx["last_project_display_name"] == "XN"

    def test_legacy_syncs_active(self):
        update_context("s2", last_project_code="Y", last_project_display_name="YN")
        ctx = get_context("s2")
        assert ctx["active_project_code"] == "Y"

    def test_answered_fields_accumulate(self):
        update_context("s3", last_requested_metric="total_cost")
        update_context("s3", last_requested_metric="net_profit")
        ctx = get_context("s3")
        assert "total_cost" in ctx["last_answered_fields"]
        assert "net_profit" in ctx["last_answered_fields"]

    def test_no_duplicate_fields(self):
        update_context("s4", last_requested_metric="total_cost")
        update_context("s4", last_requested_metric="total_cost")
        ctx = get_context("s4")
        assert ctx["last_answered_fields"].count("total_cost") == 1

    def test_new_project_resets_state(self):
        update_context("s5", active_project_code="OLD", last_result_type="project_kpi",
                       last_requested_metric="total_cost")
        update_context("s5", active_project_code="NEW", last_result_type="project_summary")
        ctx = get_context("s5")
        assert ctx["active_project_depth"] == 1
        assert ctx["last_answered_fields"] == []

    def test_new_project_does_not_carry_answered_fields(self):
        update_context("s5b", active_project_code="OLD", last_result_type="project_kpi",
                       last_requested_metric="total_cost")
        update_context("s5b", active_project_code="NEW", last_result_type="project_kpi",
                       last_requested_metric="net_profit")
        assert get_context("s5b")["last_answered_fields"] == ["net_profit"]

    def test_context_snapshot_cannot_mutate_shared_lists(self):
        update_context("s5c", last_project_list=["P1"])
        snapshot = get_context("s5c")
        snapshot["last_project_list"].append("P2")
        assert get_context("s5c")["last_project_list"] == ["P1"]

    def test_depth_increments(self):
        update_context("s6", active_project_code="P", last_result_type="project_kpi")
        update_context("s6", active_project_code="P", last_result_type="project_kpi")
        ctx = get_context("s6")
        assert ctx["active_project_depth"] >= 2

    def test_assessment_advances_phase(self):
        update_context("s7", active_project_code="P", last_result_type="assessment")
        ctx = get_context("s7")
        assert ctx["conversation_phase"] == "assessment"

    def test_comparison_stored(self):
        update_context("s8", last_result_type="comparison",
                       last_comparison={"codes":["P1","P2"],"field":None})
        ctx = get_context("s8")
        assert ctx["last_comparison"]["codes"] == ["P1","P2"]


# ════ GROUP 4: Portfolio switch followup ═══════════════════════════════

class TestPortfolioSwitchFollowup:

    def test_portfolio_q_no_gate(self):
        ctx = {**_ctx_project(), "last_result_scope":"portfolio", "last_result_type":"portfolio_kpi"}
        g = gate_check("كم إجمالي الإيرادات؟", ctx)
        assert not g.fires

    def test_possessive_re_engages_after_portfolio(self):
        ctx = {**_ctx_project(depth=3), "last_result_scope":"portfolio",
               "last_answered_fields":["total_revenue"]}
        g = gate_check("وتكاليفه؟", ctx)
        assert g.fires
        assert g.field == "total_cost"


# ════ GROUP 5: Followup after comparison ════════════════════════════════

class TestFollowupAfterComparison:

    def test_possessive_after_comparison(self):
        g = gate_check("وتكاليفه؟", _ctx_comparison())
        assert g.fires
        assert g.field == "total_cost"

    def test_assessment_after_comparison(self):
        g = gate_check("هل يحتاج متابعة؟", _ctx_comparison())
        assert g.fires
        assert g.intent == "assessment"

    def test_new_project_after_comparison_no_fire(self):
        g = gate_check("اعطني ملخص مشروع الباحث الاجتماعي الثاني", _ctx_comparison())
        assert not g.fires


# ════ GROUP 6: Parametrized field inference ═════════════════════════════

@pytest.mark.parametrize("query,expected_field", [
    ("وتكاليفه؟",  "total_cost"),
    ("تكاليفه",    "total_cost"),
    ("وتكلفته",    "total_cost"),
    ("وربحه",      "pl"),       # pl is the P&L profit field
    ("ربحه",       "pl"),       # pl is the P&L profit field
    ("وايراداته",  "total_revenue"),
    ("ايراداته",   "total_revenue"),
    ("وهامشه",     "profit_pct"),
    ("ومتى ينتهي", "effective_end_date"),
    ("وتاريخه",    "effective_end_date"),
    ("وتقدمه",     "progress_completed"),
    ("تقدمه",      "progress_completed"),
])
def test_field_inference_parametrized(query, expected_field):
    g = gate_check(query, _ctx_project())
    assert g.fires, f"gate must fire for {query!r}"
    assert g.field == expected_field, f"{query!r} → expected {expected_field}, got {g.field}"


# ════ GROUP 7: Composer template fallbacks ═══════════════════════════════

class TestComposerFallbacks:

    def _project(self, **kw):
        base = {
            "project_code":"T-001","project_name_ar":"مشروع الاختبار",
            "project_name_en":"Test Project","status":"Ongoing",
            "progress_completed":42.5,"project_manager":"Ahmed",
            "total_contract_value":1_120_000_000,"total_revenue":475_010_000,
            "total_cost":259_140_000,"net_profit":215_860_000,
            "profit_pct":45.4,"backlog":642_700_000,
            "effective_end_date":"2029-04-17","days_remaining":1370,"risk":40_500_000,
        }
        base.update(kw)
        return base

    def test_summary_not_empty(self):
        """Test template fallback directly (mocking openai dependency)"""
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), None, True, 1)
        assert len(text) > 20
        assert "مشروع الاختبار" in text

    def test_field_response_not_empty(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), "total_cost", True, 2)
        assert len(text) > 5

    def test_assessment_with_facts(self):
        import unittest.mock, sys
        with unittest.mock.patch.dict(sys.modules, {
            'openai': unittest.mock.MagicMock(), 'config': unittest.mock.MagicMock()
        }):
            from modules.response_composer import _template_assessment
            text = _template_assessment("مشروع X", ["يسجل خسارة صافية","تجاوز تاريخ انتهائه"], True)
            assert "مشروع X" in text

    def test_assessment_no_facts(self):
        import unittest.mock, sys
        with unittest.mock.patch.dict(sys.modules, {
            'openai': unittest.mock.MagicMock(), 'config': unittest.mock.MagicMock()
        }):
            from modules.response_composer import _template_assessment
            text = _template_assessment("مشروع X", [], True)
            assert "مشروع X" in text
            assert len(text) > 15


# ════ GROUP 8: Normalization regression ═════════════════════════════════

@pytest.mark.parametrize("raw,expected", [
    ("وتكاليفه",  "وتكاليفه"),
    ("وَتَكَالِيفُه", "وتكاليفه"),
    ("وَرِبْحُه",  "وربحه"),
])
def test_normalization_arabic(raw, expected):
    assert normalize_text(raw) == expected, f"{raw!r} → {normalize_text(raw)!r} != {expected!r}"


# ═══════════════════════════════════════════════════════════════════════
# GROUP 10 — Response Composer: no internal field names in fallback
# ═══════════════════════════════════════════════════════════════════════

def _mock_openai_modules():
    import unittest.mock, sys
    return unittest.mock.patch.dict(sys.modules, {
        'openai': unittest.mock.MagicMock(),
        'config': unittest.mock.MagicMock(
            AZURE_OPENAI_KEY="", AZURE_OPENAI_ENDPOINT="",
            AZURE_OPENAI_API_VERSION="", AZURE_OPENAI_DEPLOYMENT="gpt-5-mini",
        ),
    })


class TestComposerNoFieldNames:
    """Template fallback must never expose internal field names."""

    def _project(self):
        return {
            "project_code": "T-001", "project_name_ar": "مشروع الاختبار",
            "status": "Ongoing", "progress_completed": 42.5,
            "project_manager": "بدر محمد آل شهراني",
            "total_contract_value": 1_120_000_000, "total_revenue": 475_010_000,
            "total_cost": 259_140_000, "net_profit": 215_860_000,
            "profit_pct": 45.44, "backlog": 642_690_000,
            "effective_end_date": "2029-04-17", "days_remaining": 1006, "risk": 40_500_000,
        }

    def test_no_field_name_in_cost_fallback(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), "total_cost", True, 2)
        assert "total_cost" not in text
        assert "259" in text or "مليون" in text

    def test_no_field_name_in_profit_fallback(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), "net_profit", True, 2)
        assert "net_profit" not in text
        assert "215" in text or "مليون" in text

    def test_no_field_name_in_margin_fallback(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), "profit_pct", True, 1)
        assert "profit_pct" not in text
        assert "45" in text
        assert "%" in text

    def test_percentage_format_correct(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), "profit_pct", True, 1)
        # Should say 45.4% not "45 ريال" or "خمسة وأربعون"
        assert "%" in text
        assert "ريال" not in text

    def test_summary_no_internal_names(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_fallback
            text = _template_fallback(self._project(), None, True, 1)
        for field in ["total_cost","net_profit","profit_pct","total_revenue","backlog"]:
            assert field not in text, f"Field name '{field}' leaked into response"

    @pytest.mark.skip(reason="requires full environment with openai+config installed")
    def test_followup_depth_concise(self):
        from modules.response_composer import _template_fallback
        # depth > 1 → very short answer
        text = _template_fallback(self._project(), "total_cost", True, 3)
        assert len(text) < 60  # just the value, not a full sentence


# ═══════════════════════════════════════════════════════════════════════
# GROUP 11 — Verified brief builder: Arabic labels only
# ═══════════════════════════════════════════════════════════════════════

class TestVerifiedBrief:

    def _project(self):
        return {
            "project_code": "T-001", "project_name_ar": "مشروع الاختبار",
            "status": "Ongoing", "progress_completed": 42.5,
            "total_cost": 259_140_000, "net_profit": 215_860_000,
            "profit_pct": 45.44,
        }

    def test_brief_uses_arabic_labels(self):
        with _mock_openai_modules():
            from modules.response_composer import _build_brief
            brief = _build_brief(self._project(), None)
        assert "total_cost" not in brief
        assert "net_profit" not in brief
        assert "profit_pct" not in brief
        assert "التكاليف" in brief
        assert "صافي الربح" in brief
        assert "هامش الربح" in brief

    def test_brief_formats_money(self):
        with _mock_openai_modules():
            from modules.response_composer import _build_brief
            brief = _build_brief(self._project(), None)
        assert "259" in brief
        assert "مليون" in brief

    def test_brief_formats_percentage(self):
        with _mock_openai_modules():
            from modules.response_composer import _build_brief
            brief = _build_brief(self._project(), None)
        assert "45.4%" in brief

    def test_brief_single_field_only_that_field(self):
        with _mock_openai_modules():
            from modules.response_composer import _build_brief
            brief = _build_brief(self._project(), "profit_pct")
        assert "هامش الربح" in brief
        assert "التكاليف" not in brief
        assert "profit_pct" not in brief

    def test_brief_percentage_not_money(self):
        with _mock_openai_modules():
            from modules.response_composer import _build_brief
            brief = _build_brief(self._project(), "profit_pct")
        # Should show 45.4% not 45.44 ريال
        assert "%" in brief
        assert "ريال" not in brief.split("هامش الربح:")[1][:20]


# ═══════════════════════════════════════════════════════════════════════
# GROUP 12 — Assessment: recommendation vs opinion distinction
# ═══════════════════════════════════════════════════════════════════════

class TestAssessmentDistinction:

    def test_recommendation_triggers(self):
        """هل يحتاج متابعة should be detected as recommendation"""
        from modules.semantic_dictionary import normalize_text
        from modules.followup_gate import _ASSESSMENT_MARKERS
        q = "هل يحتاج متابعة؟"
        qn = normalize_text(q)
        assert any(t in qn for t in _ASSESSMENT_MARKERS)

    def test_opinion_triggers(self):
        """وش رأيك should be detected as assessment"""
        from modules.semantic_dictionary import normalize_text
        from modules.followup_gate import _ASSESSMENT_MARKERS
        q = "وش رأيك فيه؟"
        qn = normalize_text(q)
        assert any(t in qn for t in _ASSESSMENT_MARKERS)

    @pytest.mark.skip(reason="requires full environment with openai+config installed")
    def test_assessment_template_no_concerns(self):
        from modules.response_composer import _template_assessment
        # Profitable project → no concerns
        text = _template_assessment("مشروع X", ["ربح صافي: 215 مليون ريال", "هامش الربح: 45.4%"], True)
        assert "مشروع X" in text
        assert len(text) > 10

    def test_assessment_template_with_loss(self):
        with _mock_openai_modules():
            from modules.response_composer import _template_assessment
            text = _template_assessment("مشروع Y", ["خسارة صافية: -17 مليون ريال"], True)
        assert "مشروع Y" in text
        assert "متابعة" in text or "خسار" in text
