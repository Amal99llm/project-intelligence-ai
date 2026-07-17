"""
modules/ai_engine.py — v12
Understanding-first + Follow-up Gate + Conversational Composer.

Entry pipeline per turn:
  0. Pending confirmation check       (ordinal/yes responses)
  1. Follow-up First Gate             (semantic, fires before understanding)
     → if fires: route to followup handler immediately
  2. understand()                     (structured meaning)
  3. Orchestrate                      (route to right handler)
  4. compose_project_response()       (natural language, context-aware)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from modules.database import log_query
from modules import session_context
from modules.intent_schema import (
    PROJECT_SUMMARY, PROJECT_FOLLOWUP, PROJECT_COMPARISON,
    PORTFOLIO_KPI, PROJECT_KPI,
    PORTFOLIO_FILTER, PORTFOLIO_RANKING, PORTFOLIO_SUMMARY,
    EXEC_ATTENTION, LIST_FOLLOWUP,
    CONTRACT_DOC, CONTRACT_VALUE,
    GROUPED_ANALYTICS, SMALL_TALK, OUT_OF_SCOPE,
)
from modules.understanding import understand, Understanding, SCOPE_PROJECT, SCOPE_PORTFOLIO
from modules.followup_gate import check as followup_gate_check
from modules.response_composer import compose_project_response, compose_assessment_response
from modules.kpi_calculator import calculate_executive_kpis, summarize_by_bu
from modules.project_repository import fetch_enriched_projects
from modules.project_entity_resolver import (
    normalize_project_text, resolve_project, format_resolution_prompt, extract_project_phrase,
)
from modules.kpi_responder import answer_kpi_question, answer_kpi_for_known_project
from modules.rag_engine import answer_contract_query, ContractQueryError
from modules.semantic_dictionary import (
    FIELDS, detect_requested_field, detect_semantic_intent,
    detect_requested_fields, detect_small_talk, is_methodology_question,
)
from modules.response_formatter import format_delay_status, format_project_metrics
from modules.contract_semantics import analyze_contract_request, render_contract_answer
from modules.knowledge_boundary import classify_boundary, boundary_answer
from modules.executive_analysis import format_attention_summary, project_attention
from modules import query_builder, query_executor, query_schema, response_formatter, verification

logger = logging.getLogger(__name__)

# ── Ordinals ─────────────────────────────────────────────────────────────────
_ORDINAL_NORM = {normalize_project_text(k): v for k, v in {
    "الاول":0,"الأول":0,"الاولى":0,"الأولى":0,"اول":0,
    "الثاني":1,"الثانيه":1,"الثانية":1,
    "الثالث":2,"الثالثه":2,"الثالثة":2,
    "الرابع":3,"الرابعه":3,"الرابعة":3,
    "الخامس":4,"الخامسه":4,"الخامسة":4,
    "first":0,"second":1,"third":2,"fourth":3,"fifth":4,
    "1":0,"2":1,"3":2,"4":3,"5":4,
}.items()}
_YES_NORM = {normalize_project_text(w) for w in (
    "نعم","ايوه","أيوه","إيه","ايه","يب","اه","آه","صح","صحيح","أكيد","اكيد","هو","هذا","yes","y"
)}
_LAST_NORM = {normalize_project_text(w) for w in ("الأخير", "الاخير", "آخر واحد", "last")}
_NEXT_NORM = {normalize_project_text(w) for w in ("اللي بعده", "التالي", "next")}


def _try_resolve_pending(query: str, ctx: dict):
    pending = ctx.get("pending_project_confirmation")
    if not pending or not pending.get("candidates"):
        return None
    candidates = pending["candidates"]
    normalized = normalize_project_text(query)
    tokens = set(normalized.split())
    if normalized in _LAST_NORM:
        idx = len(candidates) - 1
        chosen = dict(pending)
        chosen["selected_index"] = idx
        return candidates[idx]["project_code"], candidates[idx]["display_name"], chosen
    if normalized in _NEXT_NORM:
        idx = min(int(ctx.get("selected_disambiguation_index") or 0) + 1, len(candidates) - 1)
        chosen = dict(pending)
        chosen["selected_index"] = idx
        return candidates[idx]["project_code"], candidates[idx]["display_name"], chosen
    for word, idx in _ORDINAL_NORM.items():
        if (word in tokens or normalize_project_text("و" + word) in tokens) and 0 <= idx < len(candidates):
            chosen = dict(pending)
            chosen["selected_index"] = idx
            return candidates[idx]["project_code"], candidates[idx]["display_name"], chosen
    if len(candidates) == 1 and tokens & _YES_NORM:
        return candidates[0]["project_code"], candidates[0]["display_name"], pending
    named = [c for c in candidates if normalized in {
        normalize_project_text(c["display_name"]),
        normalize_project_text(c["project_code"]),
    }]
    if len(named) == 1:
        return named[0]["project_code"], named[0]["display_name"], pending
    return None


def _pending_state(resolution, kind, kpi_name=None, original_query=None, field_name=None):
    return {
        "candidates": [{"project_code": c.project_code, "display_name": c.display_name}
                       for c in resolution.candidates],
        "kind": kind, "kpi_name": kpi_name,
        "original_query": original_query, "field_name": field_name,
    }


_ROW_FIELDS = (
    "project_code","project_name_ar","project_name_en","status","bu",
    "project_manager","total_contract_value","total_revenue","total_cost",
    "backlog","pl","net_profit","profit_pct","progress_completed", "start_date",
    "effective_end_date","days_remaining","risk","variance",
)


def _trim(p): return {k: p.get(k) for k in _ROW_FIELDS}


# ── Core verified lookup ──────────────────────────────────────────────────────

def _lookup_verified(project_code: str, today: date, projects: list, query: str = "", topic: str | None = None):
    """Returns (row_dict | None, source_info | None)."""
    try:
        spec = query_schema.validate_query_spec({"project_code": project_code, "limit": 1})
        result = query_executor.execute(spec, today=today, projects=projects)
        verdict = verification.verify(result, today=today)
    except Exception:
        logger.exception("Lookup failed: %s", project_code)
        return None, None
    if not verdict.ok or not result["rows"]:
        return None, None
    src = response_formatter.build_source_attribution(spec, None, None)
    if topic and topic in FIELDS:
        src["columns"] = [FIELDS[topic].source_column]
    return result["rows"][0], src


def _format_project(row: dict, query: str, ctx: dict, field: str | None, intent: str) -> str:
    """Route to conversational composer."""
    return compose_project_response(row, query=query, ctx=ctx, field=field, intent=intent)


def _resolve_and_respond(query: str, today: date, projects: list, ctx: dict,
                          topic: str | None = None) -> tuple:
    """Resolve project name → verified lookup → compose response."""
    resolution = resolve_project(query, projects)
    if resolution.status in {"ambiguous", "confirmation"}:
        pending = _pending_state(resolution, "lookup", original_query=query)
        return format_resolution_prompt(resolution), None, {
            "pending_project_confirmation": pending,
            "last_disambiguation_options": list(pending["candidates"]),
            "last_disambiguation_query": query,
            "selected_disambiguation_index": None,
        }
    if resolution.status != "matched":
        residual = extract_project_phrase(query)
        if len(residual) < 3 and ctx.get("active_project_code"):
            # No real project mentioned → treat as followup to active project
            code = ctx["active_project_code"]
            row, src = _lookup_verified(code, today, projects, query, topic)
            if row:
                text = _format_project(row, query, ctx, topic, "general_followup")
                return text, src, {"active_project_code": code}
        return format_resolution_prompt(resolution), None, {}

    code = resolution.canonical_project_code
    name = resolution.candidates[0].display_name
    try:
        spec = query_schema.validate_query_spec({"project_code": code, "limit": 1})
        result = query_executor.execute(spec, today=today, projects=projects)
        verdict = verification.verify_project_lookup(result, resolution, today=today)
    except Exception:
        return verification.PROJECT_FALLBACK_MESSAGE, None, {}
    if not verdict.ok:
        return verification.PROJECT_FALLBACK_MESSAGE, None, {}
    row = result["rows"][0]
    src = response_formatter.build_source_attribution(spec, None, None)
    text = _format_project(row, query, ctx, topic, "field_lookup" if topic else "summary")
    return text, src, {
        "active_project_code": code,
        "active_project_display_name": name,
        "last_project_code": code,
        "last_project_display_name": name,
        "pending_project_confirmation": None,
        "last_result_type": "project_kpi" if topic else "project_summary",
        "last_requested_metric": topic,
    }


def _run_pipeline(intent_str: str, query: str, today: date, projects: list):
    try:
        spec = query_builder.build_query(query, today.isoformat())
        result = query_executor.execute(spec, today=today, projects=projects)
        verdict = verification.verify(result, today=today)
    except Exception as exc:
        logger.warning("Pipeline failed: %s | %.60s", exc, query)
        return verification.FALLBACK_MESSAGE, None
    if not verdict.ok:
        return verification.FALLBACK_MESSAGE, None
    payload = {
        "intent": intent_str, "spec": spec,
        "row_count": result["row_count"],
        "matched_row_count": result.get("matched_row_count", result["row_count"]),
        "rows": [_trim(p) for p in result["rows"][:30]],
        "aggregation": result["aggregation"],
        "aggregation_result": result["aggregation_result"],
    }
    try:
        text = response_formatter.format_answer(query, payload)
    except Exception:
        return verification.FALLBACK_MESSAGE, None
    kpi_formula = query_executor.formula_for_aggregation(result["aggregation"])
    src = response_formatter.build_source_attribution(spec, result["aggregation_result"], kpi_formula)
    src["result_context"] = {
        "project_codes": [r.get("project_code") for r in result["rows"] if r.get("project_code")],
        "intent": detect_semantic_intent(query) or intent_str,
        "filters": spec.get("filters", []),
        "sort": spec.get("sort"),
    }
    return text, src


def _list_ctx(src: dict | None) -> dict:
    ctx = (src or {}).get("result_context") or {}
    return {
        "last_result_scope": "list",
        "last_result_type": "portfolio_filter",
        "last_list_project_codes": list(ctx.get("project_codes") or []),
        "last_project_list": list(ctx.get("project_codes") or []),
        "last_list_intent": ctx.get("intent"),
        "last_list_filters": list(ctx.get("filters") or []),
        "last_list_sort": ctx.get("sort"),
    }


# ── Follow-up Gate handler ────────────────────────────────────────────────────

def _handle_followup_gate(gate, query: str, today: date, ctx: dict, projects: list) -> tuple:
    """Directly handle a follow-up without understanding overhead."""
    code = ctx.get("active_project_code") or ctx.get("last_project_code")
    field = gate.field
    intent = gate.intent

    if intent == "delay":
        row, src = _lookup_verified(code, today, projects, query)
        if not row:
            return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_FOLLOWUP, None, {}
        return format_delay_status(row), PROJECT_KPI, src, {
            "active_project_code": code, "last_project_code": code,
            "last_result_type": "project_kpi", "last_requested_metric": "days_remaining",
        }

    if intent == "assessment":
        row, src = _lookup_verified(code, today, projects, query)
        if not row:
            return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_FOLLOWUP, None, {}
        attn = project_attention(row)
        text = compose_assessment_response(row, query=query, ctx=ctx, attention_data=attn)
        upd = {
            "active_project_code": code,
            "last_project_code": code,
            "last_result_type": "assessment",
            "conversation_phase": "assessment",
            "last_requested_metric": "assessment",
        }
        return text, PROJECT_KPI, src, upd

    # field_lookup or general_followup
    row, src = _lookup_verified(code, today, projects, query, field)
    if not row:
        return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_FOLLOWUP, None, {}
    text = _format_project(row, query, ctx, field, intent)
    upd = {
        "active_project_code": code,
        "last_project_code": code,
        "last_result_type": "project_kpi" if field else "project_summary",
        "last_requested_metric": field,
    }
    return text, PROJECT_KPI if field else PROJECT_SUMMARY, src, upd


# ── List followup handler ─────────────────────────────────────────────────────

def _handle_list_followup(query: str, u: Understanding, ctx: dict, projects: list) -> str:
    codes = ctx.get("last_list_project_codes") or ctx.get("last_project_list") or []
    by_code = {p.get("project_code"): p for p in projects}
    rows = [by_code[c] for c in codes if c in by_code]
    if not rows:
        return "القائمة السابقة لم تتضمن مشاريع مطابقة."
    names = [r.get("project_name_ar") or r.get("project_name_en") or r.get("project_code") for r in rows]
    q_sem = detect_semantic_intent(query)
    intent = q_sem if q_sem else ctx.get("last_list_intent")
    _explanations = {
        "losing_projects":       "هذه المشاريع سجلت صافي ربح سالب في البيانات الحالية.",
        "profitable_projects":   "هذه المشاريع سجلت صافي ربح موجب.",
        "overdue":               "هذه مشاريع تجاوزت تاريخ انتهائها الفعلي وما زالت مفتوحة.",
        "expiring":              "هذه مشاريع ينتهي عقدها خلال 30 يومًا.",
        "ranking_profit_projects":"هذه المشاريع مرتبة من الأعلى إلى الأقل بالربح.",
    }
    lf_type = u.list_followup_type
    if lf_type == "rank":
        field = detect_requested_field(query)
        # Default field for rank: if no explicit field, infer from list intent
        if not field and rows:
            from modules.semantic_dictionary import FIELDS
            _INTENT_DEFAULT_FIELD = {
                "losing_projects": "net_profit", "profitable_projects": "net_profit",
                "ranking_profit": "net_profit", "ranking_revenue": "total_revenue",
                "ranking_backlog": "backlog", "ranking_contract": "total_contract_value",
            }
            default_canonical = _INTENT_DEFAULT_FIELD.get(intent, "net_profit")
            field = FIELDS.get(default_canonical)
        if field and rows:
            from modules.semantic_dictionary import normalize_text as _nt
            q_norm = _nt(query)
            # Detect if asking for worst/lowest vs best/highest
            _WORST_MARKERS = tuple(_nt(v) for v in (
                "اكبر خساره","اكبر خسارة","اسوا","الاسوا","اقل ربح","الاقل",
                "worst","lowest","biggest loss","most loss",
            ))
            _BEST_MARKERS = tuple(_nt(v) for v in (
                "اعلى","الاعلى","اكثر","الاكثر","اكبر ربح","افضل",
                "highest","best","most profit",
            ))
            want_worst = any(m in q_norm for m in _WORST_MARKERS)
            sortable = [(r.get(field.canonical), r) for r in rows if r.get(field.canonical) is not None]
            if sortable:
                if want_worst:
                    # ascending → most negative first
                    sortable.sort(key=lambda x: (x[0] if isinstance(x[0], (int, float)) else 0))
                    top = sortable[0]
                    top_name = top[1].get("project_name_ar") or top[1].get("project_code")
                    val = top[0]
                    from modules.response_formatter import _money
                    val_str = _money(val) if isinstance(val, (int, float)) else str(val)
                    return f"الأكبر خسارة من القائمة السابقة هو «{top_name}» بخسارة {val_str}."
                else:
                    sortable.sort(key=lambda x: -(x[0] if isinstance(x[0], (int, float)) else 0))
                    top = sortable[0]
                    top_name = top[1].get("project_name_ar") or top[1].get("project_code")
                    val = top[0]
                    from modules.response_formatter import _money
                    val_str = _money(val) if isinstance(val, (int, float)) else str(val)
                    return f"الأعلى في {field.label_ar} من القائمة السابقة هو «{top_name}» بقيمة {val_str}."
    exp = _explanations.get(intent, "هذه المشاريع طابقت شروط طلبك السابق.")
    listed = "، ".join(f"«{n}»" for n in names[:10])
    if lf_type == "explain":
        return f"{exp}\n{listed}"
    if lf_type == "why":
        # Give specific explanation based on intent
        _why_explanations = {
            "losing_projects": (
                f"هذه المشاريع تسجل تكاليف أعلى من إيراداتها، مما ينتج عنه صافي ربح سالب. "
                f"الأكبر خسارة هو «{names[0]}»."
                if names else "هذه المشاريع تسجل صافي ربح سالب وفق البيانات الحالية."
            ),
            "expiring": f"هذه المشاريع لديها تواريخ انتهاء عقود خلال 30 يوماً القادمة.",
            "overdue": f"هذه المشاريع تجاوزت تاريخ انتهائها المحدد وما زالت قيد التنفيذ.",
        }
        why_text = _why_explanations.get(intent, exp)
        return why_text
    return f"{exp} المشاريع هي: {listed}."


# ── Comparison handler ────────────────────────────────────────────────────────

def _handle_comparison(query: str, today: date, ctx: dict, projects: list, upd: dict) -> tuple:
    from modules.comparison_engine import resolve_comparison_projects, format_comparison
    pair = resolve_comparison_projects(query, projects)
    if len(pair) == 2:
        text = format_comparison(pair, query)
        codes = [p.get("project_code") for p in pair]
        names = [p.get("project_name_ar") or p.get("project_code") for p in pair]
        upd.update({
            "last_result_type": "comparison",
            "last_comparison": {"codes": codes, "field": None},
            "active_project_code": codes[0],
            "active_project_display_name": names[0],
            "last_project_code": codes[0],
        })
        return text, PROJECT_COMPARISON, None, upd
    text, src, lu = _resolve_and_respond(query, today, projects, ctx)
    upd.update(lu)
    return text, PROJECT_SUMMARY, src, upd


# ── Main orchestrator ─────────────────────────────────────────────────────────

def _orchestrate(u: Understanding, query: str, today: date, ctx: dict, projects: list) -> tuple:
    upd: dict = {}

    if u.intent == SMALL_TALK:
        st = detect_small_talk(query)
        arabic = any("\u0600" <= c <= "\u06ff" for c in query)
        import random
        _greet = random.choice([
            "هلاً وسهلاً! وش تحب تعرف عن المشاريع؟",
            "أهلاً، يسعدني أساعدك. وش تبي تعرف؟",
            "هلاً! شو اللي تحب تعرفه عن المشاريع؟",
        ])
        responses = {
            "salam":       "وعليكم السلام ورحمة الله، أهلاً. وش تحب تعرف عن المشاريع؟",
            "reply_salam": "أهلاً، كيف أقدر أساعدك؟",
            "greeting":    _greet if arabic else "Hello! What would you like to know?",
            "morning":     "صباح النور! جاهز لأي سؤال عن المشاريع.",
            "evening":     "مساء النور! وش تبي تعرف عن المشاريع؟",
            "wellbeing":   "بخير الحمدلله، شكراً. كيف أخدمك؟",
            "thanks":      random.choice(["العفو، وتحت أمرك.", "بكل سرور، وش تحتاج؟", "أهلاً وسهلاً دائماً."]),
            "bye":         random.choice(["مع السلامة، وتحت أمرك.", "يعطيك العافية، إلى اللقاء.", "في أمان الله."]),
        }
        return responses.get(st, "أقدر أساعدك في بيانات المشاريع."), SMALL_TALK, None, upd

    if u.intent == OUT_OF_SCOPE:
        boundary = classify_boundary(query)
        if boundary:
            return boundary_answer(query, boundary), OUT_OF_SCOPE, None, upd
        return "هذا السؤال خارج نطاق بيانات المشاريع المتاحة.", OUT_OF_SCOPE, None, upd

    if u.intent == LIST_FOLLOWUP:
        text = _handle_list_followup(query, u, ctx, projects)
        return text, LIST_FOLLOWUP, None, upd

    if u.intent == PORTFOLIO_SUMMARY:
        kpis = calculate_executive_kpis(projects, today=today)
        upd["last_result_type"] = "portfolio_summary"
        upd["last_result_scope"] = "portfolio"
        return response_formatter.format_portfolio_summary(kpis), PORTFOLIO_KPI, None, upd

    if u.intent == EXEC_ATTENTION:
        upd["last_result_type"] = "executive_attention"
        upd["last_result_scope"] = "portfolio"
        return format_attention_summary(projects, today), EXEC_ATTENTION, None, upd

    if u.intent in {PORTFOLIO_FILTER, PORTFOLIO_RANKING}:
        text, src = _run_pipeline(u.intent, query, today, projects)
        upd.update(_list_ctx(src))
        return text, u.intent, src, upd

    if u.intent == PORTFOLIO_KPI:
        kpi_r = answer_kpi_question(query, today=today, projects=projects, context=ctx)
        if kpi_r:
            if kpi_r.get("kpi_name"):
                upd["last_kpi_name"] = kpi_r["kpi_name"]
                upd["last_result_scope"] = "portfolio"
                upd["last_result_type"] = "portfolio_kpi"
            return kpi_r["answer"], PORTFOLIO_KPI, kpi_r.get("source"), upd
        text, src = _run_pipeline(PORTFOLIO_KPI, query, today, projects)
        return text, PORTFOLIO_KPI, src, upd

    # Project followup from understanding layer
    if u.intent == PROJECT_FOLLOWUP and u.is_followup:
        code = ctx.get("active_project_code") or ctx.get("last_project_code")
        if code:
            row, src = _lookup_verified(code, today, projects, query, u.requested_field)
            if row:
                text = _format_project(row, query, ctx, u.requested_field,
                                        "field_lookup" if u.requested_field else "general_followup")
                upd.update({
                    "active_project_code": code,
                    "last_project_code": code,
                    "last_result_type": "project_kpi" if u.requested_field else "project_summary",
                    "last_requested_metric": u.requested_field,
                })
                return text, PROJECT_KPI if u.requested_field else PROJECT_SUMMARY, src, upd

    if u.intent == PROJECT_KPI:
        kpi_r = answer_kpi_question(query, today=today, projects=projects, context=ctx)
        if kpi_r:
            if kpi_r.get("scope") == "needs_clarification" and kpi_r.get("pending_candidates"):
                upd["pending_project_confirmation"] = {
                    "candidates": kpi_r["pending_candidates"], "kind": "kpi",
                    "kpi_name": kpi_r.get("kpi_name"), "original_query": query,
                }
            if kpi_r.get("kpi_name"):
                upd["last_kpi_name"] = kpi_r["kpi_name"]
                upd["last_result_type"] = "project_kpi"
            if kpi_r.get("project_code"):
                upd["active_project_code"] = kpi_r["project_code"]
                upd["last_project_code"] = kpi_r["project_code"]
            return kpi_r["answer"], PROJECT_KPI, kpi_r.get("source"), upd
        field = u.requested_field
        text, src, lu = _resolve_and_respond(query, today, projects, ctx, topic=field)
        upd.update(lu)
        return text, PROJECT_KPI, src, upd

    if u.intent == CONTRACT_VALUE:
        text, src, lu = _resolve_and_respond(query, today, projects, ctx, topic="total_contract_value")
        upd.update(lu)
        return text, CONTRACT_VALUE, src, upd

    if u.intent == CONTRACT_DOC:
        resolution = resolve_project(query, projects)
        if resolution.status in {"ambiguous","confirmation"}:
            return format_resolution_prompt(resolution), CONTRACT_DOC, None, {
                "pending_project_confirmation": _pending_state(resolution, "contract", original_query=query)
            }
        if resolution.status != "matched":
            if ctx.get("active_project_code"):
                last_name = ctx.get("active_project_display_name") or ctx["active_project_code"]
                return (f"هل تقصد عقد مشروع {last_name}؟", CONTRACT_DOC, None, {
                    "pending_project_confirmation": {
                        "candidates": [{"project_code": ctx["active_project_code"], "display_name": last_name}],
                        "kind": "contract", "original_query": query,
                    }
                })
            return "الرجاء تحديد اسم المشروع للإجابة على سؤال العقد.", CONTRACT_DOC, None, upd
        code = resolution.canonical_project_code
        name = resolution.candidates[0].display_name
        try:
            text = answer_contract_query(query, code, name)
        except ContractQueryError:
            text = "الرجاء تحديد اسم أو رمز المشروع للإجابة على سؤال متعلق بالعقد."
        except Exception:
            logger.exception("Contract RAG: %s", code)
            text = verification.FALLBACK_MESSAGE
        upd.update({"active_project_code": code, "last_project_code": code,
                    "active_project_display_name": name, "last_project_display_name": name,
                    "pending_project_confirmation": None})
        return text, CONTRACT_DOC, None, upd

    if u.intent == PROJECT_COMPARISON:
        return _handle_comparison(query, today, ctx, projects, upd)

    if u.intent == GROUPED_ANALYTICS:
        text, src = _run_pipeline(GROUPED_ANALYTICS, query, today, projects)
        return text, GROUPED_ANALYTICS, src, upd

    # FIX: "كم مشروع/ايش عندنا" - portfolio count query
    from modules.semantic_dictionary import normalize_text as _nt2
    _PORTFOLIO_COUNT_SIGNALS = tuple(_nt2(v) for v in (
        "كم مشروع", "كم عدد", "ايش عندنا من مشاريع", "وش عندنا من مشاريع",
        "اعطني قائمة المشاريع", "عرض كل المشاريع", "وش المشاريع الموجوده",
        "كم عدد المشاريع", "how many projects", "list all projects",
    ))
    q_for_count = _nt2(query)
    if any(sig in q_for_count for sig in _PORTFOLIO_COUNT_SIGNALS):
        kpi_r = answer_kpi_question(query, today=today, projects=projects, context=ctx)
        if kpi_r:
            upd["last_result_type"] = "portfolio_kpi"
            upd["last_result_scope"] = "portfolio"
            return kpi_r["answer"], PORTFOLIO_KPI, kpi_r.get("source"), upd
        # Fallback: portfolio summary
        kpis = calculate_executive_kpis(projects, today=today)
        from modules.response_formatter import format_portfolio_summary
        upd["last_result_type"] = "portfolio_summary"
        return format_portfolio_summary(kpis), PORTFOLIO_KPI, None, upd

    # Try manager/person name search before project lookup
    _PROJECT_KW = tuple(_nt2(v) for v in (
        "مشروع","عقد","ايراد","تكلفة","ربح","backlog","هامش","تقدم","حاله","وضع",
    ))
    q_short = _nt2(query)
    is_person_query = (
        len(query.strip().split()) <= 5
        and not any(kw in q_short for kw in _PROJECT_KW)
        # Allow person search even with active project
    )
    if is_person_query:
        pm_matches = [p for p in projects
                      if p.get("project_manager") and _nt2(query.strip()) in _nt2(p["project_manager"])]
        if pm_matches:
            manager_name = pm_matches[0]["project_manager"]
            proj_names = [p.get("project_name_ar") or p.get("project_code") for p in pm_matches[:5]]
            proj_list = "، ".join(f"«{n}»" for n in proj_names)
            return (
                f"{manager_name} يدير {len(pm_matches)} مشروع في المحفظة: {proj_list}.",
                PROJECT_SUMMARY, None, upd,
            )

    # Default: project summary
    text, src, lu = _resolve_and_respond(query, today, projects, ctx)
    upd.update(lu)
    return text, PROJECT_SUMMARY, src, upd


# ── Main entry ────────────────────────────────────────────────────────────────

def _answer_inner(query: str, today: date, ctx: dict) -> tuple:
    upd: dict = {}

    # Social conversation always outranks pending choices and project context.
    if detect_small_talk(query):
        u = Understanding(intent=SMALL_TALK, scope="unknown", confidence=1.0, method="fast_path")
        return _orchestrate(u, query, today, ctx, [])

    q_normalized = normalize_project_text(query)
    pending_options = (ctx.get("pending_project_confirmation") or {}).get("candidates") or []
    if len(pending_options) >= 2 and any(marker in q_normalized for marker in ("قارن بينهم", "قارن بينهما", "كلاهما")):
        projects = fetch_enriched_projects(today=today)
        by_code = {p.get("project_code"): p for p in projects}
        rows = [by_code.get(item["project_code"]) for item in pending_options[:2]]
        rows = [row for row in rows if row]
        if len(rows) == 2:
            from modules.comparison_engine import format_comparison
            codes = [row["project_code"] for row in rows]
            return format_comparison(rows, query), PROJECT_COMPARISON, None, {
                "comparison_project_ids": codes,
                "last_comparison": {"codes": codes, "field": None},
                "last_result_type": "comparison",
            }

    # ── Step 0: Pending confirmation ─────────────────────────────────────────
    pending = _try_resolve_pending(query, ctx)
    if pending is not None:
        code, name, pstate = pending
        kind = pstate.get("kind", "lookup")
        options = list(pstate.get("candidates") or [])
        upd.update({"active_project_code": code, "last_project_code": code,
                    "active_project_display_name": name, "last_project_display_name": name,
                    "pending_project_confirmation": pstate,
                    "last_disambiguation_options": options,
                    "last_disambiguation_query": pstate.get("original_query"),
                    "selected_disambiguation_index": pstate.get("selected_index")})
        projects = fetch_enriched_projects(today=today)
        if kind == "kpi" and pstate.get("kpi_name"):
            r = answer_kpi_for_known_project(pstate["kpi_name"], code, name, today, projects)
            upd["last_kpi_name"] = r["kpi_name"]
            return r["answer"], PROJECT_KPI, r.get("source"), upd
        if kind == "contract":
            orig = pstate.get("original_query") or query
            try:
                text = answer_contract_query(orig, code, name)
            except Exception:
                text = verification.FALLBACK_MESSAGE
            return text, CONTRACT_DOC, None, upd
        if kind == "contract_metrics":
            orig = pstate.get("original_query") or query
            request = analyze_contract_request(orig)
            row, src = _lookup_verified(code, today, projects, orig)
            if not row or not request:
                return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_KPI, None, upd
            upd.update({"last_result_type": "project_kpi", "last_user_intent": request.operation})
            return render_contract_answer(row, request, today), PROJECT_KPI, src, upd
        orig = pstate.get("original_query") or query
        orig_field = detect_requested_field(orig)
        row, src = _lookup_verified(code, today, projects, orig,
                                     orig_field.canonical if orig_field else None)
        if not row:
            return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_SUMMARY, None, upd
        text = _format_project(row, orig, ctx, orig_field.canonical if orig_field else None, "field_lookup")
        upd["last_result_type"] = "project_kpi" if orig_field else "project_summary"
        return text, PROJECT_KPI if orig_field else PROJECT_SUMMARY, src, upd

    # Contract/timeline semantics share one structured operation and may carry
    # several metrics. Resolve a current-turn project before active context.
    contract_request = analyze_contract_request(query)
    if contract_request:
        projects = fetch_enriched_projects(today=today)
        phrase = extract_project_phrase(query)
        resolution = resolve_project(query, projects) if len(phrase) >= 3 else None
        if resolution and resolution.status in {"ambiguous", "confirmation"}:
            pending_state = _pending_state(resolution, "contract_metrics", original_query=query)
            return format_resolution_prompt(resolution), PROJECT_KPI, None, {
                "pending_project_confirmation": pending_state,
                "last_disambiguation_options": list(pending_state["candidates"]),
                "last_disambiguation_query": query,
            }
        if resolution and resolution.status == "matched":
            code = resolution.canonical_project_code
            name = resolution.candidates[0].display_name
        else:
            code = ctx.get("active_project_code") or ctx.get("last_project_code")
            name = ctx.get("active_project_display_name") or ctx.get("last_project_display_name")
        if not code:
            return "تقصد أي مشروع؟", PROJECT_KPI, None, {}
        row, src = _lookup_verified(code, today, projects, query)
        if not row:
            return verification.PROJECT_FALLBACK_MESSAGE, PROJECT_KPI, None, {}
        return render_contract_answer(row, contract_request, today), PROJECT_KPI, src, {
            "active_project_code": code, "active_project_display_name": name,
            "last_project_code": code, "last_project_display_name": name,
            "last_result_type": "project_kpi",
            "last_requested_metric": contract_request.metrics[-1] if contract_request.metrics else contract_request.operation,
            "last_user_intent": contract_request.operation,
        }

    # A reference to the previous project uses bounded project history.
    q_nav = normalize_project_text(query)
    if any(marker in q_nav for marker in (
        "المشروع اللي قبله", "المشروع اللي قبل", "طيب السابق", "المشروع السابق", "طيب المشروع الثاني",
    )):
        recent = ctx.get("recent_project_ids") or []
        if len(recent) > 1:
            projects = fetch_enriched_projects(today=today)
            code = recent[1]
            row, src = _lookup_verified(code, today, projects, query)
            if row:
                name = row.get("project_name_ar") or row.get("project_name_en") or code
                return _format_project(row, query, ctx, None, "summary"), PROJECT_SUMMARY, src, {
                    "active_project_code": code, "active_project_display_name": name,
                    "last_project_code": code, "last_project_display_name": name,
                    "last_result_type": "project_summary", "last_requested_metric": None,
                }

    if q_nav in {"مشروعهم", "هذا المشروع", "المشروع ذا", "نفس المشروع"} and ctx.get("active_project_code"):
        projects = fetch_enriched_projects(today=today)
        code = ctx["active_project_code"]
        row, src = _lookup_verified(code, today, projects, query)
        if row:
            return _format_project(row, query, ctx, None, "summary"), PROJECT_SUMMARY, src, {
                "active_project_code": code, "last_project_code": code,
                "last_result_type": "project_summary", "last_requested_metric": None,
            }

    # Multi-field followups are answered deterministically from one verified
    # project snapshot instead of discarding every field after the first.
    requested_fields = detect_requested_fields(query)
    explicit_named_project = "مشروع" in normalize_project_text(query).split() and len(extract_project_phrase(query)) >= 3
    if len(requested_fields) > 1 and ctx.get("active_project_code") and not explicit_named_project:
        projects = fetch_enriched_projects(today=today)
        code = ctx["active_project_code"]
        row, src = _lookup_verified(code, today, projects, query)
        if row:
            metrics = [field.canonical for field in requested_fields]
            return format_project_metrics(row, metrics), PROJECT_KPI, src, {
                "active_project_code": code, "last_project_code": code,
                "last_result_type": "project_kpi", "last_requested_metric": metrics[-1],
            }

    # ── Step 1: Follow-up First Gate ─────────────────────────────────────────
    gate = followup_gate_check(query, ctx)
    if gate.fires:
        logger.info("FollowupGate fired: field=%s intent=%s confidence=%.2f reason=%s",
                    gate.field, gate.intent, gate.confidence, gate.reason)
        projects = fetch_enriched_projects(today=today)
        return _handle_followup_gate(gate, query, today, ctx, projects)

    # ── Step 2: Understand ────────────────────────────────────────────────────
    projects = fetch_enriched_projects(today=today)
    u = understand(query, ctx)
    logger.info("Understanding: intent=%s scope=%s method=%s conf=%.2f",
                u.intent, u.scope, u.method, u.confidence)

    # ── Step 2b: Methodology ──────────────────────────────────────────────────
    if is_methodology_question(query) and ctx.get("last_requested_field") in FIELDS:
        defn = FIELDS[ctx["last_requested_field"]]
        return (f"تمت الإجابة من الحقل «{defn.source_column}» ({defn.label_ar}).",
                PROJECT_KPI, None, upd)

    # ── Step 3: Orchestrate ───────────────────────────────────────────────────
    return _orchestrate(u, query, today, ctx, projects)


def answer(query: str, user_id: str = "anonymous",
           source: str = "flask_ui", ip_address: str = "",
           session_id: str | None = None) -> dict:
    query = query.strip()[:2000]
    if not query:
        return {"answer": "من فضلك اكتب سؤالك.", "query_type": "none"}
    today = datetime.now(ZoneInfo("Asia/Riyadh")).date()
    session_id = session_id or "no-session"
    ctx = session_context.get_context(session_id)
    try:
        response_text, query_type, source_info, ctx_updates = _answer_inner(query, today, ctx)
    except Exception:
        logger.exception("Unhandled error: %.60s", query)
        response_text, query_type, source_info, ctx_updates = (
            verification.FALLBACK_MESSAGE, "error", None, {}
        )
    if ctx_updates:
        session_context.update_context(session_id, **ctx_updates)
    try:
        log_query(query_text=query, query_type=query_type, response_text=response_text,
                  user_id=user_id, source=source, ip_address=ip_address)
    except Exception:
        logger.exception("Audit log failed")
    result = {"answer": response_text, "query_type": query_type}
    if source_info and is_methodology_question(query):
        result["source"] = source_info
    return result


def generate_report(report_type: str) -> str:
    today = datetime.now(ZoneInfo("Asia/Riyadh")).date()
    try:
        projects = fetch_enriched_projects(today=today)
        if not projects:
            return "لا توجد بيانات. شغل: python scheduler.py"
        kpis = calculate_executive_kpis(projects, today=today)
        bu_summary = summarize_by_bu(projects)
        losing = query_executor.execute(query_schema.validate_query_spec({
            "filters": [{"column":"net_profit","op":"<","value":0}],
            "sort": {"column":"net_profit","direction":"ASC"}, "limit":15,
        }), today=today, projects=projects)
        expiring = query_executor.execute(query_schema.validate_query_spec({
            "filters": [{"column":"days_remaining","op":"between","value":0,"value2":90}],
            "sort": {"column":"days_remaining","direction":"ASC"}, "limit":15,
        }), today=today, projects=projects)
        top_bl = query_executor.execute(query_schema.validate_query_spec({
            "filters":[], "sort":{"column":"backlog","direction":"DESC"}, "limit":15,
        }), today=today, projects=projects)
        for r in (losing, expiring, top_bl):
            if not verification.verify(r, today=today).ok:
                return verification.FALLBACK_MESSAGE
        verified = {
            "portfolio_kpis": kpis, "bu_summary": bu_summary,
            "losing_projects": [_trim(p) for p in losing["rows"]],
            "expiring_within_90_days": [_trim(p) for p in expiring["rows"]],
            "top_backlog_projects": [_trim(p) for p in top_bl["rows"]],
        }
        instructions = {
            "executive": "قدم تقرير تنفيذي موجز للـ VP: 1) ملخص القطاع 2) أبرز 3 نقاط 3) أبرز المخاطر 4) توصيتان.",
            "risk":      "قدم تقرير مخاطر: 1) المشاريع بربحية سالبة 2) العقود تنتهي خلال 90 يوماً 3) أعلى Backlog.",
            "backlog":   "قدم تحليل Backlog: 1) التوزيع حسب BU 2) أعلى المشاريع 3) نسبة Backlog.",
            "collection":"قدم متابعة تحصيل: 1) Backlog حسب BU 2) أعلى المشاريع 3) المشاريع القريبة من الانتهاء.",
        }
        return response_formatter.format_answer(instructions.get(report_type, instructions["executive"]), verified)
    except Exception:
        logger.exception("Report failed: %s", report_type)
        return verification.FALLBACK_MESSAGE


def invalidate_cache():
    """No-op: pipeline reads fresh from DB."""
