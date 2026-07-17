"""
modules/understanding.py
-------------------------
Semantic Understanding Layer — يفهم معنى السؤال قبل أي شيء آخر.

بدل ما كل module يحاول يكتشف المعنى لنفسه بـ keywords،
هذا الـ module يفهم الكل في مكان واحد ويعطي structured meaning.

Intent categories:
    project_summary     — ملخص مشروع واحد
    project_followup    — متابعة عن مشروع في السياق
    project_comparison  — مقارنة بين مشروعين+
    portfolio_kpi       — KPI للمحفظة كاملة
    project_kpi         — KPI لمشروع محدد
    portfolio_filter    — قائمة مشاريع بشرط (خسرانة، تنتهي...)
    portfolio_ranking   — ترتيب (أعلى، أقل...)
    portfolio_summary   — ملخص/نبذة المحفظة
    executive_attention — وش يحتاج متابعة
    list_followup       — متابعة عن قائمة سابقة
    contract_document   — شروط/بنود عقد (RAG)
    contract_value      — قيمة عقد (DB)
    grouped_analytics   — تجميع بـ BU/PM/segment
    small_talk          — تحية/محادثة عامة
    out_of_scope        — خارج النطاق
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any


import config
from modules.contract_semantics import parse_future_period_days
from modules.project_entity_resolver import normalize_project_text
from modules.semantic_dictionary import (
    FIELDS,
    detect_requested_field,
    detect_semantic_intent,
    detect_small_talk,
    detect_portfolio_operation,
    is_contract_document_question,
    is_previous_list_followup,
    normalize_text,
)

logger = logging.getLogger(__name__)
_client = None


def _get_openai():
    global _client
    if _client is None:
        if not config.AZURE_OPENAI_KEY:
            raise RuntimeError("Azure OpenAI is not configured")
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _client


# ── Intent constants ─────────────────────────────────────────────────────────
PROJECT_SUMMARY    = "project_summary"
PROJECT_FOLLOWUP   = "project_followup"
PROJECT_COMPARISON = "project_comparison"
PORTFOLIO_KPI      = "portfolio_kpi"
PROJECT_KPI        = "project_kpi"
PORTFOLIO_FILTER   = "portfolio_filter"
PORTFOLIO_RANKING  = "portfolio_ranking"
PORTFOLIO_SUMMARY  = "portfolio_summary"
EXEC_ATTENTION     = "executive_attention"
LIST_FOLLOWUP      = "list_followup"
CONTRACT_DOC       = "contract_document"
CONTRACT_VALUE     = "contract_value"
GROUPED_ANALYTICS  = "grouped_analytics"
SMALL_TALK         = "small_talk"
OUT_OF_SCOPE       = "out_of_scope"

ALL_INTENTS = {
    PROJECT_SUMMARY, PROJECT_FOLLOWUP, PROJECT_COMPARISON,
    PORTFOLIO_KPI, PROJECT_KPI,
    PORTFOLIO_FILTER, PORTFOLIO_RANKING, PORTFOLIO_SUMMARY,
    EXEC_ATTENTION, LIST_FOLLOWUP,
    CONTRACT_DOC, CONTRACT_VALUE,
    GROUPED_ANALYTICS, SMALL_TALK, OUT_OF_SCOPE,
}

# ── Scope constants ──────────────────────────────────────────────────────────
SCOPE_PROJECT   = "project"
SCOPE_PORTFOLIO = "portfolio"
SCOPE_LIST      = "list"
SCOPE_UNKNOWN   = "unknown"


@dataclass
class Understanding:
    """Structured meaning of a user message."""
    intent: str
    scope: str
    confidence: float = 1.0

    # Project references
    project_mentions: list[str] = field(default_factory=list)   # raw names/codes from text
    is_followup: bool = False                                    # references last project in context

    # Field / metric
    requested_field: str | None = None    # canonical field name if asking about a specific field
    requested_kpi: str | None = None      # KPI name if asking about a KPI

    # For list followups
    list_followup_type: str | None = None  # "why", "explain", "rank", "pick"

    # For comparison
    comparison_projects: list[str] = field(default_factory=list)

    # For filters/rankings
    filter_intent: str | None = None   # e.g. "losing_projects", "expiring"
    ranking_intent: str | None = None  # e.g. "ranking_profit", "ranking_contract"

    # For grouped analytics
    group_by: str | None = None        # "bu", "pm", "segment", "dept"

    # Raw semantic detection (for fallback)
    semantic_raw: str | None = None

    # Debug
    method: str = "unknown"   # how we determined intent: "deterministic" | "llm" | "fast_path"


# ── Portfolio scope markers ──────────────────────────────────────────────────
_PORTFOLIO_MARKERS = tuple(normalize_text(v) for v in (
    "محفظه", "المحفظه", "المحفظة", "محفظة",
    "كل المشاريع", "جميع المشاريع", "لجميع المشاريع", "لكل المشاريع",
    "عبر المشاريع", "portfolio", "overall",
    "الكلي", "الاجمالي", "إجمالي",
    "ملخص المحفظة", "ملخص المحفظه",
))

# ── List followup sub-types ──────────────────────────────────────────────────
_LIST_WHY    = tuple(normalize_text(v) for v in ("ليش", "لماذا", "why", "علاش"))
_LIST_EXPL   = tuple(normalize_text(v) for v in ("ايش هذي", "وش هذي", "اشرح", "explain", "what are these"))
_LIST_RANK   = tuple(normalize_text(v) for v in (
    "ايهم اعلى", "ايهم اكبر", "ايهم اقل", "ايهم افضل", "ايهم اسوا",
    "ايهم اكثر", "ايهم اقرب", "which is highest", "which is biggest",
    "which is lowest", "which is best",
))
_LIST_PICK   = tuple(normalize_text(v) for v in ("which one", "أي واحد", "اي واحد", "اختر"))

# ── Contract DB vs RAG ──────────────────────────────────────────────────────
_CONTRACT_DB_SIGNALS = tuple(normalize_text(v) for v in (
    "قيمة العقد", "كم قيمة العقد", "قيمة المشروع", "إجمالي قيمة العقد",
    "contract value", "total contract value", "كم عقده", "عقده كم",
    "تعديلات العقد", "amendment", "كم التعديلات",
))

# ── Grouped analytics signals ────────────────────────────────────────────────
_GROUP_SIGNALS = {
    "bu":      tuple(normalize_text(v) for v in ("بي يو", "وحدة الأعمال", "business unit", " bu ", "حسب القطاع")),
    "pm":      tuple(normalize_text(v) for v in ("مدير المشروع", "project manager", "حسب المدير", "pm")),
    "segment": tuple(normalize_text(v) for v in ("القطاع", "segment", "حسب القطاع")),
    "dept":    tuple(normalize_text(v) for v in ("الإدارة", "القسم", "department", "dept")),
}


def _is_portfolio_scope(q_norm: str) -> bool:
    return any(m in q_norm for m in _PORTFOLIO_MARKERS)


def _detect_list_followup_type(q_norm: str) -> str | None:
    if any(m in q_norm for m in _LIST_RANK):  return "rank"
    if any(m in q_norm for m in _LIST_WHY):   return "why"
    if any(m in q_norm for m in _LIST_EXPL):  return "explain"
    if any(m in q_norm for m in _LIST_PICK):  return "pick"
    return None


def _detect_group_by(q_norm: str) -> str | None:
    for key, signals in _GROUP_SIGNALS.items():
        if any(s in q_norm for s in signals):
            return key
    return None


def _extract_project_mentions(query: str) -> list[str]:
    """
    Extract raw project name/code mentions from query text.
    Returns a list of candidate strings for the entity resolver.
    We don't resolve here — just extract.
    """
    from modules.project_entity_resolver import extract_project_phrase
    phrase = extract_project_phrase(query)
    if phrase and len(phrase) >= 3:
        return [phrase]
    return []


# ── LLM understanding (fallback for ambiguous queries) ──────────────────────

_LLM_TOOL = {
    "type": "function",
    "function": {
        "name": "understand_query",
        "description": "Classify a portfolio management question into structured meaning.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": sorted(ALL_INTENTS),
                    "description": "The primary intent of the question.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["project", "portfolio", "list", "unknown"],
                },
                "is_followup": {
                    "type": "boolean",
                    "description": "True if the question refers to a project/result already in context.",
                },
                "project_mentions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Raw project names or codes mentioned in the question.",
                },
                "filter_intent": {
                    "type": "string",
                    "description": "For portfolio_filter: losing_projects, expiring, overdue, profitable_projects, etc.",
                },
                "ranking_intent": {
                    "type": "string",
                    "description": "For portfolio_ranking: ranking_profit, ranking_contract, ranking_backlog, etc.",
                },
                "group_by": {
                    "type": "string",
                    "description": "For grouped_analytics: bu, pm, segment, dept.",
                },
                "list_followup_type": {
                    "type": "string",
                    "enum": ["why", "explain", "rank", "pick"],
                    "description": "Sub-type for list_followup intent.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0-1.",
                },
            },
            "required": ["intent", "scope", "is_followup", "confidence"],
        },
    },
}

_LLM_SYSTEM = """أنت نظام فهم الاستعلامات لمنصة إدارة المحفظة التنفيذية.
مهمتك: تحليل سؤال المستخدم وإرجاع معناه الهيكلي فقط — لا تجيب على السؤال.

السياق المتاح:
- last_project: اسم آخر مشروع ناقشه المستخدم
- last_result_scope: هل آخر نتيجة كانت مشروع أو محفظة أو قائمة
- last_list_intent: نوع آخر قائمة ظهرت

قواعد التصنيف:
- portfolio_kpi: سؤال عن KPI للمحفظة كاملة (إجمالي الإيرادات، هامش الربح الكلي...)
- project_kpi: سؤال عن KPI لمشروع محدد
- project_followup: سؤال قصير يشير لمشروع في السياق (وتكاليفه؟ ومتى ينتهي؟)
- list_followup: متابعة عن قائمة سابقة (ليش؟ وش هذي؟ أيهم أكبر؟)
- contract_value: يسأل عن قيمة العقد الرقمية
- contract_document: يسأل عن شروط/بنود/سياسات العقد (يحتاج RAG)
- portfolio_summary: يريد ملخص شامل للمحفظة
- executive_attention: وش يحتاج متابعة / الأولويات / المخاطر

أجب فقط باستدعاء الأداة understand_query."""


def understand(query: str, ctx: dict | None = None) -> Understanding:
    """
    Main entry point. Returns structured Understanding of the query.
    Uses deterministic fast-path first, falls back to LLM if needed.
    """
    ctx = ctx or {}
    q_norm = normalize_text(query)
    sem = detect_semantic_intent(query)
    portfolio_op = detect_portfolio_operation(query)
    temporal_filter = None
    if any(term in q_norm for term in ("تنتهي", "ينتهي", "تخلص", "ending")) and parse_future_period_days(query):
        temporal_filter = "ending_within_month"
    elif any(term in q_norm for term in ("بدات هالسنه", "بدأت هالسنة", "بدات هذه السنه", "بدأت هذه السنة", "started this year")):
        temporal_filter = "started_this_year"
    numeric_filter = bool(
        re.search(r"(?:هامش\w*|margin)\s*(?:اقل من|دون|تحت|below|less than)\s*\d", q_norm)
    )

    # ── Fast path 1: Small talk ──────────────────────────────────────────────
    st = detect_small_talk(query)
    if st:
        return Understanding(intent=SMALL_TALK, scope=SCOPE_UNKNOWN,
                             confidence=1.0, method="fast_path")

    # Structural portfolio operations must be routed before person/project
    # lookup heuristics.  "كم المشاريع المكتملة" is a count, and "أعلى
    # مشروع ربحية" is a rank even though both contain the word project.
    if portfolio_op:
        if portfolio_op["operation"] == "count":
            return Understanding(
                intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
                filter_intent="status_count" if portfolio_op.get("status") else "project_count",
                confidence=0.98, method="structural",
            )
        return Understanding(
            intent=PORTFOLIO_RANKING, scope=SCOPE_PORTFOLIO,
            ranking_intent=f"rank:{portfolio_op['metric']}:{portfolio_op['direction']}",
            confidence=0.98, method="structural",
        )
    if temporal_filter:
        return Understanding(
            intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
            filter_intent=temporal_filter, confidence=0.98, method="structural",
        )
    if numeric_filter:
        return Understanding(
            intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
            filter_intent="numeric_margin", confidence=0.98, method="structural",
        )

    # ── Fast path 2: Out of scope (personal/contact) ─────────────────────────
    from modules.knowledge_boundary import classify_boundary
    boundary = classify_boundary(query)
    req_field = detect_requested_field(query)
    if boundary and not sem and not (boundary.topic == "unknown_person" and req_field):
        return Understanding(intent=OUT_OF_SCOPE, scope=SCOPE_UNKNOWN,
                             confidence=1.0, method="fast_path")

    # ── Fast path 3: Semantic intent (deterministic patterns) ────────────────
    # List followup — must check BEFORE semantic, because "ليش" after a list
    has_list_ctx = ctx.get("last_result_scope") == "list"
    lf_type = _detect_list_followup_type(q_norm)
    if has_list_ctx and (is_previous_list_followup(query) or lf_type):
        actual_lf_type = lf_type or "explain"
        return Understanding(
            intent=LIST_FOLLOWUP, scope=SCOPE_LIST,
            confidence=1.0, list_followup_type=actual_lf_type,
            semantic_raw=sem, method="fast_path",
        )

    # Portfolio filter patterns
    _FILTER_PATTERNS = {
        "losing_projects":   ("خسرانه", "خسرانة", "المشاريع خسران", "مشاريع خاسره", "losing"),
        "expiring":          ("قربت تخلص", "قربت تنتهي", "تنتهي قريبا", "expiring", "بنتهي", "راح تنتهي", "خلال شهر", "اللي راح تنتهي", "اللي بنتهي"),
        "overdue":           ("متاخره", "تجاوزت التاريخ", "overdue"),
        "profitable_projects": ("الرابحه", "مشاريع رابحه", "profitable", "ربحانه", "ربحان", "مشاريع ربحانه"),
        "ongoing_projects":    ("المشاريع الجاريه", "المشاريع الجارية", "مشروع جاري", "مشروع شغال", "المشاريع المستمره", "المشاريع المستمرة", "ongoing projects"),
    }
    for filter_key, patterns in _FILTER_PATTERNS.items():
        if any(normalize_text(p) in q_norm for p in patterns):
            return Understanding(
                intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
                confidence=0.95, filter_intent=filter_key,
                method="deterministic",
            )

    # Explicit semantic intents from semantic_dictionary
    if sem:
        if sem.startswith("ranking_"):
            return Understanding(intent=PORTFOLIO_RANKING, scope=SCOPE_PORTFOLIO,
                                 confidence=0.95, ranking_intent=sem, method="deterministic")
        if sem == "executive_attention":
            return Understanding(intent=EXEC_ATTENTION, scope=SCOPE_PORTFOLIO,
                                 confidence=0.95, method="deterministic")
        if sem == "portfolio_summary":
            return Understanding(intent=PORTFOLIO_SUMMARY, scope=SCOPE_PORTFOLIO,
                                 confidence=0.95, method="deterministic")
        if sem == "losing_projects":
            return Understanding(intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
                                 confidence=0.95, filter_intent="losing_projects", method="deterministic")
        if sem == "expiring":
            return Understanding(intent=PORTFOLIO_FILTER, scope=SCOPE_PORTFOLIO,
                                 confidence=0.95, filter_intent="expiring", method="deterministic")

    # ── Fast path 4: Portfolio scope (explicit) ──────────────────────────────
    # Extended portfolio scope: "عندنا X مشاريع", "كم مشروع"
    _EXTENDED_PORTFOLIO = tuple(normalize_text(v) for v in (
        "عندنا مشاريع", "عندكم مشاريع", "كم مشروع", "كم عدد",
        "اعطني قائمة", "عرض كل", "وش المشاريع", "ايش عندنا",
    ))
    is_portfolio = _is_portfolio_scope(q_norm) or any(m in q_norm for m in _EXTENDED_PORTFOLIO)

    # ── Fast path 5: Contract routing ────────────────────────────────────────
    if is_contract_document_question(query):
        db_contract = any(s in q_norm for s in _CONTRACT_DB_SIGNALS)
        if not db_contract:
            return Understanding(intent=CONTRACT_DOC, scope=SCOPE_PROJECT,
                                 project_mentions=_extract_project_mentions(query),
                                 confidence=0.9, method="deterministic")

    if any(s in q_norm for s in _CONTRACT_DB_SIGNALS):
        return Understanding(intent=CONTRACT_VALUE, scope=SCOPE_PROJECT,
                             project_mentions=_extract_project_mentions(query),
                             confidence=0.9, method="deterministic")

    # ── Fast path 6: Field/KPI detection ────────────────────────────────────
    field_def = detect_requested_field(query)
    if field_def:
        # If portfolio scope is explicit → always portfolio KPI
        # (profit margin, backlog, revenue etc. are all valid portfolio KPIs)
        if is_portfolio:
            return Understanding(
                intent=PORTFOLIO_KPI, scope=SCOPE_PORTFOLIO,
                requested_field=field_def.canonical,
                confidence=0.92, method="deterministic",
            )
        # Followup about current project?
        has_project_ctx = bool(ctx.get("last_project_code") or ctx.get("active_project_code"))
        mentions = _extract_project_mentions(query)
        is_fo = has_project_ctx and len(mentions) == 0
        return Understanding(
            intent=PROJECT_FOLLOWUP if is_fo else PROJECT_KPI,
            scope=SCOPE_PROJECT,
            requested_field=field_def.canonical,
            is_followup=is_fo,
            project_mentions=mentions,
            confidence=0.9, method="deterministic",
        )

    # ── Fast path 7: Grouped analytics ──────────────────────────────────────
    group_by = _detect_group_by(q_norm)
    if group_by:
        return Understanding(intent=GROUPED_ANALYTICS, scope=SCOPE_PORTFOLIO,
                             group_by=group_by, confidence=0.85, method="deterministic")

    # ── Fast path 8: Comparison signals ─────────────────────────────────────
    _CMP_SIGNALS = tuple(normalize_text(v) for v in (
        "قارن", "مقارنة", "مقابل", "vs", "versus", "compare",
    ))
    if any(s in q_norm for s in _CMP_SIGNALS):
        return Understanding(
            intent=PROJECT_COMPARISON, scope=SCOPE_PROJECT,
            project_mentions=_extract_project_mentions(query),
            confidence=0.85, method="deterministic",
        )

    # ── Fast path 9: Short followup markers ─────────────────────────────────
    _FOLLOWUP_MARKERS = tuple(normalize_text(v) for v in (
        "وتكلفته", "وتكاليفه", "وربحه", "وايراداته", "وهامشه",
        "وعقده", "وباكلوجه", "ونسبته", "وتقدمه", "وحالته", "ومتى ينتهي",
        "تكلفته", "تكاليفه", "تكلفتها", "ربحه", "ربحها", "هامشه",
        "وضعه", "حالته", "تقدمه", "ايراداته",
    ))
    if ctx.get("last_project_code") and any(m in q_norm for m in _FOLLOWUP_MARKERS):
        return Understanding(
            intent=PROJECT_FOLLOWUP, scope=SCOPE_PROJECT,
            is_followup=True,
            requested_field=field_def.canonical if field_def else None,
            confidence=0.95, method="fast_path",
        )

    # ── Fast path 10: Portfolio scope with KPI alias ─────────────────────────
    if is_portfolio:
        from modules.kpi_responder import identify_kpi
        kpi = identify_kpi(query)
        if kpi:
            return Understanding(intent=PORTFOLIO_KPI, scope=SCOPE_PORTFOLIO,
                                 requested_kpi=kpi, confidence=0.9, method="deterministic")
        return Understanding(intent=PORTFOLIO_SUMMARY, scope=SCOPE_PORTFOLIO,
                             confidence=0.8, method="deterministic")

    # ── LLM fallback ────────────────────────────────────────────────────────
    return _llm_understand(query, ctx, q_norm)


def _llm_understand(query: str, ctx: dict, q_norm: str) -> Understanding:
    """Use LLM to understand ambiguous queries."""
    try:
        ctx_summary = {
            "last_project": ctx.get("last_project_display_name") or ctx.get("last_project_code"),
            "last_result_scope": ctx.get("last_result_scope"),
            "last_list_intent": ctx.get("last_list_intent"),
        }
        messages = [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": f"Context: {json.dumps(ctx_summary, ensure_ascii=False)}\nQuestion: {query}"},
        ]
        resp = _get_openai().chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=[_LLM_TOOL],
            tool_choice={"type": "function", "function": {"name": "understand_query"}},
            temperature=0,
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("No tool call")
        parsed = json.loads(tool_calls[0].function.arguments)
        intent = parsed.get("intent")
        scope = parsed.get("scope")
        if intent not in ALL_INTENTS:
            raise ValueError(f"Unsupported intent: {intent!r}")
        if scope not in {SCOPE_PROJECT, SCOPE_PORTFOLIO, SCOPE_LIST, SCOPE_UNKNOWN}:
            raise ValueError(f"Unsupported scope: {scope!r}")
        confidence = float(parsed.get("confidence"))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return Understanding(
            intent=intent,
            scope=scope,
            confidence=confidence,
            is_followup=bool(parsed.get("is_followup", False)),
            project_mentions=parsed.get("project_mentions") or [],
            filter_intent=parsed.get("filter_intent"),
            ranking_intent=parsed.get("ranking_intent"),
            group_by=parsed.get("group_by"),
            list_followup_type=parsed.get("list_followup_type"),
            method="llm",
        )
    except Exception as exc:
        logger.warning("LLM understanding failed: %s | query=%.60s", exc, query)
        # Graceful fallback
        has_ctx = bool(ctx.get("last_project_code"))
        mentions = _extract_project_mentions(query)
        if has_ctx and not mentions:
            return Understanding(intent=PROJECT_FOLLOWUP, scope=SCOPE_PROJECT,
                                 is_followup=True, confidence=0.5, method="fallback")
        return Understanding(intent=PROJECT_SUMMARY, scope=SCOPE_PROJECT,
                             project_mentions=mentions, confidence=0.5, method="fallback")
