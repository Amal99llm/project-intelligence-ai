"""
modules/semantic_interpreter.py
--------------------------------
Layer 1 of the centralized semantic interpretation pipeline (see the plan
at C:\\Users\\akram\\.claude\\plans\\misty-mixing-candy.md for the full
architecture). Gated behind config.SEMANTIC_INTERPRETER_ENABLED.

`interpret(query, ctx, projects)` is the ONLY thing this module does: it
turns one user message into a validated `Interpretation` -- an intent, a
scope, RAW entity mentions, requested operations, and context references.

Hard rules, enforced by construction, not just documented:
  - The model never sees the database and never produces a database value.
    It extracts entities as the literal words the user wrote (`entities.*`
    from the LLM call are raw text spans); resolving a mention to a real
    canonical DB value is entirely modules.entity_resolvers' job (Layer 2),
    invoked here as a synchronous merge step right after the LLM call
    returns, before any caller ever sees the Interpretation.
  - The model never computes anything, never picks a final project/row,
    and never decides the final answer text -- it only classifies meaning.
  - A low-confidence or entity-resolution failure never becomes a guess:
    it becomes `requires_clarification=True` naming the ONE ambiguous
    entity, never the whole query.
  - Any LLM failure (timeout, malformed response, unsupported intent)
    returns a CONTROLLED_FALLBACK Interpretation rather than raising --
    exactly like modules.understanding._llm_understand's existing
    try/except-to-safe-fallback pattern, which this mirrors on purpose.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import config
from modules.entity_resolvers import (
    resolve_bu,
    resolve_department,
    resolve_manager,
    resolve_program,
    resolve_segment,
    resolve_status,
)
from modules.intent_schema import ALL_INTENTS, CONTROLLED_FALLBACK, GROUP_FIELDS, SMALL_TALK
from modules.semantic_dictionary import detect_requested_field, detect_small_talk, normalize_text

logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.85
LOW_CONFIDENCE_THRESHOLD = 0.6

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


# ── Structured result shape ──────────────────────────────────────────────

@dataclass
class EntityMentions:
    projects: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    business_units: list[str] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    managers: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)


@dataclass
class Operations:
    filter: bool = False
    group_by: str | None = None
    sort: str | None = None
    limit: int | None = None
    compare: bool = False


@dataclass
class References:
    active_project: bool = False
    previous_list: bool = False
    comparison: bool = False
    ordinal: str | None = None


@dataclass
class UnresolvedEntity:
    kind: str            # "status" | "department" | "bu" | "segment" | "program" | "manager"
    mention: str
    candidates: list[str] = field(default_factory=list)


@dataclass
class Interpretation:
    intent: str
    scope: str
    entities: EntityMentions
    operations: Operations
    references: References
    confidence: float
    requires_clarification: bool = False
    clarification_question: str | None = None
    unresolved_entity: UnresolvedEntity | None = None
    method: str = "llm"   # "llm" | "deterministic_pre_pass" | "llm_error" | "llm_low_confidence"


# ── Deterministic pre-pass (structural, not semantic) ───────────────────
# Only small talk lives here. Pending-clarification/ordinal replies are
# deliberately NOT reimplemented here -- modules.ai_engine already has a
# robust, tested resolver for that (_try_resolve_pending) and callers must
# check it before ever calling interpret(), not duplicate its logic.

def _deterministic_pre_pass(query: str) -> Interpretation | None:
    if detect_small_talk(query):
        return Interpretation(
            intent=SMALL_TALK, scope="unknown",
            entities=EntityMentions(), operations=Operations(), references=References(),
            confidence=1.0, method="deterministic_pre_pass",
        )
    return None


# ── LLM structured extraction ────────────────────────────────────────────

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "projects": {"type": "array", "items": {"type": "string"}},
        "departments": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "array", "items": {"type": "string"}},
        "programs": {"type": "array", "items": {"type": "string"}},
        "business_units": {"type": "array", "items": {"type": "string"}},
        "segments": {"type": "array", "items": {"type": "string"}},
        "managers": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
    },
}

_INTERPRET_TOOL = {
    "type": "function",
    "function": {
        "name": "interpret_query",
        "description": (
            "Classify a project-portfolio question into structured meaning. Extract entities "
            "as the literal words the user wrote -- never a database code, an English column "
            "name, or a translated/resolved value. You never query data, compute a result, or "
            "choose a final record; you only classify meaning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": sorted(ALL_INTENTS)},
                "scope": {"type": "string", "enum": ["portfolio", "project", "list", "unknown"]},
                "entities": _ENTITY_SCHEMA,
                "operations": {
                    "type": "object",
                    "properties": {
                        "filter": {"type": "boolean"},
                        "group_by": {
                            "type": ["string", "null"],
                            "enum": sorted(GROUP_FIELDS) + [None],
                            "description": "Canonical column to group by -- never a free-text label.",
                        },
                        "sort": {
                            "type": ["string", "null"],
                            "enum": ["asc", "desc", None],
                            "description": "Direction only. The metric to sort/rank by belongs in entities.metrics.",
                        },
                        "limit": {"type": ["integer", "null"]},
                        "compare": {"type": "boolean"},
                    },
                },
                "references": {
                    "type": "object",
                    "properties": {
                        "active_project": {"type": "boolean"},
                        "previous_list": {"type": "boolean"},
                        "comparison": {"type": "boolean"},
                        "ordinal": {"type": ["string", "null"]},
                    },
                },
                "confidence": {
                    "type": "number",
                    "description": "0-1. Lower when the question is vague, uses an entity name "
                                    "you're not sure about, or could plausibly mean two things.",
                },
                "requires_clarification": {"type": "boolean"},
            },
            "required": ["intent", "scope", "entities", "operations", "references", "confidence"],
        },
    },
}

_SYSTEM_PROMPT = """أنت طبقة الفهم الدلالي لمنصة إدارة محفظة مشاريع تنفيذية.
مهمتك: تصنيف معنى السؤال فقط -- لا تجيب عليه، لا تحسب، لا تستعلم قاعدة البيانات، ولا تختار سجلاً نهائياً.

قواعد الكيانات (entities) -- إلزامية:
- استخرج كل كيان كما كتبه المستخدم حرفياً (نص عربي أو إنجليزي كما ورد).
- ممنوع منعاً باتاً إخراج قيمة قاعدة بيانات (مثل "BPO-Specialized Pr")، اسم عمود (مثل "profit_pct")،
  أو ترجمة/قيمة محلولة -- فقط الكلمات التي استخدمها المستخدم فعلياً.
- إذا لم يُذكر كيان من نوع معين، اترك قائمته فارغة -- لا تخترع.

تمييز نوع الكيان التنظيمي -- هذا هو مصدر الخطأ الأكثر شيوعاً، انتبه له:
- "departments": أي وحدة تنظيمية/إدارية يُشار لها بصيغة "إدارة X" أو "قسم X" أو باسمها المختصر
  (مثل: إدارة التفتيش، إدارة الصحة، مراكز الأعمال، إسناد العمليات، المشاريع المتخصصة). هذا هو
  التصنيف الافتراضي لأي اسم إدارة/قسم/قطاع تنظيمي في هذه المنصة.
- "business_units" و"segments": نادراً ما تُستخدم في هذه المنصة -- ضعها هنا فقط إذا قال المستخدم
  حرفياً "وحدة الأعمال" أو "القطاع" كمصطلح مالي/تشغيلي منفصل عن الإدارة، وليس كمرادف للإدارة.
  عند الشك بين "إدارة" و"وحدة أعمال/قطاع"، اختر "departments" دائماً.

تمييز الترتيب (portfolio_ranking) عن التجميع (grouped_analytics) -- مصدر خطأ شائع آخر:
- "portfolio_ranking": ترتيب مشاريع فردية حسب مقياس، والإجابة مشروع واحد بعينه
  (مثل: "أكبر مشروع؟"، "أعلى مشروع ربحية؟"، "أصغر مشروع؟").
- "grouped_analytics": تجميع/عد المشاريع حسب فئة (إدارة/مدير/حالة...)، والإجابة اسم الفئة
  الفائزة (إدارة أو مدير)، وليس مشروعاً بعينه. استخدمه دائماً عندما يُسأل عن "أكثر مدير"،
  "أعلى إدارة"، "كم مشروع عند كل مدير"، أو أي سؤال شكله "مين/وش أكثر [فئة تنظيمية] ...؟" --
  ضع الفئة في operations.group_by (project_manager/dept/bu/segment/status)، والمقياس
  (إن وجد) في entities.metrics.
  أمثلة: "مين أكثر مدير ماسك مشاريع؟" -> grouped_analytics, group_by=project_manager.
          "وش أعلى إدارة من ناحية الإيرادات؟" -> grouped_analytics, group_by=dept, metrics=["الإيرادات"].

قواعد الثقة (confidence):
- 0.85 فأعلى: السؤال واضح تماماً، لا لبس.
- 0.6-0.85: السؤال مفهوم لكن قد يحتاج تأكيد كيان واحد (اسم قسم غير مألوف، مثلاً).
- أقل من 0.6: السؤال غامض أو قد يعني أكثر من شيء.

السياق المتاح لك: آخر مشروع/قائمة/مقارنة في المحادثة (إن وجدت) -- استخدمه فقط لتحديد
references (active_project / previous_list / comparison)، لا لتخمين كيان غير مذكور.

أجب فقط باستدعاء الأداة interpret_query."""


def _call_llm(query: str, ctx: dict) -> dict | None:
    try:
        ctx_summary = {
            "last_project": ctx.get("last_project_display_name") or ctx.get("last_project_code"),
            "last_result_scope": ctx.get("last_result_scope") or ctx.get("last_scope"),
            "last_list_intent": ctx.get("last_list_intent"),
            "has_previous_list": bool(ctx.get("last_project_list") or ctx.get("last_list_project_codes")),
            "has_comparison": bool(ctx.get("last_compared_project_ids") or ctx.get("comparison_project_ids")),
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {json.dumps(ctx_summary, ensure_ascii=False)}\nQuestion: {query}"},
        ]
        resp = _get_openai().chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=[_INTERPRET_TOOL],
            tool_choice={"type": "function", "function": {"name": "interpret_query"}},
        )
        tool_calls = resp.choices[0].message.tool_calls
        if not tool_calls:
            return None
        return json.loads(tool_calls[0].function.arguments)
    except Exception as exc:
        logger.warning("semantic_interpreter LLM call failed: %s | query=%.60s", exc, query)
        return None


# ── Layer 2 merge: resolve raw mentions before any caller sees them ─────

def _resolve_entities(
    raw_entities: dict, projects: list[dict],
) -> tuple[EntityMentions, list[UnresolvedEntity]]:
    unresolved: list[UnresolvedEntity] = []

    def _resolve_list(mentions: list[str], resolver, kind: str) -> list[str]:
        resolved: list[str] = []
        for mention in mentions:
            result = resolver(mention)
            if result.status in {"exact", "fuzzy"} and result.value:
                if result.value not in resolved:
                    resolved.append(result.value)
            else:
                unresolved.append(UnresolvedEntity(kind=kind, mention=mention, candidates=list(result.candidates)))
        return resolved

    status = _resolve_list(raw_entities.get("status") or [], lambda m: resolve_status(m), "status")
    departments = _resolve_list(
        raw_entities.get("departments") or [], lambda m: resolve_department(m, projects), "department",
    )
    business_units = _resolve_list(
        raw_entities.get("business_units") or [], lambda m: resolve_bu(m, projects), "bu",
    )
    segments = _resolve_list(
        raw_entities.get("segments") or [], lambda m: resolve_segment(m, projects), "segment",
    )
    programs = _resolve_list(
        raw_entities.get("programs") or [], lambda m: resolve_program(m, projects), "program",
    )
    managers = _resolve_list(
        raw_entities.get("managers") or [], lambda m: resolve_manager(m, projects), "manager",
    )

    metrics: list[str] = []
    for mention in raw_entities.get("metrics") or []:
        field_def = detect_requested_field(mention)
        if field_def:
            metrics.append(field_def.canonical)

    entities = EntityMentions(
        projects=list(raw_entities.get("projects") or []),  # resolved downstream by project_entity_resolver
        departments=departments,
        status=status,
        programs=programs,
        business_units=business_units,
        segments=segments,
        managers=managers,
        metrics=metrics,
        dates=list(raw_entities.get("dates") or []),
    )
    return entities, unresolved


def _clarification_question(unresolved: UnresolvedEntity) -> str:
    labels = {
        "status": "الحالة", "department": "الإدارة", "bu": "وحدة الأعمال",
        "segment": "القطاع", "program": "البرنامج", "manager": "مدير المشروع",
    }
    label = labels.get(unresolved.kind, unresolved.kind)
    if unresolved.candidates:
        options = "، ".join(f"«{c}»" for c in unresolved.candidates[:5])
        return f"ما قصدك بالضبط بـ«{unresolved.mention}» في {label}؟ يمكن تقصد: {options}."
    return f"ما قدرت أتأكد من {label} «{unresolved.mention}» -- تقدر توضحها أكثر؟"


# ── Public entry point ───────────────────────────────────────────────────

def interpret(query: str, ctx: dict | None = None, projects: list[dict] | None = None) -> Interpretation:
    """Turn one user message into a validated Interpretation. Never raises
    for an LLM failure/timeout/malformed response -- returns a
    CONTROLLED_FALLBACK Interpretation instead, so callers can fall back to
    the existing deterministic pipeline without a try/except of their own
    around this specific failure mode (genuinely unexpected bugs still
    propagate, same as everywhere else in this codebase)."""
    ctx = ctx or {}

    pre_pass = _deterministic_pre_pass(query)
    if pre_pass is not None:
        return pre_pass

    raw = _call_llm(query, ctx)
    if raw is None:
        return Interpretation(
            intent=CONTROLLED_FALLBACK, scope="unknown",
            entities=EntityMentions(), operations=Operations(), references=References(),
            confidence=0.0, method="llm_error",
        )

    intent = raw.get("intent")
    if intent not in ALL_INTENTS:
        intent = CONTROLLED_FALLBACK
    scope = raw.get("scope") if raw.get("scope") in {"portfolio", "project", "list", "unknown"} else "unknown"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if projects is None:
        from modules.project_repository import fetch_enriched_projects
        projects = fetch_enriched_projects()

    raw_entities = raw.get("entities") or {}
    entities, unresolved = _resolve_entities(raw_entities, projects)

    # Defense in depth: re-validate against the whitelist in Python rather
    # than trusting the API to have honored the JSON-schema enum exactly.
    raw_operations = raw.get("operations") or {}
    raw_group_by = raw_operations.get("group_by")
    raw_sort = raw_operations.get("sort")
    raw_limit = raw_operations.get("limit")
    operations = Operations(
        filter=bool(raw_operations.get("filter")),
        group_by=raw_group_by if raw_group_by in GROUP_FIELDS else None,
        sort=raw_sort if raw_sort in {"asc", "desc"} else None,
        limit=raw_limit if isinstance(raw_limit, int) and raw_limit > 0 else None,
        compare=bool(raw_operations.get("compare")),
    )
    raw_references = raw.get("references") or {}
    references = References(
        active_project=bool(raw_references.get("active_project")),
        previous_list=bool(raw_references.get("previous_list")),
        comparison=bool(raw_references.get("comparison")),
        ordinal=raw_references.get("ordinal"),
    )

    base = Interpretation(
        intent=intent, scope=scope, entities=entities, operations=operations,
        references=references, confidence=confidence, method="llm",
    )

    # Confidence gate.
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        target = unresolved[0] if unresolved else None
        question = _clarification_question(target) if target else (
            "ما قدرت أفهم طلبك بالضبط -- تقدر توضح أكثر؟"
        )
        base.requires_clarification = True
        base.clarification_question = question
        base.unresolved_entity = target
        base.method = "llm_low_confidence"
        return base

    if confidence < HIGH_CONFIDENCE_THRESHOLD and unresolved:
        target = unresolved[0]
        base.requires_clarification = True
        base.clarification_question = _clarification_question(target)
        base.unresolved_entity = target
        base.method = "llm_low_confidence"
        return base

    # High confidence: an unresolved entity is still surfaced as a focused
    # clarification (never silently dropped/guessed), even though the
    # overall question was clear.
    if unresolved:
        target = unresolved[0]
        base.requires_clarification = True
        base.clarification_question = _clarification_question(target)
        base.unresolved_entity = target
        return base

    base.requires_clarification = bool(raw.get("requires_clarification"))
    if base.requires_clarification and not base.clarification_question:
        base.clarification_question = "تقدر توضح طلبك أكثر؟"
    return base
