"""
modules/response_composer.py  — v2
Conversational Response Composer.

Receives ONLY verified, pre-computed data and produces executive-quality Arabic.

Changes from v1:
- _build_verified_summary now uses Arabic labels + pre-formatted values (not raw field names)
- compose_project_response sends a structured natural-language brief, not a Python dict
- compose_assessment_response distinguishes "وش رأيك" (balanced opinion) vs "هل يحتاج متابعة" (yes/no)
- Template fallback no longer exposes internal field names
- Type annotation fix: AzureOpenAI
"""

from __future__ import annotations

import logging
from typing import Any

import config
from modules.semantic_dictionary import normalize_text

logger = logging.getLogger(__name__)
_client = None


def _get_openai():
    global _client
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _client


# ── Value formatters ──────────────────────────────────────────────────────────

def _money(value: Any, arabic: bool = True) -> str:
    if value is None:
        return "غير متوفر" if arabic else "N/A"
    amount = float(value)
    absolute = abs(amount)
    sign = "" if amount >= 0 else "-"
    if absolute >= 1_000_000_000:
        n = f"{absolute/1_000_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{n} مليار ريال" if arabic else f"{sign}SAR {n}B"
    if absolute >= 1_000_000:
        n = f"{absolute/1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{sign}{n} مليون ريال" if arabic else f"{sign}SAR {n}M"
    return f"{sign}{absolute:,.0f} ريال" if arabic else f"{sign}SAR {absolute:,.0f}"


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}%"


def _status_ar(status: str | None) -> str:
    return {
        "Ongoing": "جاري", "Completed": "مكتمل", "Closed": "مغلق",
        "On Hold": "متوقف مؤقتًا", "On-hold": "متوقف مؤقتًا",
        "Pipeline": "قيد الإعداد",
    }.get(status or "", status or "غير محدد")


# ── Verified data brief builder ───────────────────────────────────────────────

def _build_brief(project: dict, field: str | None) -> str:
    """
    Build a clean Arabic-labeled data brief for the LLM.
    NEVER exposes internal field names (profit_pct, total_cost, etc.).
    Values are pre-formatted (currency, percentage, date).
    """
    name = project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code")
    lines = [f"المشروع: {name}"]

    # Field label map: canonical → (Arabic label, formatted value)
    def _fmt(canonical: str) -> str | None:
        v = project.get(canonical)
        if v is None:
            return None
        if canonical in ("total_contract_value","total_revenue","total_cost","net_profit","backlog","risk","pl"):
            return _money(v)
        if canonical in ("profit_pct",):
            return _pct(v)
        if canonical in ("start_date",):
            return str(v)
        if canonical in ("customer_id", "officer_name", "note"):
            return str(v)
        if canonical == "progress_completed":
            return f"{float(v):.1f}%"
        if canonical == "status":
            return _status_ar(str(v))  # always translate to Arabic
        if canonical == "days_remaining":
            d = int(float(v))
            if d < 0:
                return f"تجاوز تاريخ الانتهاء بـ{abs(d)} يوم"
            return f"{d} يوم متبقي"
        return str(v)

    _LABEL_MAP = {
        "status":               "الحالة",
        "progress_completed":   "التقدم",
        "project_manager":      "المدير",
        "total_contract_value": "قيمة العقد",
        "total_revenue":        "الإيرادات",
        "total_cost":           "التكاليف",
        "net_profit":           "صافي الربح",
        "profit_pct":           "هامش الربح",
        "backlog":              "Backlog",
        "effective_end_date":   "تاريخ الانتهاء",
        "start_date":           "تاريخ البداية",
        "days_remaining":       "الوقت المتبقي",
        "risk":                 "قيمة المخاطر",
        "variance":             "الانحراف",
        "customer_id":          "العميل",
        "officer_name":         "المسؤول التنفيذي",
        "note":                 "الملاحظات",
        "amendment_crs":        "التعديلات",
    }

    if field:
        # Single field mode — only send the requested field
        label = _LABEL_MAP.get(field, field)
        v = project.get(field)
        formatted = _fmt(field) if v is not None else "غير متوفر"
        lines.append(f"{label}: {formatted}")
    else:
        # Full brief — all available fields with Arabic labels
        for canonical, label in _LABEL_MAP.items():
            formatted = _fmt(canonical)
            if formatted is not None:
                lines.append(f"{label}: {formatted}")

    return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

_BASE_SYSTEM = """أنت Project Intelligence AI — مساعد تنفيذي ذكي متخصص في تحليل محفظة المشاريع لشركة Elm.
لا تجب كقاعدة بيانات ولا كـ SQL Engine. أجب كخبير PMO يقدم ملخصات وتحليلات وتوصيات للإدارة العليا.

قواعد مطلقة — لا تُكسر أبداً:
١. لا تخترع أي رقم — استخدم فقط ما في VERIFIED_DATA.
٢. لا تجري حسابات — الأرقام جاهزة ومحققة.
٣. لا تذكر أسماء الأعمدة أبداً (ممنوع: total_cost, net_profit, profit_pct, etc).
٤. لا تقل "وجدت صفاً" أو "القيمة الموجودة" أو "وفق البيانات".
٥. لا تختم بـ "إذا كان لديك استفسارات" أو ما شابه — أنهِ بهدوء.
٦. لا تكرر السؤال في الجواب.

تنسيق الأرقام — إلزامي:
- الأموال: 215.86 مليون ريال | 1.12 مليار ريال (لا تكتب: 215860000)
- النسب: 45.4% (لا تكتب: 45.44 بالمئة)
- التواريخ: 17 أبريل 2029 (لا تكتب: 2029-04-17)
- الحالات: جاري التنفيذ | مكتمل | مغلق | ملغي | متوقف (لا تكتب: Ongoing | Completed)

أسماء الأشخاص:
- إذا كان الاسم بالإنجليزي (Badr Mohammed A Alshahrani) → حوّله للعربية (بدر محمد آل شهراني).

أسلوب الرد:
- السؤال المباشر عن قيمة ⇒ أجب بالقيمة + شرح مختصر في جملة.
- السؤال العام (وش وضعه؟ / يحتاج متابعة؟) ⇒ حلّل الحالة + الإنجاز + الربحية + المخاطر معاً واستنتج.
- القوائم ⇒ لا تقل "وجدت X مشروعاً مطابقاً"، قل "يوجد X مشروع..." وأضف توصية مختصرة.
- المتابعة (وتكاليفه؟ / وربحه؟) ⇒ جملة واحدة أو اثنتان، مباشر بدون مقدمة.

إذا كان السؤال متابعة لمشروع سبق ذكره:
- لا تطلب اسم المشروع مرة ثانية.
- أجب عن نفس المشروع مباشرة.

إذا لم يُذكر مشروع في السياق وكان السؤال يحتاج مشروعاً:
- اسأل: "تقصد أي مشروع؟" ولا تخمن.
"""

_FOLLOWUP_SUFFIX = """
هذا سؤال متابعة. المستخدم يعرف المشروع — لا تعيد تقديمه.
أجب بجملة واحدة مباشرة. مثال: "بلغت التكاليف 259.14 مليون ريال." """

_ASSESSMENT_SUFFIX = """
هذا سؤال تقييمي تنفيذي. حلّل هذه العوامل معاً: الحالة + نسبة الإنجاز + الربحية + المخاطر + الجدول الزمني.
الهيكل:
- الوضع العام (جملة واحدة)
- أبرز نقطة إيجابية (إن وجدت في البيانات)
- أبرز نقطة تستحق الانتباه (إن وجدت في البيانات)
- تقييم ختامي مختصر
لا تخترع معلومات غير موجودة في VERIFIED_DATA."""

_RECOMMENDATION_SUFFIX = """
سؤال تنفيذي محدد: هل يحتاج هذا المشروع متابعة أو تدخلاً؟
أجب بوضوح: نعم أو لا — ثم اذكر السبب الرئيسي الموثق من البيانات فقط.
إذا كانت جميع المؤشرات إيجابية، قل ذلك بثقة.
لا تكتب "قد" أو "ربما" إذا كانت البيانات واضحة."""


# ── Public API ────────────────────────────────────────────────────────────────

def compose_project_response(
    project: dict,
    query: str,
    ctx: dict,
    field: str | None = None,
    intent: str = "field_lookup",
) -> str:
    """Compose a natural project response using Azure OpenAI."""
    if field:
        from modules.response_formatter import format_project_metrics
        return format_project_metrics(project, [field])
    arabic = any("\u0600" <= c <= "\u06ff" for c in query)
    depth  = ctx.get("active_project_depth", 1)
    answered = ctx.get("last_answered_fields") or []
    is_followup = depth > 1 or bool(answered) or intent in ("field_lookup", "general_followup")

    brief = _build_brief(project, field)

    # Context hint — natural language, not Python dict
    ctx_lines = []
    if is_followup:
        ctx_lines.append("هذا سؤال متابعة في نفس المحادثة.")
    if answered:
        ctx_lines.append(f"سبق الإجابة عن: {', '.join(answered[-4:])} — لا تكررها.")
    if ctx.get("conversation_phase") == "assessment":
        ctx_lines.append("المستخدم في مرحلة التقييم.")

    system = _BASE_SYSTEM
    if is_followup and not ctx_lines:
        system += _FOLLOWUP_SUFFIX
    elif ctx_lines:
        system += "\n\nملاحظات:\n" + "\n".join(f"- {l}" for l in ctx_lines)

    user_content = f"السؤال: {query}\n\nVERIFIED_DATA:\n{brief}"

    try:
        resp = _get_openai().chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Composer LLM failed: %s", exc)
        return _template_fallback(project, field, arabic, depth)


def compose_assessment_response(
    project: dict,
    query: str,
    ctx: dict,
    attention_data: dict | None = None,
) -> str:
    """
    Compose an executive assessment.
    Distinguishes:
    - "وش رأيك فيه" → balanced opinion (ASSESSMENT_SUFFIX)
    - "هل يحتاج متابعة" → yes/no recommendation (RECOMMENDATION_SUFFIX)
    """
    arabic = any("\u0600" <= c <= "\u06ff" for c in query)

    # Build verified facts list
    net   = project.get("net_profit")
    margin = project.get("profit_pct")
    days  = project.get("days_remaining")
    risk  = project.get("risk")
    status = project.get("status")

    facts: list[str] = []
    if net is not None and float(net) < 0:
        facts.append(f"خسارة صافية: {_money(net)}")
    elif net is not None and float(net) > 0:
        facts.append(f"ربح صافي: {_money(net)}")
    if margin is not None:
        facts.append(f"هامش الربح: {_pct(margin)}")
    if days is not None:
        d = int(float(days))
        if d < 0 and status not in {"Completed","Closed"}:
            facts.append(f"تجاوز تاريخ الانتهاء بـ{abs(d)} يوم — ما زال مفتوحاً")
        elif 0 <= d <= 30 and status not in {"Completed","Closed"}:
            facts.append(f"ينتهي خلال {d} يوماً")
        elif d > 0:
            facts.append(f"متبقي {d} يوم على الانتهاء")
    if risk is not None and float(risk) > 0:
        facts.append(f"مخاطر مسجلة: {_money(risk)}")
    if project.get("progress_completed") is not None:
        facts.append(f"نسبة الإنجاز: {float(project['progress_completed']):.1f}%")
    if attention_data:
        score = attention_data.get("attention_score", 0)
        if score > 0:
            facts.append(f"درجة الأولوية التنفيذية: {score}/10")

    facts_text = "\n".join(f"• {f}" for f in facts) if facts else "• لا توجد مؤشرات سلبية في البيانات"

    # Detect question type
    q_norm = normalize_text(query)
    _RECOMMENDATION_TRIGGERS = tuple(normalize_text(v) for v in (
        "هل يحتاج متابعة", "هل يحتاج", "يحتاج تدخل", "يستاهل انتباه",
        "needs attention", "is it on track",
    ))
    is_recommendation = any(t in q_norm for t in _RECOMMENDATION_TRIGGERS)

    system = _BASE_SYSTEM + (_RECOMMENDATION_SUFFIX if is_recommendation else _ASSESSMENT_SUFFIX)

    brief = _build_brief(project, None)
    user_content = (
        f"السؤال: {query}\n\n"
        f"الحقائق الموثقة:\n{facts_text}\n\n"
        f"VERIFIED_DATA:\n{brief}"
    )

    try:
        resp = _get_openai().chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.15,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Assessment composer failed: %s", exc)
        name = project.get("project_name_ar") or project.get("project_code")
        return _template_assessment(name, facts, arabic)


# ── Template fallbacks (no LLM) ───────────────────────────────────────────────

def _template_fallback(project: dict, field: str | None, arabic: bool, depth: int) -> str:
    """Clean template fallback — NEVER exposes internal field names."""
    name = project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code")

    if field:
        # Complete field label + formatter map
        _MONEY_FIELDS = {"total_cost","total_revenue","net_profit","backlog",
                         "total_contract_value","pl","risk","contingency",
                         "pm_up_to_2025","gp_2026","pm_2026","variance","etc_cost","etc_revenue"}
        _PCT_FIELDS   = {"profit_pct","pm_pct_up_to_2025","pm_pct_2026","planned_pm_pct","etc_pct"}
        _FIELD_LABELS = {
            "total_cost":           "التكاليف",
            "total_revenue":        "الإيرادات",
            "net_profit":           "صافي الربح",
            "pl":                   "صافي الربح والخسارة",
            "profit_pct":           "هامش الربح",
            "total_contract_value": "قيمة العقد",
            "backlog":              "الـ Backlog",
            "progress_completed":   "نسبة الإنجاز",
            "status":               "الحالة",
            "effective_end_date":   "تاريخ الانتهاء",
            "start_date":           "تاريخ البداية",
            "days_remaining":       "الوقت المتبقي",
            "project_manager":      "مدير المشروع",
            "risk":                 "قيمة المخاطر",
            "contingency":          "الاحتياطي",
            "customer_id":          "العميل / الجهة",
            "officer_name":         "المسؤول التنفيذي",
            "note":                 "الملاحظات",
            "amendment_crs":        "التعديلات",
            "variance":             "الانحراف",
            "planned_pm_pct":       "الهامش المخطط",
            "etc_cost":             "التكلفة المتوقعة",
            "etc_revenue":          "الإيراد المتوقع",
            "pm_pct_up_to_2025":    "الهامش حتى 2025",
            "pm_pct_2026":          "هامش 2026",
        }
        label = _FIELD_LABELS.get(field, field)
        v = project.get(field)
        if v is None:
            return f"لا تتوفر بيانات {label} لهذا المشروع."
        # Format value
        if field in _MONEY_FIELDS and v is not None:
            try:
                val_str = _money(float(v))
            except (TypeError, ValueError):
                val_str = str(v)
        elif field in _PCT_FIELDS and v is not None:
            try:
                val_str = _pct(float(v))
            except (TypeError, ValueError):
                val_str = str(v)
        elif field == "progress_completed" and v is not None:
            try:
                val_str = f"{float(v):.1f}%"
            except (TypeError, ValueError):
                val_str = str(v)
        elif field == "status":
            val_str = _status_ar(str(v))
        elif field == "days_remaining" and v is not None:
            try:
                d = int(float(v))
                val_str = f"تجاوز تاريخ الانتهاء بـ{abs(d)} يوم" if d < 0 else f"{d} يوم متبقٍ"
            except (TypeError, ValueError):
                val_str = str(v)
        else:
            val_str = str(v)
        # Natural response based on depth
        if depth > 1:
            # Followup: just the value, very short
            return val_str
        # First time asking about this field
        return f"{label}: {val_str}."

    # Full summary
    status = _status_ar(project.get("status"))
    parts = [f"مشروع «{name}» {status}."]
    if project.get("progress_completed") is not None:
        parts.append(f"التقدم {float(project['progress_completed']):.1f}%.")
    if project.get("project_manager"):
        parts.append(f"يديره {project['project_manager']}.")
    if project.get("total_contract_value") is not None:
        parts.append(f"قيمة العقد {_money(project['total_contract_value'])}.")
    net = project.get("net_profit")
    if net is not None:
        label = "صافي الربح" if float(net) >= 0 else "صافي الخسارة"
        parts.append(f"{label} {_money(abs(float(net)))}.")
    if project.get("effective_end_date"):
        from modules.response_formatter import format_arabic_date
        parts.append(f"ينتهي {format_arabic_date(project['effective_end_date'])}.")
    return " ".join(parts)


def _template_assessment(name: str, facts: list[str], arabic: bool) -> str:
    if not facts or all("ربح صافي" in f or "هامش" in f for f in facts):
        return f"المشروع «{name}» يسير بشكل معقول وفق البيانات المتاحة — لا توجد مؤشرات تستدعي التدخل."
    concerns = [f for f in facts if any(w in f for w in ("خسارة","تجاوز","مخاطر"))]
    if concerns:
        return f"مشروع «{name}» يستدعي المتابعة: {' — '.join(concerns)}."
    return f"مشروع «{name}»: {' | '.join(facts)}."
