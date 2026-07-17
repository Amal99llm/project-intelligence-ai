"""
modules/response_formatter.py
-------------------------------
Section 17 Step 4 — LLM Formatting Only.

The only remaining "creative" LLM call in the structured pipeline. It is
given already-verified data (see modules.verification) and is only allowed
to explain, summarize, translate or improve the wording of that data.
Section 20 — Prevent LLM Math: the system prompt explicitly forbids
arithmetic and forbids stating any number that isn't present in the input,
so the model has no way to produce a number that wasn't already verified.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime

from openai import AzureOpenAI

import config
from modules.project_entity_resolver import normalize_project_text
from modules.semantic_dictionary import FIELDS, detect_requested_field
from modules.verified_response import project_field_payload

logger = logging.getLogger(__name__)

_AR_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
              "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")


def format_arabic_date(value) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value[:10])
        except ValueError:
            return value
    if not isinstance(value, date):
        return str(value)
    return f"{value.day} {_AR_MONTHS[value.month - 1]} {value.year}"


METRIC_RESPONSE_TEMPLATES = {
    "total_cost": "بلغت تكاليف المشروع {value}.",
    "pl": "يحقق المشروع صافي ربح قدره {value}.",
    "net_profit": "يحقق المشروع صافي ربح قدره {value}.",
    "profit_pct": "هامش الربح الحالي يبلغ {value}.",
    "risk": "تبلغ قيمة المخاطر المسجلة {value}.",
    "total_revenue": "بلغت إيرادات المشروع {value}.",
    "backlog": "تبلغ الأعمال المتبقية في المشروع {value}.",
    "contract_value": "قيمة العقد الأساسية {value}.",
    "amendment_crs": "تبلغ قيمة تعديلات العقد {value}.",
    "total_contract_value": "قيمة العقد الإجمالية بعد التعديلات {value}.",
    "project_manager": "مدير المشروع هو {value}.",
    "status": "حالة المشروع هي {value}.",
    "progress_completed": "بلغت نسبة إنجاز المشروع {value}.",
    "start_date": "بدأ المشروع في {value}.",
    "effective_end_date": "من المخطط أن ينتهي المشروع في {value}.",
}


def _metric_value(project: dict, metric: str) -> str:
    value = project.get(metric)
    definition = FIELDS.get(metric)
    if value is None:
        return "غير متوفر"
    if definition and definition.data_type == "money" or metric in {"pl", "net_profit", "risk"}:
        return _money(value)
    if definition and definition.data_type == "percentage":
        return f"{float(value):.2f}".rstrip("0").rstrip(".") + "%"
    if definition and definition.data_type == "date" or metric in {"start_date", "effective_end_date"}:
        return format_arabic_date(value)
    if metric == "status":
        return {"Ongoing": "جاري التنفيذ", "Completed": "مكتمل", "Closed": "مغلق",
                "On-hold": "متوقف مؤقتًا", "On Hold": "متوقف مؤقتًا"}.get(str(value), str(value))
    return str(value)


def format_project_metrics(project: dict, metrics: list[str]) -> str:
    metrics = list(dict.fromkeys(m for m in metrics if m in FIELDS))
    if metrics == ["start_date", "effective_end_date"] or set(metrics) == {"start_date", "effective_end_date"}:
        return (f"بدأ المشروع في {_metric_value(project, 'start_date')}، ومن المخطط أن ينتهي في "
                f"{_metric_value(project, 'effective_end_date')}.")
    sentences = []
    for metric in metrics:
        value = _metric_value(project, metric)
        template = METRIC_RESPONSE_TEMPLATES.get(metric)
        if template:
            sentences.append(template.format(value=value))
        else:
            sentences.append(f"{FIELDS[metric].label_ar}: {value}.")
    return " ".join(sentences)


def format_delay_status(project: dict) -> str:
    days = project.get("days_remaining")
    end = project.get("effective_end_date")
    status = str(project.get("status") or "")
    closed = status in {"Completed", "Closed", "Cancelled", "Canceled"}
    if days is not None and float(days) < 0 and not closed:
        return f"نعم، المشروع متأخر بـ {abs(int(float(days)))} يومًا وما زال مصنفًا {_metric_value(project, 'status')}."
    return f"لا، المشروع غير متأخر؛ موعد انتهائه في {format_arabic_date(end)}."

_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _openai_client


_SYSTEM_PROMPT = """أنت المستشار التنفيذي الذكي لمنصة Elm لإدارة محفظة المشاريع.
بياناتك محسوبة ومُحققة مسبقاً. مهمتك تحويلها لكلام طبيعي يبدو وكأنه من محلل بشري محترف يجلس مع VP.

قواعد مطلقة:
١. لا تحسب أي أرقام — كلها جاهزة في VERIFIED_DATA.
٢. لا تذكر أي رقم أو معلومة غير موجودة في VERIFIED_DATA.
٣. لا تخترع أو تقدّر أي شيء.
٤. إذا لم توجد بيانات: قل ذلك بوضوح بدون تخمين.
٥. لا تذكر قواعد البيانات أو JSON أو SQL أو الأعمدة أو "وفق البيانات المتاحة".
٦. الأرقام بالعربية الطبيعية: "259 مليون ريال" لا "259,144,698.68" أبداً.
٧. ابدأ بالجواب مباشرة — بدون مقدمات مثل "بناءً على البيانات" أو "وفقاً للمعلومات".
٨. لا تختم بعبارات مثل "إذا كان لديك استفسارات" أو "لا تتردد" — أنهِ الجواب بهدوء.
٩. تنوّع في الأسلوب — لا تبدأ كل جواب بنفس الجملة.
١٠. الأسماء تُكتب حرفياً كما في VERIFIED_DATA — ممنوع منعاً باتاً خلط حروف عربية وإنجليزية في اسم واحد.
    مثال صح: "بدر محمد آل شهراني" — مثال خطأ: "بدر محمد الشahrani"
١١. الأرقام تُكتب بالأرقام دائماً — ممنوع كتابتها بالحروف.
    صح: "215.86 مليون ريال" — خطأ: "مئتان وخمسة عشر مليون..."
    صح: "45.4%" — خطأ: "خمسة وأربعون فاصل أربعة..."
١٢. إذا كان السؤال متابعة (وتكاليفه، وربحه...) أجب بجملة واحدة أو اثنتين فقط.
١٣. للأسئلة التقييمية — استند للحقائق الموجودة فقط. إذا المشروع يسير بشكل جيد قل ذلك بوضوح.
١٤. اللغة حسب السؤال. العربية السعودية المحادثية مقبولة ومفضّلة.
"""


def compose_execution_results(results: list[dict]) -> str:
    """Compose already-verified task outcomes without adding facts."""
    usable = [item for item in results if item.get("answer")]
    if not usable:
        return "لا تتوفر نتائج موثقة كافية للإجابة."
    if len(usable) == 1:
        return usable[0]["answer"]
    sections = []
    for index, item in enumerate(usable, 1):
        sections.append(f"{index}. {item['answer']}")
    return "\n\n".join(sections)


def format_answer(query: str, verified_data: dict) -> str:
    """Verbalize verified values without any generative numeric surface.

    This intentionally does not call a language model: every number in the
    returned sentence is formatted directly from the verified payload.
    """
    arabic = _is_arabic(query)
    aggregation = verified_data.get("aggregation")
    value = verified_data.get("aggregation_result")
    rows = verified_data.get("rows") or []
    spec = verified_data.get("spec") or {}
    intent = verified_data.get("intent")
    matched_count = verified_data.get("matched_row_count", len(rows))

    if verified_data.get("portfolio_kpis") and not aggregation:
        return format_portfolio_summary(verified_data["portfolio_kpis"], arabic=arabic)

    if aggregation is not None:
        column = aggregation.get("column")
        func = aggregation.get("func")
        definition = FIELDS.get(column)
        label_ar = definition.label_ar if definition else {
            "net_profit": "صافي الربح", "profit_pct": "هامش الربح",
            "project_code": "عدد المشاريع",
        }.get(column, "النتيجة")
        label_en = definition.label_en if definition else {
            "net_profit": "Net profit", "profit_pct": "Profit margin",
            "project_code": "Project count",
        }.get(column, "Result")
        if value is None:
            return "لا توجد قيمة موثقة مطابقة للسؤال." if arabic else "No verified value matched the question."
        if func == "COUNT":
            count = int(value)
            status_filter = next((f for f in spec.get("filters", []) if f.get("column") == "status"), None)
            statuses = status_filter.get("value") if status_filter else None
            statuses = statuses if isinstance(statuses, list) else [statuses] if statuses else []
            if arabic and statuses == ["Ongoing"]:
                return f"يوجد {count} مشروعًا جاريًا."
            if arabic and set(statuses) == {"Completed", "Closed"}:
                return f"يوجد {count} مشروعًا مكتملًا."
            return (f"يوجد {count} مشروعًا في المحفظة." if arabic else f"The portfolio contains {count} projects.")
        if definition and definition.data_type == "money" or column == "net_profit":
            rendered = _money(value, arabic)
        elif definition and definition.data_type == "percentage" or column == "profit_pct":
            rendered = f"{float(value):.1f}%"
        else:
            rendered = f"{float(value):,.2f}" if isinstance(value, (int, float)) else str(value)
        return (f"{label_ar}: {rendered}." if arabic else f"{label_en}: {rendered}.")

    if not rows:
        from modules.semantic_dictionary import detect_semantic_intent as _dsi
        _sem = _dsi(query) if query else None
        _no_match_msgs = {
            "expiring": "ما في عقود تنتهي خلال 30 يوماً القادمة وفق البيانات الحالية.",
            "losing_projects": "ما في مشاريع بربحية سالبة حالياً.",
            "overdue": "ما في مشاريع تجاوزت تاريخ انتهائها.",
        }
        if arabic:
            _generic = [
                "ما وجدت مشاريع مطابقة — جرّب صياغة مختلفة أو أعطني اسم المشروع.",
                "ما قدرت أحدد المشروع — تقدر تعطيني اسمه أو رمزه؟",
                "ما عندي نتائج لهذا الطلب — حاول تعيد الصياغة.",
            ]
            import random
            return _no_match_msgs.get(_sem, random.choice(_generic))
        return "I found no matching projects in the current data."

    sort = spec.get("sort") or {}
    sort_column = sort.get("column")
    if len(rows) == 1:
        row = rows[0]
        name = row.get("project_name_ar") or row.get("project_name_en") or row.get("project_code")
        if sort_column in row and row.get(sort_column) is not None:
            definition = FIELDS.get(sort_column)
            raw = row[sort_column]
            rendered = (_money(raw, arabic) if (definition and definition.data_type == "money") or sort_column in {"net_profit", "pl"}
                        else f"{float(raw):.2f}%" if (definition and definition.data_type == "percentage") or sort_column == "profit_pct"
                        else str(raw))
            label = definition.label_ar if arabic and definition else definition.label_en if definition else sort_column
            if intent == "portfolio_ranking":
                direction = sort.get("direction")
                qualifier = "الأعلى" if direction == "DESC" else "الأقل"
                if sort_column == "backlog":
                    return f"أعلى مشروع من حيث الأعمال المتبقية هو «{name}»، بقيمة تبلغ {rendered}."
                return (f"المشروع {qualifier} حسب {label} هو «{name}» بقيمة {rendered}." if arabic
                        else f"The {qualifier} project by {label} is {name}, at {rendered}.")
            return (f"المشروع المطابق بالاسم هو «{name}»، و{label} {rendered}." if arabic
                    else f"The project matching that name is {name}, with {label} of {rendered}.")
        return format_project_summary(row, query=query)

    lines = []
    for index, row in enumerate(rows, 1):
        name = row.get("project_name_ar") or row.get("project_name_en") or row.get("project_code")
        suffix = ""
        if sort_column and row.get(sort_column) is not None:
            definition = FIELDS.get(sort_column)
            raw = row[sort_column]
            rendered = (_money(raw, arabic) if (definition and definition.data_type == "money") or sort_column in {"net_profit", "pl"}
                        else f"{float(raw):.2f}%" if (definition and definition.data_type == "percentage") or sort_column == "profit_pct"
                        else format_arabic_date(raw) if (definition and definition.data_type == "date") else str(raw))
            label = definition.label_ar if definition else sort_column
            suffix = f" — {label}: {rendered}"
        lines.append(f"{index}. {name}{suffix}")
    if intent == "portfolio_filter" and arabic:
        margin_filter = next((f for f in spec.get("filters", []) if f.get("column") == "profit_pct"), None)
        if margin_filter and margin_filter.get("op") == "<":
            intro = f"يوجد {matched_count} مشروعًا بهامش ربح أقل من {margin_filter['value']:g}%، وهذه أبرز {len(rows)} مشاريع:"
        else:
            intro = f"يوجد {matched_count} مشروعًا ضمن نتائج الفلترة، وهذه أبرز {len(rows)} مشاريع:"
    else:
        intro = f"وجدت {len(rows)} مشروعًا مطابقًا بالاسم:" if arabic else f"I found {len(rows)} projects matching that name:"
    return intro + "\n" + "\n".join(lines)


def format_portfolio_summary(kpis: dict, arabic: bool = True) -> str:
    """Compact executive summary built only from canonical KPI values."""
    if not arabic:
        return (
            f"The portfolio contains {int(kpis['total_projects'])} projects, including "
            f"{int(kpis['active_projects'])} active projects. Total revenue is "
            f"{_money(kpis['total_revenue'], False)}, net profit is {_money(kpis['net_profit'], False)}, "
            f"and backlog is {_money(kpis['backlog'], False)}."
        )
    return (
        f"تضم المحفظة {int(kpis['total_projects'])} مشروعًا، منها {int(kpis['active_projects'])} مشروعًا جاريًا. "
        f"بلغ إجمالي الإيرادات {_money(kpis['total_revenue'])}، وصافي الربح {_money(kpis['net_profit'])}، "
        f"فيما يبلغ رصيد الأعمال المتبقية {_money(kpis['backlog'])}."
    )


def build_source_attribution(spec: dict, aggregation_result, kpi_formula: str | None = None) -> dict:
    """Section 17 Step 5 -- deterministic source attribution, never
    produced by the LLM. Attached to the API response and audit log."""
    columns = {f["column"] for f in spec.get("filters", [])}
    if spec.get("aggregation") and spec["aggregation"].get("column"):
        columns.add(spec["aggregation"]["column"])
    return {
        "table": "backlog_projects",
        "columns": sorted(columns),
        "filters": spec.get("filters", []),
        "sort": spec.get("sort"),
        "limit": spec.get("limit"),
        "formula": kpi_formula,
        "aggregation_result": aggregation_result,
    }


_ARABIC_MONTHS = (
    "", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
)


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in str(text or ""))


def _status(project: dict, arabic: bool = True) -> str:
    raw = project.get("status") or ""
    if not arabic:
        return raw or "unavailable"
    return {
        "Ongoing": "جاري", "Completed": "مكتمل", "Closed": "مغلق",
        "On-hold": "متوقف مؤقتًا", "On Hold": "متوقف مؤقتًا", "Pipeline": "قيد الإعداد",
    }.get(raw, raw or "غير متوفر")


def _money(value, arabic: bool = True) -> str:
    if value is None:
        return "غير متوفر" if arabic else "unavailable"
    amount = float(value)
    absolute = abs(amount)
    if absolute >= 1_000_000_000:
        number = f"{amount / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{number} مليار ريال" if arabic else f"SAR {number}B"
    if absolute >= 1_000_000:
        number = f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{number} مليون ريال" if arabic else f"SAR {number}M"
    return f"{amount:,.2f} ريال" if arabic else f"SAR {amount:,.2f}"


def _date_text(value, arabic: bool = True) -> str:
    if not value:
        return "غير متوفر" if arabic else "unavailable"
    try:
        from datetime import date
        parsed = date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return str(value)
    if arabic:
        return f"{parsed.day} {_ARABIC_MONTHS[parsed.month]} {parsed.year}"
    return parsed.strftime("%B %d, %Y")


def project_followup_topic(query: str) -> str | None:
    """Return a canonical field from the centralized semantic dictionary."""
    field = detect_requested_field(query)
    return field.canonical if field else None


def format_project_followup(project: dict, topic: str, query: str = "") -> str:
    legacy = {
        "manager": "project_manager", "end_date": "effective_end_date",
        "revenue": "total_revenue", "cost": "total_cost", "profit": "pl",
        "contract": "total_contract_value", "progress": "progress_completed",
    }
    topic = legacy.get(topic, topic)
    arabic = _is_arabic(query) if query else True
    name = project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code")
    if topic not in FIELDS:
        return format_project_summary(project, query=query)
    payload = project_field_payload(project, [topic])
    value = payload.verified_values[topic]
    definition = FIELDS[topic]
    if value is None or (isinstance(value, str) and not value.strip()):
        return (f"لا تتوفر قيمة موثقة لـ {definition.label_ar} في مشروع «{name}»." if arabic
                else f"No verified {definition.label_en.lower()} is available for {name}.")
    if topic == "status":
        return (f"حالة مشروع «{name}» هي {_status(project)}." if arabic
                else f"{name} is {_status(project, False)}.")
    if topic == "profit_pct" and arabic and "ربحي" in normalize_project_text(query):
        net = project.get("net_profit")
        if net is not None:
            result = "صافي ربح" if float(net) >= 0 else "صافي خسارة"
            return f"حقق مشروع «{name}» {result} قدره {_money(abs(float(net)))}، بهامش {float(value):.1f}%."
    if definition.data_type == "money":
        if topic == "total_contract_value" and arabic:
            return f"قيمة عقد مشروع «{name}» هي {_money(value)}."
        if topic == "total_cost" and arabic:
            return f"إجمالي التكاليف لمشروع «{name}» {_money(value)}."
        if topic == "total_revenue" and arabic:
            return f"إيرادات مشروع «{name}» تبلغ {_money(value)}."
        if topic in ("net_profit", "pl") and arabic:
            net = float(value)
            label = "صافي الربح" if net >= 0 else "صافي الخسارة"
            return f"{label} لمشروع «{name}» {_money(abs(net))}."
        if topic == "backlog" and arabic:
            return f"الـ Backlog المتبقي لمشروع «{name}» {_money(value)}."
        return (f"{definition.label_ar} لمشروع «{name}» {_money(value)}." if arabic
                else f"{definition.label_en} for {name}: {_money(value, False)}.")
    if definition.data_type == "percentage":
        rendered = f"{float(value):.1f}%"
    elif definition.data_type == "date":
        rendered = _date_text(value, arabic)
    elif definition.data_type == "days":
        rendered = f"{int(value)} يومًا" if arabic else f"{int(value)} days"
    else:
        rendered = str(value)
    return (f"{definition.label_ar} لمشروع «{name}»: {rendered}." if arabic
            else f"{definition.label_en} for {name}: {rendered}.")


def format_project_summary(project: dict, query: str = "") -> str:
    """Natural executive summary; detailed fields only when explicitly requested."""
    arabic = _is_arabic(query) if query else True
    q = normalize_project_text(query)
    detailed = any(marker in q for marker in ("كل التفاصيل", "تفاصيل كامله", "جدول", "full details", "all details"))
    name = project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code")
    manager = project.get("project_manager")
    progress = project.get("progress_completed")
    net = project.get("net_profit")

    if not arabic:
        parts = [f"{name} is currently {_status(project, False)}."]
        if progress is not None: parts.append(f"Completion is {float(progress):.1f}%.")
        if manager: parts.append(f"It is managed by {manager}.")
        if project.get("total_contract_value") is not None: parts.append(f"The contract value is {_money(project['total_contract_value'], False)}.")
        if project.get("total_revenue") is not None: parts.append(f"Total revenue is {_money(project['total_revenue'], False)}.")
        if project.get("total_cost") is not None: parts.append(f"Total cost is {_money(project['total_cost'], False)}.")
        if net is not None: parts.append(f"Net {'profit' if float(net) >= 0 else 'loss'} is {_money(abs(float(net)), False)}.")
        if project.get("effective_end_date"): parts.append(f"The expected completion date is {_date_text(project['effective_end_date'], False)}.")
        summary = " ".join(parts)
    else:
        parts = [f"مشروع «{name}» حاليًا {_status(project)}."]
        if progress is not None: parts.append(f"بلغت نسبة الإنجاز {float(progress):.1f}%.")
        if manager: parts.append(f"يديره {manager}.")
        if project.get("total_contract_value") is not None: parts.append(f"تبلغ قيمة العقد {_money(project['total_contract_value'])}.")
        if project.get("total_revenue") is not None: parts.append(f"إجمالي الإيرادات {_money(project['total_revenue'])}.")
        if project.get("total_cost") is not None: parts.append(f"إجمالي التكاليف {_money(project['total_cost'])}.")
        if net is not None: parts.append(f"صافي {'الربح' if float(net) >= 0 else 'الخسارة'} {_money(abs(float(net)))}.")
        if project.get("effective_end_date"): parts.append(f"تاريخ الانتهاء المتوقع {_date_text(project['effective_end_date'])}.")
        summary = " ".join(parts)
    if not detailed:
        return summary
    return summary + (f"\n\nرمز المشروع: {project.get('project_code')}\nBacklog: {_money(project.get('backlog'), arabic)}")
