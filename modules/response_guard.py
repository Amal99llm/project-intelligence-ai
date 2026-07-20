"""Validation and deterministic rendering for verified v2 tool results."""
from __future__ import annotations

from datetime import date, datetime
import re

from modules.business_glossary import CURRENCY_FIELDS, LABELS, PERCENT_FIELDS, STATUS_DISPLAY_AR

INTERNAL_NAMES = {
    "profit_pct", "effective_end_date", "portfolio_filter", "rule_flags", "raw_data",
    "project_id", "project_name", "canonical_fields", "revenue_current", "pl", "dept",
    "amendment_crs", "progress_completed", "net_etc", "etc_pct", "end_date", "start_date",
    "amended_end_date",
}

# Catches any raw snake_case identifier (e.g. "end_date", "project_manager")
# that the composer echoed instead of an Arabic label. A legitimate Arabic
# answer never contains an underscore-joined lowercase English token, so
# this is a safe global guard rather than a fixed, gameable name list.
_INTERNAL_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
FIELD_CUES = {
    "backlog": ("المتبقي", "باقي", "الأعمال"), "revenue": ("الإيراد", "جاب"),
    "total_revenue": ("الإيرادات",), "total_cost": ("التكلفة", "كلف", "صرف"),
    "effective_end_date": ("ينتهي", "انتهاء", "يخلص", "الموعد"),
    "project_manager": ("مدير", "يدير", "ماسك"), "officer_name": ("مسؤول",),
    "profit_margin": ("هامش",), "profit_loss": ("ربح", "خسارة"),
    "progress": ("إنجاز",), "status": ("حالة", "وضع"),
}


def _number(value):
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _magnitude_number(value):
    """Render a large currency value using Arabic business shorthand
    (مليون / مليار) per the business glossary, instead of a raw long
    number. Values under one million keep the grouped-decimal format."""
    value = float(value)
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{sign}{_number(magnitude / 1_000_000_000)} مليار"
    if magnitude >= 1_000_000:
        return f"{sign}{_number(magnitude / 1_000_000)} مليون"
    return f"{sign}{_number(magnitude)}"


def format_value(field, value):
    if value is None:
        return None
    if field in CURRENCY_FIELDS:
        return f"{_magnitude_number(value)} ريال"
    if field in PERCENT_FIELDS:
        number = float(value) * 100 if field == "progress" and abs(float(value)) <= 1 else float(value)
        return f"{_number(number)}%"
    if field == "status":
        return STATUS_DISPLAY_AR.get(value, value)
    if isinstance(value, (date, datetime)):
        return value.strftime("%d/%m/%Y")
    return str(value)


def deterministic_answer(tool_result, fields, *, days_remaining=None, correction=False):
    name = tool_result["project_name"]
    values = tool_result["fields"]
    if days_remaining is not None:
        end = format_value("effective_end_date", values.get("effective_end_date"))
        prefix = "تصحيحًا: " if correction else ""
        return f"{prefix}متبقي {days_remaining} يوم على انتهاء مشروع «{name}»، وتاريخ انتهائه {end}."
    available = [(field, values.get(field)) for field in fields if values.get(field) is not None]
    if not available:
        return f"لا توجد قيمة موثقة متاحة لهذا الطلب في مشروع «{name}»."
    if len(available) == 1:
        field, value = available[0]
        rendered = format_value(field, value)
        prefix = "تصحيحًا: " if correction else ""
        if field == "project_manager":
            return f"{prefix}يدير مشروع «{name}» {rendered}."
        if field == "officer_name":
            return f"{prefix}المسؤول عن مشروع «{name}» هو {rendered}."
        if field == "effective_end_date":
            return f"{prefix}ينتهي مشروع «{name}» بتاريخ {rendered}."
        return f"{prefix}{LABELS.get(field, field)} لمشروع «{name}» {rendered}."
    return "\n".join(f"- {LABELS.get(field, field)}: {format_value(field, value)}" for field, value in available)


def _numeric_tokens(text):
    return [token.replace(",", "").lstrip("0") or "0" for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]


def _allowed_numbers(tool_result, requested_fields):
    allowed = set()
    values = tool_result.get("fields", {})
    for field in requested_fields:
        value = values.get(field)
        if value is None:
            continue
        allowed.update(_numeric_tokens(str(value)))
        allowed.update(_numeric_tokens(format_value(field, value) or ""))
    allowed.update(_numeric_tokens(tool_result.get("project_name", "")))
    allowed.update(_numeric_tokens(str(tool_result.get("verified_calculations", {}))))
    return allowed


def validate(answer, tool_result, requested_fields):
    """Reject any model prose that is not faithful to this turn's result."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    lowered = answer.lower()
    if any(name.lower() in lowered for name in INTERNAL_NAMES) or _INTERNAL_TOKEN.search(lowered):
        return False
    if tool_result.get("project_name") and tool_result["project_name"] not in answer:
        return False
    if not set(_numeric_tokens(answer)).issubset(_allowed_numbers(tool_result, requested_fields)):
        return False
    present = [field for field in requested_fields if tool_result.get("fields", {}).get(field) is not None]
    if any(field in CURRENCY_FIELDS for field in present) and _numeric_tokens(answer) and "ريال" not in answer:
        return False
    if any(field in PERCENT_FIELDS for field in present) and _numeric_tokens(answer) and "%" not in answer:
        return False
    if "دولار" in answer or "درهم" in answer:
        return False
    if len(requested_fields) == 1:
        cues = FIELD_CUES.get(requested_fields[0])
        if cues and not any(cue in answer for cue in cues):
            return False
    return True


def _all_numbers(value):
    if isinstance(value, dict):
        return {token for item in value.values() for token in _all_numbers(item)}
    if isinstance(value, (list, tuple)):
        return {token for item in value for token in _all_numbers(item)}
    return set(_numeric_tokens(str(value)))


def validate_generic(answer, tool_result, metric=None):
    if not isinstance(answer, str) or not answer.strip():
        return False
    lowered = answer.lower()
    if any(name.lower() in lowered for name in INTERNAL_NAMES) or _INTERNAL_TOKEN.search(lowered):
        return False
    if not set(_numeric_tokens(answer)).issubset(_all_numbers(tool_result)):
        return False
    if metric in CURRENCY_FIELDS and _numeric_tokens(answer) and "ريال" not in answer:
        return False
    if metric in PERCENT_FIELDS and _numeric_tokens(answer) and "%" not in answer:
        return False
    return "دولار" not in answer and "درهم" not in answer


def deterministic_tool_answer(tool_name, tool_result, arguments):
    if tool_name == "filter_projects":
        names = [item.get("project_name") for item in tool_result if item and item.get("project_name")]
        return "المشاريع المطابقة:\n" + "\n".join(f"- {name}" for name in names) if names else "لا توجد مشاريع مطابقة موثقة."
    if tool_name == "aggregate_portfolio":
        metric = arguments["metric"]
        if isinstance(tool_result, dict) and "value" in tool_result:
            value = format_value(metric, tool_result["value"])
            return f"{LABELS.get(metric, metric)} للمحفظة: {value}." if value is not None else "لا توجد قيمة موثقة لهذا التجميع."
        return "\n".join(f"- {item.get('group')}: {format_value(metric, item.get('value'))}" for item in tool_result)
    if tool_name == "compare_projects":
        fields = arguments["canonical_fields"]
        return "\n\n".join(deterministic_answer(item, fields) for item in tool_result)
    if tool_name == "get_contract_context":
        return tool_result.get("answer") or "لا توجد إجابة عقدية موثقة."
    return "تعذر عرض النتيجة الموثقة بصيغة آمنة."