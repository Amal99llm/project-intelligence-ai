"""Structured semantics and deterministic answers for project-contract dialogue."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from modules.semantic_dictionary import detect_requested_fields, extract_field_concepts, normalize_text
from modules.response_formatter import _money, format_arabic_date, format_project_metrics


@dataclass(frozen=True)
class ContractRequest:
    metrics: tuple[str, ...] = field(default_factory=tuple)
    operation: str = "get"  # get | remaining | duration | elapsed | expiry | amendment


def _has(q: str, *phrases: str) -> bool:
    return any(normalize_text(phrase) in q for phrase in phrases)


def parse_future_period_days(query: str) -> int | None:
    """Parse a bounded colloquial future period into days."""
    import re
    q = normalize_text(query)
    numeric = re.search(r"خلال\s+(\d+)\s*(?:يوم|ايام)", q)
    if numeric:
        return min(int(numeric.group(1)), 730)
    if any(t in q for t in ("خلال اسبوعين", "الاسبوعين الجايه")):
        return 14
    if any(t in q for t in ("خلال اسبوع", "الاسبوع الجاي")):
        return 7
    if any(t in q for t in ("خلال ثلاث شهور", "خلال 3 شهور", "خلال ثلاثه اشهر")):
        return 90
    if any(t in q for t in ("خلال شهر", "الشهر الجاي", "بعد شهر")):
        return 30
    return None


def analyze_contract_request(query: str) -> ContractRequest | None:
    q = normalize_text(query)
    concepts = extract_field_concepts(query)
    metrics = [item.canonical for item in detect_requested_fields(query)]
    contract_metrics = {"contract_value", "amendment_crs", "total_contract_value"}

    def with_contract(*values: str) -> list[str]:
        return [metric for metric in metrics if metric not in contract_metrics] + list(values)

    contract_context = any(t in q for t in ("عقد", "قيمته", "قيمه", "القيمه", "القيمة", "cr", "تعديل", "contract", "amendment"))
    if {"amendment", "end"}.issubset(concepts) and "value" not in concepts:
        return ContractRequest(tuple(metric for metric in metrics if metric != "amendment_crs"), "amendment")
    if any(t in q for t in ("العقد الاساسي", "العقد الأساسي", "قيمه اساسيه", "قيمة أساسية", "base contract value")):
        metrics = with_contract("contract_value")
    elif any(t in q for t in ("قيمه التعديلات", "قيمة التعديلات", "قيمه ال cr", "قيمة الـ cr", "كم cr", "contract amendments", "how much are the amendments")):
        metrics = with_contract("amendment_crs")
    elif _has(q, "هل صار عليه تعديل", "هل عليه تعديلات"):
        metrics = with_contract("amendment_crs")
    elif any(t in q for t in ("بعد التعديلات", "القيمه الاجماليه", "القيمة الإجمالية", "total after cr", "total after amendments")):
        metrics = with_contract("contract_value", "amendment_crs", "total_contract_value")
    elif contract_context and any(t in q for t in ("كم قيمته", "قيمه العقد", "قيمة العقد", "قيمه عقده", "قيمة عقده",
                                                    "عقده بكم", "قيمه المشروع", "قيمة المشروع")):
        metrics = with_contract("total_contract_value")

    timeline_metrics = {"start_date", "effective_end_date", "days_remaining"}
    if (any(t in q for t in ("كم باقي", "باقي له", "باقي عليه", "باقي على", "قرب يخلص", "باقي كثير", "time is left", "time left", "remaining time", "how much remaining"))
            and (not metrics or set(metrics) <= timeline_metrics)):
        return ContractRequest(tuple(metrics), "remaining")
    if any(t in q for t in ("كم مده", "كم مدة", "كم مدته", "من البدايه للنهايه", "من البداية للنهاية", "contract duration", "how long is the contract")):
        return ContractRequest(tuple(metrics), "duration")
    if any(t in q for t in ("كم له شغال", "من متى شغال")):
        return ContractRequest(tuple(metrics), "elapsed")
    if any(t in q for t in ("هل انتهى", "هل انتها", "العقد منتهي", "العقد ساري", "قرب ينتهي", "يحتاج تجديد", "has it expired", "close to expiry", "close to expiring")):
        return ContractRequest(tuple(metrics), "expiry")
    if _has(q, "تعديل على تاريخ", "تعديل على النهايه", "تعديل على النهاية"):
        return ContractRequest(tuple(metrics), "amendment")

    if metrics and (contract_context or any(m in metrics for m in {
        "start_date", "effective_end_date", "days_remaining", "contract_value",
        "amendment_crs", "total_contract_value", "support_document",
    })):
        return ContractRequest(tuple(metrics), "get")
    return None


def _period_text(days: int) -> str:
    days = abs(int(days))
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    parts = []
    if years:
        parts.append(f"{years} " + ("سنة" if years == 1 else "سنوات"))
    if months:
        parts.append(f"{months} " + ("شهر" if months == 1 else "أشهر"))
    if days and not years:
        parts.append(f"{days} يومًا")
    return " و".join(parts) or "أقل من يوم"


def _as_date(value) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def render_contract_answer(project: dict, request: ContractRequest, today: date) -> str:
    status = str(project.get("status") or "")
    closed = status in {"Completed", "Closed", "Cancelled", "Canceled"}
    start = _as_date(project.get("start_date"))
    end = _as_date(project.get("effective_end_date"))
    days_remaining = project.get("days_remaining")
    if days_remaining is None and end:
        days_remaining = (end - today).days

    # Metric execution is shared; only the final wording follows the user's
    # language. The orchestrator attaches this flag for English turns.
    english = bool(project.get("_response_language") == "en")
    name = project.get("project_name_en") or project.get("project_name_ar") or project.get("project_code")
    if english:
        if request.operation == "remaining":
            return ("No valid end date is recorded for this project." if end is None else
                    f"{name} has {int(days_remaining)} days remaining, ending on {end.isoformat()}.")
        if request.operation == "duration":
            return ("Valid start and end dates are not both recorded." if not start or not end else
                    f"The contract duration is {(end - start).days} days, from {start.isoformat()} to {end.isoformat()}.")
        if request.operation == "expiry":
            return ("No valid end date is recorded for this project." if end is None else
                    (f"The contract expired {abs(int(days_remaining))} days ago." if days_remaining < 0 else f"The contract expires in {int(days_remaining)} days."))
        if request.operation == "amendment":
            amended = project.get("amended_end_date")
            return (f"The amended end date is {amended}." if amended else "No amended end date is recorded.")
        labels = {"contract_value": "Base contract value", "amendment_crs": "Contract amendments", "total_contract_value": "Total contract value",
                  "start_date": "Start date", "effective_end_date": "Effective end date", "days_remaining": "Days remaining"}
        return "\n".join(f"{labels.get(metric, metric)}: {_money(project.get(metric) or 0, False) if metric in {'contract_value','amendment_crs','total_contract_value'} else project.get(metric)}." for metric in request.metrics)

    if request.operation == "remaining":
        if end is None:
            return "لا يتوفر تاريخ نهاية مسجل لهذا المشروع."
        if closed:
            return f"المشروع مكتمل، وكان تاريخ انتهائه {format_arabic_date(end)}."
        days = int(float(days_remaining))
        if days < 0:
            answer = f"العقد تجاوز تاريخ نهايته بـ {abs(days)} يومًا، وما زالت حالته جاري التنفيذ."
        else:
            answer = f"باقي على نهاية المشروع تقريبًا {_period_text(days)}، وينتهي في {format_arabic_date(end)}."
        if "start_date" in request.metrics and start:
            answer = f"بدأ المشروع في {format_arabic_date(start)}. " + answer
        return answer

    if request.operation == "duration":
        if not start or not end:
            return "لا تتوفر تواريخ بداية ونهاية مكتملة لحساب مدة المشروع."
        days = (end - start).days
        return (f"مدة المشروع المخططة نحو {_period_text(days)}، من {format_arabic_date(start)} "
                f"إلى {format_arabic_date(end)}.")

    if request.operation == "elapsed":
        if not start:
            return "لا يتوفر تاريخ بداية مسجل لهذا المشروع."
        return f"المشروع بدأ قبل {_period_text((today - start).days)}، وتحديدًا في {format_arabic_date(start)}."

    if request.operation == "expiry":
        if end is None:
            return "لا يتوفر تاريخ نهاية مسجل لهذا العقد."
        days = int(float(days_remaining))
        if closed:
            return f"العقد مغلق، وكان تاريخ انتهائه {format_arabic_date(end)}."
        if days < 0:
            return f"نعم، العقد انتهى منذ {abs(days)} يومًا وما زال مصنفًا جاري التنفيذ."
        if days <= 30:
            return (f"العقد ينتهي خلال {days} يومًا، في {format_arabic_date(end)}؛ لذلك قد يحتاج بدء إجراءات "
                    "التجديد أو الإقفال حسب خطة المشروع.")
        return f"العقد ساري، ويتبقى على نهايته {_period_text(days)} حتى {format_arabic_date(end)}."

    if request.operation == "amendment":
        original = project.get("end_date")
        amended = project.get("amended_end_date")
        if amended:
            return (f"نعم، تاريخ النهاية الأصلي كان {format_arabic_date(original)}، وبعد التعديل أصبح "
                    f"{format_arabic_date(amended)}.")
        return f"لا يوجد تاريخ نهاية معدل مسجل؛ تاريخ النهاية الحالي {format_arabic_date(original)}."

    metrics = list(request.metrics)
    if set(metrics) >= {"contract_value", "amendment_crs", "total_contract_value"}:
        return (f"قيمة العقد الأساسية {_money(project.get('contract_value') or 0)}، وبعد التعديلات البالغة "
                f"{_money(project.get('amendment_crs') or 0)} أصبحت القيمة الإجمالية "
                f"{_money(project.get('total_contract_value') or 0)}.")
    return format_project_metrics(project, metrics)
