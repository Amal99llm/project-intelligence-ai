"""Transparent, deterministic executive attention rules over verified rows."""

from __future__ import annotations

from datetime import date
from typing import Any


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def project_attention(project: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    """Return facts and a transparent priority score; never infer causes."""
    reasons: list[dict[str, Any]] = []
    profit = _number(project.get("net_profit"))
    margin = _number(project.get("profit_pct"))
    days = _number(project.get("days_remaining"))
    risk = _number(project.get("risk"))
    variance = _number(project.get("variance"))
    status = project.get("status")

    if profit is not None and profit < 0:
        reasons.append({"rule": "negative_profit", "field": "net_profit", "value": profit, "weight": 30})
    if days is not None and days < 0 and status not in {"Completed", "Closed"}:
        reasons.append({"rule": "overdue_open_project", "field": "days_remaining", "value": days, "weight": 25})
    elif days is not None and 0 <= days < 30 and status not in {"Completed", "Closed"}:
        reasons.append({"rule": "contract_expiring_within_30_days", "field": "days_remaining", "value": days, "weight": 15})
    if risk is not None and risk > 0:
        reasons.append({"rule": "recorded_risk", "field": "risk", "value": risk, "weight": min(20, 5 + abs(risk))})
    if variance is not None and variance < 0:
        reasons.append({"rule": "negative_variance", "field": "variance", "value": variance, "weight": 10})
    if margin is not None and margin < 5:
        reasons.append({"rule": "low_profit_margin", "field": "profit_pct", "value": margin, "weight": 10})

    return {
        "project_code": project.get("project_code"),
        "project_name": project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code"),
        "attention_score": round(sum(float(reason["weight"]) for reason in reasons), 2),
        "reasons": reasons,
    }


def rank_attention(projects: list[dict[str, Any]], today: date | None = None, limit: int = 5) -> list[dict[str, Any]]:
    ranked = [project_attention(project, today) for project in projects]
    ranked = [item for item in ranked if item["reasons"]]
    ranked.sort(key=lambda item: (-item["attention_score"], item["project_name"] or ""))
    return ranked[:limit]


def project_opportunity(project: dict[str, Any]) -> dict[str, Any]:
    """Score upside from several verified dimensions; no single metric can win alone."""
    facts: list[dict[str, Any]] = []
    profit, margin = _number(project.get("net_profit")), _number(project.get("profit_pct"))
    backlog, progress = _number(project.get("backlog")), _number(project.get("progress_completed"))
    days, risk = _number(project.get("days_remaining")), _number(project.get("risk"))
    etc_revenue, planned_profit = _number(project.get("etc_revenue")), _number(project.get("planned_profit"))
    if profit is not None and profit > 0: facts.append({"rule": "positive_profit", "value": profit, "weight": 20})
    if margin is not None and margin >= 10: facts.append({"rule": "healthy_margin", "value": margin, "weight": 15})
    if backlog is not None and backlog > 0: facts.append({"rule": "remaining_backlog", "value": backlog, "weight": 15})
    if progress is not None and 0.25 <= progress <= 0.9: facts.append({"rule": "healthy_delivery_window", "value": progress, "weight": 10})
    if days is not None and days > 30: facts.append({"rule": "contract_time_remaining", "value": days, "weight": 10})
    if risk in (None, 0): facts.append({"rule": "no_recorded_risk", "value": risk, "weight": 10})
    if etc_revenue is not None and etc_revenue > 0: facts.append({"rule": "future_revenue_recorded", "value": etc_revenue, "weight": 10})
    if planned_profit is not None and planned_profit > 0: facts.append({"rule": "planned_profit_positive", "value": planned_profit, "weight": 10})
    return {
        "project_code": project.get("project_code"),
        "project_name": project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code"),
        "opportunity_score": sum(item["weight"] for item in facts), "facts": facts,
    }


def rank_opportunities(projects: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows = [project_opportunity(project) for project in projects]
    rows = [row for row in rows if len(row["facts"]) >= 3]
    rows.sort(key=lambda row: (-row["opportunity_score"], row["project_name"] or ""))
    return rows[:limit]


def format_executive_analysis(intent: str, projects: list[dict[str, Any]], today: date | None = None, limit: int = 3) -> str:
    """Present verified executive indicators and recommendations."""
    labels = {
        "negative_profit": "صافي ربح سالب", "overdue_open_project": "مشروع مفتوح تجاوز تاريخ انتهائه",
        "contract_expiring_within_30_days": "موعد انتهاء خلال 30 يومًا", "recorded_risk": "مخاطر مسجلة",
        "negative_variance": "انحراف سلبي عن الخطة", "low_profit_margin": "هامش ربح منخفض",
        "positive_profit": "ربحية موجبة", "healthy_margin": "هامش ربح جيد",
        "remaining_backlog": "أعمال متبقية قابلة للتنفيذ", "healthy_delivery_window": "مرحلة تنفيذ مناسبة",
        "contract_time_remaining": "مدة تعاقدية متبقية", "no_recorded_risk": "لا توجد مخاطر مسجلة",
        "future_revenue_recorded": "إيرادات مستقبلية مسجلة", "planned_profit_positive": "ربحية مخططة موجبة",
    }
    if intent in {"executive_priority", "executive_risk"}:
        ranked = rank_attention(projects, today, limit=limit)
        if not ranked: return "لا تظهر مؤشرات موثقة تستوفي قواعد المخاطر والمتابعة الحالية."
        item = ranked[0]
        rules = "، ".join(labels[reason["rule"]] for reason in item["reasons"])
        heading = "أعلى أولوية متابعة" if intent == "executive_priority" else "أبرز مصدر قلق"
        return (f"{heading}: «{item['project_name']}». الحقائق الموثقة هي القيم المسجلة للمشروع. "
                f"المؤشرات التنفيذية: {rules}. التوصية: مراجعة هذه المؤشرات مع مسؤول المشروع والتحقق من خطة المعالجة.")
    if intent == "executive_opportunity":
        ranked = rank_opportunities(projects, limit=1)
        if not ranked: return "لا تتوفر حاليًا فرصة تستوفي ثلاثة أبعاد موثقة على الأقل من قواعد الفرص."
        item = ranked[0]
        rules = "، ".join(labels[fact["rule"]] for fact in item["facts"])
        return (f"أبرز فرصة وفق التقييم متعدد العوامل هي «{item['project_name']}». "
                f"المؤشرات التنفيذية: {rules}. التوصية: التحقق من قابلية تحويل الأعمال المتبقية والربحية المخططة إلى نتائج، دون افتراض سبب غير موثق.")
    risks = rank_attention(projects, today, limit=1)
    opportunities = rank_opportunities(projects, limit=1)
    total_profit = sum(_number(project.get("net_profit")) or 0 for project in projects)
    parts = [f"1. المحفظة تضم {len(projects)} مشروعًا، وصافي الربح المحسوب {total_profit:,.2f} ريال."]
    parts.append(f"2. أبرز متابعة حسب القواعد: «{risks[0]['project_name']}»." if risks else "2. لا توجد مؤشرات متابعة ضمن القواعد الحالية.")
    parts.append(f"3. أبرز فرصة متعددة العوامل: «{opportunities[0]['project_name']}»." if opportunities else "3. لا توجد فرصة تستوفي الحد الأدنى من القواعد.")
    return "موجز الاجتماع من بيانات موثقة وقواعد معلنة:\n" + "\n".join(parts[:limit])


def format_attention_summary(
    projects: list[dict[str, Any]], today: date | None = None, *, single: bool = False,
) -> str:
    ranked = rank_attention(projects, today, limit=1 if single else 5)
    if not ranked:
        return "لا تظهر في البيانات الحالية مشاريع تستوفي قواعد المتابعة التنفيذية المحددة."
    labels = {
        "negative_profit": "صافي ربح سالب",
        "overdue_open_project": "تجاوز تاريخ الانتهاء وما زال مفتوحًا",
        "contract_expiring_within_30_days": "ينتهي خلال 30 يومًا",
        "recorded_risk": "لديه قيمة مخاطر مسجلة",
        "negative_variance": "انحراف سلبي عن الخطة",
        "low_profit_margin": "هامش ربح أقل من 5%",
    }
    lines = []
    for index, item in enumerate(ranked, 1):
        reasons = "، ".join(labels[reason["rule"]] for reason in item["reasons"])
        lines.append(f"{index}. {item['project_name']}: {reasons}.")
    if single:
        item = ranked[0]
        reasons = "، ".join(labels[reason["rule"]] for reason in item["reasons"])
        return (
            f"أعلى مشروع يحتاج متابعة وفق القواعد المحددة هو «{item['project_name']}». "
            f"المؤشرات الموثقة: {reasons}. التوصية: مراجعة هذه المؤشرات مع مسؤول المشروع."
        )
    return "أولوية المتابعة وفق القواعد المعلنة:\n" + "\n".join(lines)
