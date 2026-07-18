"""Deterministic, explainable executive analysis over canonical project rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import re
from typing import Any

from modules.kpi_calculator import calculate_executive_kpis
from modules.response_formatter import _money, format_arabic_date
from modules.semantic_dictionary import normalize_text


class AnalysisType(str, Enum):
    PORTFOLIO_HEALTH = "portfolio_health"
    MANAGEMENT_PRIORITIES = "management_priorities"
    RISK_ANALYSIS = "risk_analysis"
    CONTRACT_EXPIRATION = "contract_expiration"
    RENEWAL_CANDIDATES = "renewal_candidates"
    EXECUTIVE_MEETING_BRIEF = "executive_meeting_brief"
    PROJECT_ESCALATION = "project_escalation"


@dataclass(frozen=True)
class ExecutiveRequest:
    analyses: tuple[AnalysisType, ...]
    limit: int = 3
    period_days: int | None = None
    assumed_period: bool = False


@dataclass(frozen=True)
class AttentionAssessment:
    project_code: str | None
    project_name: str
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


EXECUTIVE_THRESHOLDS = {
    "low_margin_pct": 5.0, "warning_margin_pct": 10.0,
    "near_expiry_days": 30, "expiry_warning_days": 90,
    "high_risk_amount": 1_000_000.0, "progress_gap_pct": 20.0,
    "high_backlog_ratio": 0.25,
}
_CLOSED = {"completed", "closed", "cancelled", "canceled", "مكتمل", "مغلق", "ملغي", "ملغى"}


def _num(value: Any) -> float | None:
    try:
        return None if value is None or isinstance(value, bool) else float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _active(project: dict[str, Any]) -> bool:
    return normalize_text(str(project.get("status") or "")) not in _CLOSED


def _name(project: dict[str, Any]) -> str:
    return project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code") or "مشروع غير مسمى"


def _extract_limit(q: str) -> int:
    match = re.search(r"(?:اهم|اقرب|اعلى|ابرز)\s+(\d+)", q)
    if match:
        return max(1, min(5, int(match.group(1))))
    for token, value in {"ثلاث": 3, "ثلاثه": 3, "ثلاثة": 3, "خمس": 5, "خمسه": 5, "خمسة": 5, "واحد": 1}.items():
        if token in q:
            return value
    return 3


def _extract_period(query: str) -> tuple[int, bool]:
    q = normalize_text(query)
    numeric = re.search(r"(?:خلال|في)\s+(\d+)\s*(?:يوم|ايام)", q)
    if numeric: return min(730, int(numeric.group(1))), False
    if "اسبوعين" in q: return 14, False
    if "اسبوع" in q: return 7, False
    if "ثلاث" in q and ("شهور" in q or "اشهر" in q): return 90, False
    if "شهر" in q: return 30, False
    return 90, True


def classify_executive_request(query: str) -> ExecutiveRequest | None:
    """Classify concepts before entity resolution into controlled operations."""
    q = normalize_text(query)
    analyses: list[AnalysisType] = []
    meeting = any(x in q for x in ("اجتماع", "dashboard", "داشبورد", "الزبده التنفيذيه", "الزبدة التنفيذية"))
    priority = any(x in q for x in ("تحتاج متابعه", "يحتاج متابعه", "تحتاج تدخل", "يحتاج تدخل", "تدخل الاداره", "الاولويات", "المقلقه", "مقلقه", "تصعيد"))
    risk = "مخاطر" in q or "مخاطره" in q
    renewal = "تجديد" in q and ("العقود" in q or "عقود" in q)
    expiration = ("العقود" in q or "عقود" in q) and any(x in q for x in ("تنتهي", "بتنتهي", "قربت تخلص", "قريبه من الانتهاء", "قريبة من الانتهاء"))
    portfolio = "ملخص المحفظه" in q or "ملخص المحفظة" in q
    if meeting:
        analyses.append(AnalysisType.EXECUTIVE_MEETING_BRIEF)
    else:
        if portfolio: analyses.append(AnalysisType.PORTFOLIO_HEALTH)
        if priority:
            analyses.append(AnalysisType.PROJECT_ESCALATION if ("تدخل" in q or "تصعيد" in q) and "مشروع" in q and "مشاريع" not in q else AnalysisType.MANAGEMENT_PRIORITIES)
        if risk: analyses.append(AnalysisType.RISK_ANALYSIS)
        if renewal: analyses.append(AnalysisType.RENEWAL_CANDIDATES)
        elif expiration: analyses.append(AnalysisType.CONTRACT_EXPIRATION)
    if not analyses: return None
    period, assumed = _extract_period(query)
    return ExecutiveRequest(tuple(dict.fromkeys(analyses)), _extract_limit(q), period, assumed)


def evaluate_project_attention(project: dict[str, Any], today: date | None = None) -> AttentionAssessment:
    today = today or date.today()
    reasons: list[tuple[float, str]] = []
    days, profit, margin = _num(project.get("days_remaining")), _num(project.get("net_profit")), _num(project.get("profit_pct"))
    risk, variance = _num(project.get("risk")), _num(project.get("variance"))
    progress, backlog, contract = _num(project.get("progress_completed")), _num(project.get("backlog")), _num(project.get("total_contract_value"))
    if _active(project) and days is not None and days < 0: reasons.append((30, f"المشروع متأخر عن نهايته التعاقدية بـ {abs(int(days))} يومًا وما زال نشطًا"))
    if profit is not None and profit < 0: reasons.append((30, f"يسجل صافي خسارة قدرها {_money(abs(profit))}"))
    if risk is not None and risk > 0: reasons.append((20 if risk >= EXECUTIVE_THRESHOLDS["high_risk_amount"] else 10, f"تبلغ قيمة المخاطر المالية المسجلة {_money(risk)}"))
    if margin is not None and margin < EXECUTIVE_THRESHOLDS["low_margin_pct"]: reasons.append((15, f"هامش الربح منخفض عند {margin:.1f}%"))
    if variance is not None and variance < 0: reasons.append((10, f"يسجل انحرافًا سلبيًا قدره {abs(variance):.1f}%"))
    if _active(project) and days is not None and 0 <= days <= EXECUTIVE_THRESHOLDS["near_expiry_days"]: reasons.append((15, f"ينتهي العقد خلال {int(days)} يومًا"))
    if _active(project) and progress is not None and project.get("start_date") and project.get("effective_end_date"):
        try:
            start, end = date.fromisoformat(str(project["start_date"])[:10]), date.fromisoformat(str(project["effective_end_date"])[:10])
            elapsed = max(0, min(100, (today - start).days / max(1, (end - start).days) * 100))
            gap = elapsed - progress
            if gap >= EXECUTIVE_THRESHOLDS["progress_gap_pct"]: reasons.append((15, f"الإنجاز {progress:.0f}% ومتأخر بنحو {gap:.0f} نقطة عن الزمن المنقضي"))
        except ValueError: pass
    if backlog and days is not None and days <= EXECUTIVE_THRESHOLDS["expiry_warning_days"] and (not contract or backlog / max(contract, 1) >= EXECUTIVE_THRESHOLDS["high_backlog_ratio"]):
        reasons.append((10, f"توجد أعمال متبقية بقيمة {_money(backlog)} مع قرب نهاية العقد"))
    return AttentionAssessment(project.get("project_code"), _name(project), sum(x[0] for x in reasons), tuple(x[1] for x in reasons))


def rank_management_priorities(projects: list[dict[str, Any]], today: date, limit: int = 3) -> list[AttentionAssessment]:
    items = [evaluate_project_attention(p, today) for p in projects if _active(p)]
    return sorted((x for x in items if x.reasons), key=lambda x: (-x.score, x.project_name))[:limit]


def _priority_text(items: list[AttentionAssessment], limit: int) -> str:
    if not items: return "لا تظهر في البيانات الحالية مشاريع نشطة تحمل إشارات أولوية قابلة للقياس."
    lines = [f"أهم {min(limit, len(items))} مشاريع تستحق المتابعة حاليًا:"]
    for idx, item in enumerate(items, 1):
        lines.append(f"{idx}. «{item.project_name}»")
        lines.extend(f"   - {reason}." for reason in item.reasons[:3])
    return "\n".join(lines)


def _risk_text(projects: list[dict[str, Any]], limit: int) -> str:
    rows = sorted(((p, _num(p.get("risk"))) for p in projects if (_num(p.get("risk")) or 0) > 0), key=lambda x: (-x[1], _name(x[0])))[:limit]
    if not rows: return "المخاطر: لا توجد قيم مخاطر مالية رقمية مسجلة يمكن ترتيبها بثقة."
    lines = ["أعلى المخاطر المالية المسجلة حاليًا:"] + [f"{i}. «{_name(p)}» — {_money(risk)}" for i, (p, risk) in enumerate(rows, 1)]
    lines.append("البيانات تتضمن قيمة مالية للمخاطر، ولا تتضمن وصفًا تفصيليًا لكل خطر.")
    return "\n".join(lines)


def _expiration_rows(projects, days, limit):
    rows = [p for p in projects if _active(p) and _num(p.get("days_remaining")) is not None and 0 <= _num(p.get("days_remaining")) <= days]
    return sorted(rows, key=lambda p: (_num(p.get("days_remaining")), _name(p)))[:limit]


def _expiration_text(projects, request):
    rows = _expiration_rows(projects, request.period_days or 90, request.limit)
    prefix = "اعتبرت العقود القريبة من الانتهاء هي التي تنتهي خلال 90 يومًا.\n" if request.assumed_period else ""
    if not rows: return prefix + f"لا توجد عقود نشطة تنتهي خلال {request.period_days or 90} يومًا وفق البيانات الحالية."
    lines = [prefix + f"العقود النشطة التي تنتهي خلال {request.period_days or 90} يومًا:"]
    lines += [f"{i}. «{_name(p)}» — خلال {int(_num(p['days_remaining']))} يومًا، في {format_arabic_date(p.get('effective_end_date'))}." for i, p in enumerate(rows, 1)]
    return "\n".join(lines)


def _renewal_text(projects, request):
    candidates = [p for p in _expiration_rows(projects, request.period_days or 90, 100) if (_num(p.get("progress_completed")) is not None and _num(p.get("progress_completed")) < 100) or (_num(p.get("backlog")) or 0) > 0][:request.limit]
    heading = "هذه العقود مرشحة لمراجعة التجديد أو الإقفال، وليست توصية مؤكدة بالتجديد:"
    if not candidates: return heading + "\n- لا تظهر حاليًا عقود نشطة قريبة من الانتهاء وبها أعمال غير مكتملة."
    lines = [heading]
    for i, p in enumerate(candidates, 1):
        reasons = [f"ينتهي خلال {int(_num(p['days_remaining']))} يومًا", "المشروع ما زال نشطًا"]
        if _num(p.get("progress_completed")) is not None: reasons.append(f"نسبة الإنجاز {_num(p['progress_completed']):.0f}%")
        if (_num(p.get("backlog")) or 0) > 0: reasons.append(f"الأعمال المتبقية {_money(_num(p['backlog']))}")
        lines.append(f"{i}. «{_name(p)}»: " + "؛ ".join(reasons) + ".")
    return "\n".join(lines)


def _portfolio_text(projects, today):
    k = calculate_executive_kpis(projects, today=today)
    return (f"تضم المحفظة {k['total_projects']} مشروعًا، منها {k['active_projects']} مشروعًا نشطًا. الإيرادات {_money(k['total_revenue'])}، "
            f"وصافي الربح {_money(k['net_profit'])}، وهامش الربح {k['profit_margin_pct']:.2f}%، والأعمال المتبقية {_money(k['backlog'])}.")


def execute_executive_request(request: ExecutiveRequest, projects: list[dict[str, Any]], today: date) -> str:
    sections: list[str] = []
    for analysis in request.analyses:
        if analysis == AnalysisType.PORTFOLIO_HEALTH: sections.append("ملخص المحفظة:\n" + _portfolio_text(projects, today))
        elif analysis in {AnalysisType.MANAGEMENT_PRIORITIES, AnalysisType.PROJECT_ESCALATION}:
            limit = 1 if analysis == AnalysisType.PROJECT_ESCALATION else request.limit
            sections.append(_priority_text(rank_management_priorities(projects, today, limit), limit))
        elif analysis == AnalysisType.RISK_ANALYSIS: sections.append(_risk_text(projects, request.limit))
        elif analysis == AnalysisType.CONTRACT_EXPIRATION: sections.append(_expiration_text(projects, request))
        elif analysis == AnalysisType.RENEWAL_CANDIDATES: sections.append(_renewal_text(projects, request))
        elif analysis == AnalysisType.EXECUTIVE_MEETING_BRIEF:
            losses = sorted((p for p in projects if (_num(p.get("net_profit")) or 0) < 0), key=lambda p: _num(p.get("net_profit")))[:3]
            sections += ["الملخص التنفيذي:\n" + _portfolio_text(projects, today), _priority_text(rank_management_priorities(projects, today, 3), 3),
                         _expiration_text(projects, ExecutiveRequest((AnalysisType.CONTRACT_EXPIRATION,), 3, 90, False)), _risk_text(projects, 3),
                         "أكبر المشاريع الخاسرة:\n" + ("\n".join(f"- «{_name(p)}» — صافي خسارة {_money(abs(_num(p.get('net_profit'))))}." for p in losses) or "- لا توجد مشاريع خاسرة مسجلة."),
                         "التوصية: ركّز الاجتماع على المشاريع المتأخرة والخاسرة، ثم احسم خطة العقود القريبة من الانتهاء."]
    return "\n\n".join(sections)
