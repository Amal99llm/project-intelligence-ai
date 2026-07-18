"""Canonical executive KPI calculations for the project portfolio.

All dashboard surfaces consume the result of this module.

Profit/loss source (decision recorded here, single place):
  This module previously always recomputed profit as
  (total_revenue - total_cost), on the reasoning that a stored derived
  column can silently drift from its inputs. After review, the project
  owner has decided the finance-approved, official figure is the stored
  `pl` ("P&L") column -- the ~1 SAR differences observed against
  (total_revenue - total_cost) are accepted rounding, not drift worth
  distrusting the official column over. PROFIT_LOSS_SOURCE below is the
  single switch that decision is recorded through; flip it back to
  "computed" to revert to always deriving profit from revenue/cost.
  A data-quality signal (`pl_variance_flagged_projects` in the KPI dict
  below) still flags any project where the two disagree by more than a
  small tolerance, purely for visibility -- it never blocks an answer.
"""

from __future__ import annotations

from datetime import date, datetime
from modules.time_utils import riyadh_today
from typing import Any, Iterable, Mapping

PROFIT_LOSS_SOURCE = "stored_pl"  # "stored_pl" | "computed"
PL_VARIANCE_TOLERANCE = 1.01  # SAR; just above the ~1 SAR rounding noise observed


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def normalize_progress(value: Any) -> float:
    """Normalize Excel fractional progress to a clamped 0-100 percentage."""
    progress = _number(value)
    if -1.5 <= progress <= 1.5:
        progress *= 100
    return _clamp(progress)


def effective_end_date(project: Mapping[str, Any]) -> date | None:
    """Use amended_end_date when populated, otherwise end_date."""
    return _date(project.get("amended_end_date")) or _date(project.get("end_date"))


def project_financials(project: Mapping[str, Any]) -> tuple[float, float]:
    """Return (net_profit, profit_margin_pct) per PROFIT_LOSS_SOURCE above.
    This is the single function both project-level and portfolio-level
    profit figures go through, so the two can never disagree."""
    revenue = _number(project.get("total_revenue"))
    if PROFIT_LOSS_SOURCE == "stored_pl" and "pl" in project and project.get("pl") is not None:
        profit = _number(project.get("pl"))
    else:
        cost = _number(project.get("total_cost"))
        profit = revenue - cost
    margin = profit / revenue * 100 if revenue else 0.0
    return profit, margin


def _computed_profit(project: Mapping[str, Any]) -> float:
    """Always (total_revenue - total_cost), regardless of
    PROFIT_LOSS_SOURCE -- used only for the pl-vs-computed data-quality
    variance check, never for an answer."""
    return _number(project.get("total_revenue")) - _number(project.get("total_cost"))


def calculate_executive_kpis(
    projects: Iterable[Mapping[str, Any]], today: date | None = None
) -> dict[str, Any]:
    """Calculate the single canonical set of executive dashboard KPIs.

    Portfolio-health component normalization:
    - Profitability: portfolio margin, with 20% margin representing 100 points.
    - Schedule: share of dated projects that are not overdue; Completed/Closed
      projects count as schedule-healthy.
    - Completion: average clamped progress_completed percentage.
    - Contract risk: share of Ongoing projects not overdue, expiring within
      30 days, or missing an effective end date.
    - Data quality: completeness of code, name, status, effective end date,
      and project manager.
    """
    today = today or riyadh_today()

    # project_code is the database's unique project identifier.  Deduplicating
    # here prevents accidental row multiplication from inflating any KPI.
    unique: dict[str, Mapping[str, Any]] = {}
    for project in projects:
        code = str(project.get("project_code") or "").strip()
        if code:
            unique[code] = project
    portfolio = list(unique.values())

    total_contract_value = sum(_number(p.get("total_contract_value")) for p in portfolio)
    total_revenue = sum(_number(p.get("total_revenue")) for p in portfolio)
    total_cost = sum(_number(p.get("total_cost")) for p in portfolio)
    backlog = sum(_number(p.get("backlog")) for p in portfolio)

    enriched = []
    pl_variance_flagged = 0
    for project in portfolio:
        end = effective_end_date(project)
        days = (end - today).days if end else None
        profit, _ = project_financials(project)
        status = str(project.get("status") or "").strip()
        enriched.append((project, status, end, days, profit))
        if abs(profit - _computed_profit(project)) > PL_VARIANCE_TOLERANCE:
            pl_variance_flagged += 1

    # Portfolio-level profit/margin are derived from the exact same
    # per-project project_financials() figures used everywhere else
    # (single-project summaries, KPI Registry) -- they can never disagree.
    net_profit = sum(row[4] for row in enriched)
    profit_margin = net_profit / total_revenue * 100 if total_revenue else 0.0

    active = [row for row in enriched if row[1] == "Ongoing"]
    completed = sum(1 for row in enriched if row[1] in {"Completed", "Closed"})
    losing = sum(1 for row in enriched if row[4] < 0)
    expiring = sum(1 for row in enriched if row[3] is not None and 0 <= row[3] < 30)

    profitability_score = _clamp(profit_margin / 20 * 100)

    dated = [row for row in enriched if row[2] is not None]
    schedule_healthy = sum(
        1 for row in dated
        if row[1] in {"Completed", "Closed"} or (row[3] is not None and row[3] >= 0)
    )
    schedule_score = schedule_healthy / len(dated) * 100 if dated else 0.0

    completion_score = (
        sum(normalize_progress(row[0].get("progress_completed")) for row in enriched)
        / len(enriched)
        if enriched else 0.0
    )

    # Missing dates are contract risk, as are overdue and imminent contracts.
    at_risk = sum(1 for row in active if row[3] is None or row[3] < 30)
    contract_risk_score = 100 - at_risk / len(active) * 100 if active else 100.0

    required_values = 0
    for project, status, end, _days, _profit in enriched:
        required_values += sum((
            bool(str(project.get("project_code") or "").strip()),
            bool(str(project.get("project_name_ar") or project.get("project_name_en") or "").strip()),
            bool(status),
            bool(end),
            bool(str(project.get("project_manager") or "").strip()),
        ))
    data_quality_score = required_values / (len(enriched) * 5) * 100 if enriched else 0.0

    health_score = round(
        profitability_score * 0.30
        + schedule_score * 0.25
        + completion_score * 0.20
        + contract_risk_score * 0.15
        + data_quality_score * 0.10
    )
    health_label = (
        "Excellent" if health_score >= 85 else
        "Good" if health_score >= 70 else
        "Warning" if health_score >= 50 else
        "Critical"
    )

    return {
        "total_projects": len(portfolio),
        "total_contract_value": total_contract_value,
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "net_profit": net_profit,
        "profit_margin_pct": profit_margin,
        "losing_projects": losing,
        "active_projects": len(active),
        "completed_projects": completed,
        "contracts_expiring_soon": expiring,
        "backlog": backlog,
        "amendments_total": sum(_number(p.get("amendment_crs")) for p in portfolio),
        "current_year_revenue": sum(_number(p.get("revenue_current")) for p in portfolio),
        "current_year_cost": sum(
            _number(p.get("cost_of_revenue")) + _number(p.get("other_cost")) for p in portfolio
        ),
        "portfolio_health_score": health_score,
        "portfolio_health_label": health_label,
        "portfolio_health_components": {
            "profitability": round(profitability_score, 2),
            "schedule_health": round(schedule_score, 2),
            "project_completion": round(completion_score, 2),
            "contract_risk": round(contract_risk_score, 2),
            "data_quality": round(data_quality_score, 2),
        },
        # Data-quality visibility only (see PROFIT_LOSS_SOURCE note at top of
        # file) -- count of projects where the official `pl` column and
        # (total_revenue - total_cost) disagree by more than PL_VARIANCE_TOLERANCE.
        # Never used to alter an answer, just surfaced for internal review.
        "pl_variance_flagged_projects": pl_variance_flagged,
    }


# ── KPI Registry (Section 18) ───────────────────────────────────────────────
# Every named executive KPI has exactly one calculation, defined here and
# nowhere else. Dashboard, API and chatbot all read from this registry so
# there is a single source of truth; nothing scattered across HTML/JS/Python.

def _kpi_total_projects(projects, today=None): return calculate_executive_kpis(projects, today)["total_projects"]
def _kpi_total_contract_value(projects, today=None): return calculate_executive_kpis(projects, today)["total_contract_value"]
def _kpi_revenue(projects, today=None): return calculate_executive_kpis(projects, today)["total_revenue"]
def _kpi_cost(projects, today=None): return calculate_executive_kpis(projects, today)["total_cost"]
def _kpi_backlog(projects, today=None): return calculate_executive_kpis(projects, today)["backlog"]
def _kpi_profit_loss(projects, today=None): return calculate_executive_kpis(projects, today)["net_profit"]
def _kpi_profit_margin(projects, today=None): return calculate_executive_kpis(projects, today)["profit_margin_pct"]
def _kpi_losing_projects(projects, today=None): return calculate_executive_kpis(projects, today)["losing_projects"]
def _kpi_active_projects(projects, today=None): return calculate_executive_kpis(projects, today)["active_projects"]
def _kpi_completed_projects(projects, today=None): return calculate_executive_kpis(projects, today)["completed_projects"]
def _kpi_contracts_expiring_soon(projects, today=None): return calculate_executive_kpis(projects, today)["contracts_expiring_soon"]
def _kpi_amendments_total(projects, today=None): return calculate_executive_kpis(projects, today)["amendments_total"]
def _kpi_current_year_revenue(projects, today=None): return calculate_executive_kpis(projects, today)["current_year_revenue"]
def _kpi_current_year_cost(projects, today=None): return calculate_executive_kpis(projects, today)["current_year_cost"]


def _registry_entry(display_name, source_columns, formula, description, compute, filters=None):
    return {
        "display_name": display_name,
        "source_table": "backlog_projects",
        "source_columns": list(source_columns),
        "formula": formula,
        "description": description,
        "filters": list(filters or []),
        "compute": compute,
    }


KPI_REGISTRY: dict[str, dict[str, Any]] = {
    "total_projects": _registry_entry(
        "إجمالي المشاريع", ["project_code"], "COUNT(DISTINCT project_code)",
        "عدد رموز المشاريع الفريدة في المحفظة.", _kpi_total_projects,
    ),
    "total_contract_value": _registry_entry(
        "إجمالي قيمة العقود", ["total_contract_value"], "SUM(total_contract_value)",
        "مجموع إجمالي قيمة العقد بعد التعديلات لكل مشروع فريد.", _kpi_total_contract_value,
    ),
    "revenue": _registry_entry(
        "إجمالي الإيرادات", ["total_revenue"], "SUM(total_revenue)",
        "مجموع الإيرادات الكلية لجميع المشاريع الفريدة.", _kpi_revenue,
    ),
    "cost": _registry_entry(
        "إجمالي التكاليف", ["total_cost"], "SUM(total_cost)",
        "مجموع التكاليف الكلية لجميع المشاريع الفريدة.", _kpi_cost,
    ),
    "backlog": _registry_entry(
        "Backlog", ["backlog"], "SUM(backlog)",
        "مجموع القيم المخزنة في عمود Backlog لجميع المشاريع الفريدة.", _kpi_backlog,
    ),
    "profit_loss": _registry_entry(
        "صافي الربح", ["pl"],
        "SUM(pl)",
        "مجموع عمود P&L (pl) الرسمي المعتمد لكل مشروع فريد.", _kpi_profit_loss,
    ),
    "profit_margin": _registry_entry(
        "هامش الربح", ["pl", "total_revenue"],
        "SUM(pl) / SUM(total_revenue) * 100",
        "صافي ربح المحفظة (عمود P&L الرسمي) كنسبة من إجمالي الإيرادات؛ لا يُحسب بمتوسط نسب المشاريع.",
        _kpi_profit_margin,
    ),
    "losing_projects": _registry_entry(
        "المشاريع الخاسرة", ["pl"],
        "COUNT(*) WHERE pl < 0",
        "عدد المشاريع التي عمود P&L (pl) الخاص بها أقل من صفر.", _kpi_losing_projects,
        ["pl < 0"],
    ),
    "active_projects": _registry_entry(
        "المشاريع النشطة", ["status"], "COUNT(*) WHERE status = 'Ongoing'",
        "عدد المشاريع ذات حالة Ongoing فقط.", _kpi_active_projects, ["status = 'Ongoing'"],
    ),
    "completed_projects": _registry_entry(
        "المشاريع المنتهية", ["status"], "COUNT(*) WHERE status IN ('Completed', 'Closed')",
        "عدد المشاريع المكتملة أو المغلقة.", _kpi_completed_projects,
        ["status IN ('Completed', 'Closed')"],
    ),
    "contracts_expiring_soon": _registry_entry(
        "العقود المنتهية قريبًا", ["amended_end_date", "end_date"],
        "COUNT(*) WHERE 0 <= (COALESCE(amended_end_date, end_date) - today) < 30",
        "عدد العقود التي ينتهي تاريخها الفعلي خلال أقل من 30 يومًا.",
        _kpi_contracts_expiring_soon, ["0 <= days_remaining < 30"],
    ),
    "amendments_total": _registry_entry(
        "إجمالي تعديلات العقود", ["amendment_crs"], "SUM(amendment_crs)",
        "مجموع قيمة تعديلات العقود.", _kpi_amendments_total,
    ),
    "current_year_revenue": _registry_entry(
        "إيرادات الفترة الحالية", ["revenue_current"], "SUM(revenue_current)",
        "مجموع الإيرادات المسجلة للفترة الحالية.", _kpi_current_year_revenue,
    ),
    "current_year_cost": _registry_entry(
        "تكاليف الفترة الحالية", ["cost_of_revenue", "other_cost"],
        "SUM(cost_of_revenue) + SUM(other_cost)",
        "مجموع تكلفة الإيرادات والتكاليف الأخرى للفترة الحالية.", _kpi_current_year_cost,
    ),
}


def compute_kpi(name: str, projects: Iterable[Mapping[str, Any]], today: date | None = None) -> Any:
    """Compute a single named KPI from the registry. Raises KeyError if the
    KPI name is not registered — callers must never fall back to ad-hoc math."""
    entry = KPI_REGISTRY[name]
    return entry["compute"](list(projects), today=today)


def summarize_by_bu(projects: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group portfolio totals by business unit. Single implementation used
    by VP reports and the dashboard — must not be re-derived elsewhere."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for project in projects:
        bu = str(project.get("bu") or "غير محدد")
        groups.setdefault(bu, []).append(project)

    summary = []
    for bu, items in groups.items():
        revenue = sum(_number(p.get("total_revenue")) for p in items)
        cost = sum(_number(p.get("total_cost")) for p in items)
        summary.append({
            "bu": bu,
            "count": len(items),
            "total_contract_value": sum(_number(p.get("total_contract_value")) for p in items),
            "total_revenue": revenue,
            "backlog": sum(_number(p.get("backlog")) for p in items),
            "net_profit": revenue - cost,
        })
    return sorted(summary, key=lambda x: -x["total_contract_value"])
