"""
modules/project_repository.py
------------------------------
The one place that reads backlog_projects out of the database and turns
each row into the canonical project dict used everywhere downstream: the
dashboard API, the KPI Registry, and the structured query executor.

Previously this row-shaping logic existed twice, independently, in
app.py's /api/projects and in ai_engine.py's context builder — the two
implementations had already drifted (different column sets). This module
replaces both.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from modules.database import BacklogProject, get_session
from modules.kpi_calculator import normalize_progress, project_financials
from modules.time_utils import riyadh_today


def _to_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _value(value: Any) -> Any:
    """Preserve NULL as missing and zero as a real verified value."""
    return None if value is None else value


def _percentage(value: Any) -> float | None:
    return None if value is None else float(value) * 100


def fetch_enriched_projects(today: date | None = None) -> list[dict[str, Any]]:
    """Fetch every project row, enriched with the calculated fields
    (net_profit, profit_pct, effective_end_date, days_remaining) that the
    KPI Registry and structured query executor rely on. Stored profit
    fields on the row are never used — profit is always recalculated from
    revenue and cost here, once."""
    today = today or riyadh_today()
    with get_session() as session:
        rows = session.query(BacklogProject).all()
        projects = []
        for r in rows:
            end = _to_date(r.amended_end_date) or _to_date(r.end_date)
            days_rem = (end - today).days if end else None
            # Guard: if end date is suspiciously old (pre-2000) treat as no date
            if end and end.year < 2000:
                end = None
                days_rem = None
            project = {
                "project_type": r.project_type,
                "category": r.category,
                "program": r.program,
                "project_code": r.project_code,
                "wbs_pc": r.wbs_pc,
                "wbs": r.wbs,
                "project_name_en": r.project_name_en,
                "project_name_ar": r.project_name_ar,
                "pc": r.pc,
                "cc": r.cc,
                "bu": r.bu,
                "segment": r.segment,
                "dept": r.dept,
                "status": r.status,
                "progress_completed": normalize_progress(r.progress_completed),
                "start_date": str(r.start_date) if r.start_date else None,
                "end_date": str(r.end_date) if r.end_date else None,
                "amended_end_date": str(r.amended_end_date) if r.amended_end_date else None,
                "effective_end_date": str(end) if end else None,
                "days_remaining": days_rem,
                "contract_value": _value(r.contract_value),
                "amendment_crs": _value(r.amendment_crs),
                "total_contract_value": _value(r.total_contract_value),
                "previous_years_rev": _value(r.previous_years_rev),
                "revenue_current": _value(r.revenue_current),
                "other_income": _value(r.other_income),
                "total_revenue": _value(r.total_revenue),
                "backlog": _value(r.backlog),
                "previous_years_cost": _value(r.previous_years_cost),
                "cost_of_revenue": _value(r.cost_of_revenue),
                "other_cost": _value(r.other_cost),
                "total_cost": _value(r.total_cost),
                "pl": _value(r.pl),
                "pm_up_to_2025": _value(r.pm_up_to_2025),
                "pm_pct_up_to_2025": _percentage(r.pm_pct_up_to_2025),
                "gp_2026": _value(r.gp_2026),
                "pm_2026": _value(r.pm_2026),
                "planned_profit": _value(r.planned_profit),
                "planned_pm_pct": _percentage(r.planned_pm_pct),
                "variance": _percentage(r.variance),
                "pm_pct_2026": _percentage(r.pm_pct_2026),
                "po": _value(r.po),
                "hr": _value(r.hr),
                "other_external": _value(r.other_external),
                "other_internal": _value(r.other_internal),
                "risk": _value(r.risk),
                "contingency": _value(r.contingency),
                "total_planned_cost": _value(r.total_planned_cost),
                "net_etc": _value(r.net_etc),
                "etc_pct": _percentage(r.etc_pct),
                "acc_rev": _value(r.acc_rev),
                "pb": _value(r.pb),
                "adv": _value(r.adv),
                "ar": _value(r.ar),
                "contract_assets": _value(r.contract_assets),
                "ap": _value(r.ap),
                "acc_exp": _value(r.acc_exp),
                "contract_liabilities": _value(r.contract_liabilities),
                "deferred_cost": _value(r.deferred_cost),
                "open_po": _value(r.open_po),
                "ecl_ar": _value(r.ecl_ar),
                "ecl_acc_rev": _value(r.ecl_acc_rev),
                "etc_cost": _value(r.etc_cost),
                "etc_revenue": _value(r.etc_revenue),
                "customer_id": r.customer_id,
                "project_manager": r.project_manager,
                "officer_name": r.officer_name,
                "support_document": r.support_document,
                "note": r.note,
            }
            profit, margin = project_financials(project)
            project["net_profit"] = profit
            project["profit_pct"] = margin
            project["expected_progress_pct"] = None
            project["progress_gap"] = None
            start = _to_date(r.start_date)
            if start and end and end > start and project["progress_completed"] is not None:
                elapsed = max(0.0, min(1.0, (today - start).days / (end - start).days))
                project["expected_progress_pct"] = elapsed * 100
                project["progress_gap"] = project["expected_progress_pct"] - project["progress_completed"]
            projects.append(project)
    return projects


def fetch_project_codes(today: date | None = None) -> set[str]:
    """Cheap existence check used by the verification layer — does not
    require the full enrichment pass."""
    with get_session() as session:
        rows = session.query(BacklogProject.project_code).all()
        return {code for (code,) in rows if code}
