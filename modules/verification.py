"""
modules/verification.py
------------------------
Section 21 — Verification Layer. Runs after modules.query_executor and
before any answer is formatted for the user. Nothing here calls the LLM.

Fix (see audit history): the previous version only checked that returned
project_code values currently exist in the database. That does NOT catch
an executor bug that returns the wrong-but-real subset of projects (wrong
filter applied, wrong sort order, limit applied at the wrong stage) --
every code in a wrong-but-real result set still "exists", so the old check
would pass it through untouched.

This version independently re-derives, from a fresh database read, which
project_codes -- in which order -- the validated spec SHOULD have
produced (using the same filter predicate as modules.query_executor, but
never reusing modules.query_executor's own chosen rows as the basis for
comparison), and requires an exact match against what was actually
returned. It also re-derives the aggregation the same independent way.

Checks performed:
  1. The spec is re-validated against the whitelist.
  2. The exact ordered list of project_codes that should have been
     returned (fresh read -> filters -> sort -> limit, applied
     independently here) must match what was actually returned, in order.
  3. Any aggregation result is recomputed independently over the
     independently-derived row set (respecting `aggregate_over_limited_rows`
     exactly as modules.query_executor does) and must match.
  4. If a named KPI formula was used, it must be a formula that is
     actually registered in modules.kpi_calculator.KPI_REGISTRY.

If any check fails, the caller must not send the answer — it must send the
fixed fallback message instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from modules import query_schema
from modules.kpi_calculator import KPI_REGISTRY
from modules.project_repository import fetch_enriched_projects, fetch_project_codes

FALLBACK_MESSAGE = "لا تتوفر لدي بيانات موثوقة كافية للإجابة الآن. جرّب إعادة صياغة السؤال."
PROJECT_FALLBACK_MESSAGE = "لم أتمكن من التحقق من المشروع المقصود. جرّب كتابة اسم أو رمز أوضح."


@dataclass
class VerificationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _numbers_match(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 0.01) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def _independently_derive_rows(
    spec: dict[str, Any], fresh_projects: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-apply filters, sort and limit from scratch against a fresh read.
    Returns (rows_for_aggregation, rows_for_display) -- these differ only
    when `aggregate_over_limited_rows` is set (an explicit top/bottom-N
    aggregation request), matching modules.query_executor.execute()'s own
    semantics exactly, but computed here independently."""
    # Local import to avoid a module import cycle at load time; this reuses
    # the shared filter predicate, not modules.query_executor's own chosen
    # rows, so it remains an independent re-derivation.
    from modules.query_executor import _matches

    rows = [p for p in fresh_projects if all(_matches(p, f) for f in spec["filters"])]

    sort = spec.get("sort")
    if sort:
        col = sort["column"]
        reverse = sort["direction"] == "DESC"
        with_value = [p for p in rows if p.get(col) is not None]
        without_value = [p for p in rows if p.get(col) is None]
        with_value.sort(key=lambda p: p[col], reverse=reverse)
        rows = with_value + without_value

    limit = spec.get("limit")
    aggregate_over_limited_rows = spec.get("aggregate_over_limited_rows", False)
    aggregation = spec.get("aggregation")

    if aggregation and limit and aggregate_over_limited_rows:
        rows_for_aggregation = rows[:limit]
        rows_for_display = rows_for_aggregation
    else:
        rows_for_aggregation = rows
        rows_for_display = rows[:limit] if limit else rows

    return rows_for_aggregation, rows_for_display


def verify(execution_result: dict[str, Any], today: date | None = None) -> VerificationResult:
    """Verify a modules.query_executor.execute() result before it is
    allowed to reach the LLM formatter or the user."""
    reasons: list[str] = []

    spec = execution_result.get("spec")
    if spec is None:
        return VerificationResult(ok=False, reasons=["missing query spec on execution result"])

    try:
        query_schema.validate_query_spec(spec)
    except query_schema.QueryValidationError as exc:
        return VerificationResult(ok=False, reasons=[f"spec failed whitelist re-check: {exc}"])

    returned_rows = execution_result.get("rows", [])
    returned_codes = [p.get("project_code") for p in returned_rows]

    fresh_projects = fetch_enriched_projects(today=today)
    rows_for_aggregation, rows_for_display = _independently_derive_rows(spec, fresh_projects)
    expected_codes = [p.get("project_code") for p in rows_for_display]

    if returned_codes != expected_codes:
        reasons.append(
            "returned rows do not match an independent re-derivation of the spec's "
            f"filters/sort/limit -- expected {len(expected_codes)} project(s) "
            f"({expected_codes[:10]}{'...' if len(expected_codes) > 10 else ''}), "
            f"got {len(returned_codes)} ({returned_codes[:10]}{'...' if len(returned_codes) > 10 else ''})"
        )

    agg = execution_result.get("aggregation")
    if agg is not None:
        func, col = agg["func"], agg.get("column")
        from modules.query_executor import _AGG_TO_KPI_NAME, _aggregate

        kpi_name = _AGG_TO_KPI_NAME.get((func, col))
        if kpi_name is not None and kpi_name not in KPI_REGISTRY:
            reasons.append(f"aggregation references unregistered KPI: {kpi_name!r}")
        else:
            recomputed = _aggregate(rows_for_aggregation, agg) if rows_for_aggregation or func == "COUNT" else 0
            reported = execution_result.get("aggregation_result")
            if not _numbers_match(recomputed, reported):
                reasons.append(
                    f"aggregation mismatch: reported={reported!r} recomputed={recomputed!r}"
                )

    if reasons:
        return VerificationResult(ok=False, reasons=reasons)
    return VerificationResult(ok=True)


def verify_project_lookup(
    execution_result: dict[str, Any], resolution: Any, today: date | None = None
) -> VerificationResult:
    """Apply stricter invariants to a resolved single-project lookup."""
    reasons: list[str] = []

    if getattr(resolution, "status", None) != "matched":
        reasons.append("project resolution is not a high-confidence match")
        return VerificationResult(ok=False, reasons=reasons)

    expected_code = getattr(resolution, "canonical_project_code", None)
    rows = execution_result.get("rows") or []
    spec = execution_result.get("spec") or {}
    try:
        query_schema.validate_query_spec(spec)
    except query_schema.QueryValidationError as exc:
        reasons.append(f"project query spec failed validation: {exc}")
    if not expected_code:
        reasons.append("resolved project has no canonical project_code")
    elif expected_code not in fetch_project_codes(today=today):
        reasons.append("resolved project_code no longer exists in the database")
    if spec.get("project_code") != expected_code:
        reasons.append("query spec does not contain the resolved canonical project_code")
    if len(rows) != 1:
        reasons.append(f"resolved lookup returned {len(rows)} rows instead of exactly one")
    elif rows[0].get("project_code") != expected_code:
        reasons.append("returned project_code differs from the resolved canonical code")
    else:
        from modules.project_entity_resolver import normalize_project_text

        candidate = resolution.candidates[0]
        expected_names = {
            normalize_project_text(candidate.project_name_ar),
            normalize_project_text(candidate.project_name_en),
        } - {""}
        returned_names = {
            normalize_project_text(rows[0].get("project_name_ar")),
            normalize_project_text(rows[0].get("project_name_en")),
        } - {""}
        if not expected_names.intersection(returned_names):
            reasons.append("returned project name is not one of the resolved canonical names")

    return VerificationResult(ok=not reasons, reasons=reasons)