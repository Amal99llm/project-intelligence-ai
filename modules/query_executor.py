"""
modules/query_executor.py
--------------------------
Executes an already-validated structured query spec (see
modules.query_schema.validate_query_spec) against the project database.

No LLM involvement anywhere in this module. Filtering is plain Python
comparisons over modules.project_repository's canonical project dicts (so
computed fields like net_profit/days_remaining are filterable exactly like
raw DB columns, with no separate SQL-vs-Python code path to drift). Any
requested aggregation is computed through modules.kpi_calculator's KPI
Registry wherever a named KPI applies — this module never redefines what
"revenue" or "profit" means.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from modules import query_schema
from modules.kpi_calculator import KPI_REGISTRY
from modules.project_repository import fetch_enriched_projects

_COMPARATORS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

# Aggregations that correspond to a named, registered executive KPI must be
# computed through that KPI's single formula rather than a generic sum.
_AGG_TO_KPI_NAME = {
    ("SUM", "total_revenue"): "revenue",
    ("SUM", "total_cost"): "cost",
    ("SUM", "backlog"): "backlog",
    ("SUM", "total_contract_value"): "total_contract_value",
    ("SUM", "net_profit"): "profit_loss",
}


def _matches(project: dict[str, Any], f: dict[str, Any]) -> bool:
    value = project.get(f["column"])
    is_string_column = f["column"] in query_schema.RAW_STRING_COLUMNS

    if f["op"] == "between":
        if value is None:
            return False
        return f["value"] <= value <= f["value2"]

    if f["op"] == "in":
        if value is None:
            return False
        if is_string_column:
            left = str(value).strip().lower()
            return left in {str(v).strip().lower() for v in f["value"]}
        return value in f["value"]

    if is_string_column:
        left = str(value or "").strip().lower()
        right = str(f["value"] or "").strip().lower()
        if f["op"] == "==":
            return left == right
        if f["op"] == "!=":
            return left != right
        if f["op"] == "contains":
            return right in left
        return _COMPARATORS[f["op"]](left, right)

    if value is None:
        return False
    return _COMPARATORS[f["op"]](value, f["value"])


def _aggregate(rows: list[dict[str, Any]], agg: dict[str, Any]) -> Any:
    func, col = agg["func"], agg.get("column")

    if func == "COUNT":
        return len(rows)

    kpi_name = _AGG_TO_KPI_NAME.get((func, col))
    if kpi_name is not None:
        return KPI_REGISTRY[kpi_name]["compute"](rows)

    values = [p[col] for p in rows if p.get(col) is not None]
    if func == "SUM":
        return sum(values)
    if func == "AVG":
        return sum(values) / len(values) if values else 0
    if func == "MAX":
        return max(values) if values else None
    if func == "MIN":
        return min(values) if values else None
    raise ValueError(f"Unsupported aggregation: {agg!r}")


def formula_for_aggregation(agg: dict[str, Any] | None) -> str | None:
    """Human-readable formula for source attribution (Section 17 Step 5).
    Prefers the KPI Registry's own formula text when the aggregation maps
    to a named KPI, so attribution never drifts from the actual formula."""
    if not agg:
        return None
    kpi_name = _AGG_TO_KPI_NAME.get((agg["func"], agg.get("column")))
    if kpi_name is not None:
        return KPI_REGISTRY[kpi_name]["formula"]
    return f"{agg['func']}({agg.get('column') or '*'})"


def execute(spec: dict[str, Any], today: date | None = None,
            projects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run a validated query spec. `projects` can be injected for testing;
    production callers omit it and it is fetched fresh from the database."""
    all_projects = projects if projects is not None else fetch_enriched_projects(today=today)

    rows = [p for p in all_projects if all(_matches(p, f) for f in spec["filters"])]

    if spec["sort"]:
        col = spec["sort"]["column"]
        reverse = spec["sort"]["direction"] == "DESC"
        with_value = [p for p in rows if p.get(col) is not None]
        without_value = [p for p in rows if p.get(col) is None]
        with_value.sort(key=lambda p: p[col], reverse=reverse)
        rows = with_value + without_value

    # Section 20 fix -- aggregation must cover every row matching the
    # filters by default. A `limit` only narrows what's aggregated when the
    # validated spec explicitly says so (modules.query_schema guarantees
    # `aggregate_over_limited_rows` is only ever true when both `limit` and
    # `sort` are present -- i.e. a genuine, explicit "top N" request).
    # Otherwise the limit is applied only to what's *displayed*, after
    # aggregation, never before it.
    aggregate_over_limited_rows = spec.get("aggregate_over_limited_rows", False)

    def limited_with_ties(source):
        limit = spec.get("limit")
        if not limit or len(source) <= limit or not spec.get("sort"):
            return source[:limit] if limit else source
        boundary = source[limit - 1].get(spec["sort"]["column"])
        end = limit
        while end < len(source) and source[end].get(spec["sort"]["column"]) == boundary:
            end += 1
        return source[:end]

    if spec["aggregation"] and spec["limit"] and aggregate_over_limited_rows:
        aggregated_rows = limited_with_ties(rows)
        display_rows = aggregated_rows
    elif spec["aggregation"]:
        aggregated_rows = rows
        display_rows = limited_with_ties(rows)
    else:
        aggregated_rows = rows
        display_rows = limited_with_ties(rows)

    aggregation_result = _aggregate(aggregated_rows, spec["aggregation"]) if spec["aggregation"] else None

    return {
        "spec": spec,
        "rows": display_rows,
        "row_count": len(display_rows),
        "matched_row_count": len(rows),
        "aggregation": spec["aggregation"],
        "aggregation_result": aggregation_result,
        "aggregated_row_count": len(aggregated_rows) if spec["aggregation"] else None,
        "exclusion_metadata": {
            "missing_effective_end_date": sum(
                1 for project in all_projects if project.get("effective_end_date") is None
            ),
        },
    }
