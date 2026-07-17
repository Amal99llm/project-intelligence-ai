"""
modules/query_schema.py
------------------------
Single source of truth for what the Structured Query Builder is allowed to
ask for. Used both to build the LLM function-calling JSON schema and to
validate whatever the LLM returns before anything touches the database.

No arithmetic and no query execution happens here — this module only knows
what is *allowed*, never what the data actually is.
"""

from __future__ import annotations

from typing import Any
from modules.intent_schema import INTENTS as CONVERSATION_INTENTS

# ── Intent categories (semantic, not keyword-based) ────────────────────────

INTENT_EXECUTIVE_KPI = "executive_kpi"
INTENT_PROJECT_LOOKUP = "project_lookup"
INTENT_FINANCIAL_ANALYSIS = "financial_analysis"
INTENT_RANKING = "ranking"
INTENT_COMPARISON = "comparison"
INTENT_FILTERING = "filtering"
INTENT_CONTRACT_ANALYSIS = "contract_analysis"
INTENT_GENERAL_CONVERSATION = "general_conversation"

INTENT_CATEGORIES = {
    INTENT_EXECUTIVE_KPI,
    INTENT_PROJECT_LOOKUP,
    INTENT_FINANCIAL_ANALYSIS,
    INTENT_RANKING,
    INTENT_COMPARISON,
    INTENT_FILTERING,
    INTENT_CONTRACT_ANALYSIS,
    INTENT_GENERAL_CONVERSATION,
}

# Intents that go through the structured query pipeline (DB-backed).
# contract_analysis routes to rag_engine; general_conversation skips the DB.
STRUCTURED_QUERY_INTENTS = {
    INTENT_EXECUTIVE_KPI,
    INTENT_PROJECT_LOOKUP,
    INTENT_FINANCIAL_ANALYSIS,
    INTENT_RANKING,
    INTENT_COMPARISON,
    INTENT_FILTERING,
}

# ── Columns ──────────────────────────────────────────────────────────────
# Raw columns exist directly on BacklogProject and can be filtered in SQL.
# Computed columns are derived (never stored) and must be filtered in Python
# using modules.kpi_calculator, which is the single source of truth for them.

RAW_NUMERIC_COLUMNS = {
    "contract_value", "amendment_crs", "total_contract_value", "pl",
    "previous_years_rev", "revenue_current", "other_income", "total_revenue",
    "backlog", "previous_years_cost", "cost_of_revenue", "other_cost", "total_cost",
    "progress_completed", "acc_rev", "pb", "adv", "ar", "contract_assets", "ap",
    "acc_exp", "contract_liabilities", "deferred_cost", "open_po", "ecl_ar",
    "ecl_acc_rev", "etc_cost", "etc_revenue",
}

RAW_STRING_COLUMNS = {
    "status", "bu", "segment", "dept", "project_manager", "customer_id",
    "project_code", "project_name_en", "project_name_ar",
}

RAW_DATE_COLUMNS = {"start_date", "end_date", "amended_end_date"}

# name -> human-readable formula, purely documentary; the actual computation
# always lives in modules.kpi_calculator.
COMPUTED_COLUMNS = {
    "net_profit": "total_revenue - total_cost",
    "profit_pct": "(total_revenue - total_cost) / total_revenue * 100",
    "effective_end_date": "amended_end_date if set else end_date",
    "days_remaining": "effective_end_date - today",
}

FILTERABLE_COLUMNS = RAW_NUMERIC_COLUMNS | RAW_STRING_COLUMNS | RAW_DATE_COLUMNS | set(COMPUTED_COLUMNS)
SORTABLE_COLUMNS = FILTERABLE_COLUMNS
AGGREGATABLE_COLUMNS = RAW_NUMERIC_COLUMNS | {"net_profit", "profit_pct", "project_code"}

# ── Operators / aggregations ────────────────────────────────────────────

OPERATORS = {"<", "<=", ">", ">=", "==", "!=", "between", "contains", "in"}
MAX_IN_VALUES = 20
AGGREGATIONS = {"SUM", "COUNT", "AVG", "MAX", "MIN"}
SORT_DIRECTIONS = {"ASC", "DESC"}

MAX_LIMIT = 500
DEFAULT_LIMIT = 50


class QueryValidationError(Exception):
    """Raised when a query spec references anything outside the whitelist."""


def _validate_filter(f: dict, errors: list[str]) -> dict | None:
    if not isinstance(f, dict):
        errors.append(f"Filter is not an object: {f!r}")
        return None
    column = f.get("column")
    op = f.get("op")
    if column not in FILTERABLE_COLUMNS:
        errors.append(f"Column not allowed for filtering: {column!r}")
        return None
    if op not in OPERATORS:
        errors.append(f"Operator not allowed: {op!r}")
        return None
    if op == "contains" and column not in RAW_STRING_COLUMNS:
        errors.append(f"'contains' only allowed on text columns, not {column!r}")
        return None
    if op == "between":
        if "value" not in f or "value2" not in f:
            errors.append(f"'between' requires value and value2 on column {column!r}")
            return None
        return {"column": column, "op": op, "value": f["value"], "value2": f["value2"]}
    if op == "in":
        values = f.get("value")
        if not isinstance(values, list) or not values:
            errors.append(f"'in' requires a non-empty list value on column {column!r}")
            return None
        if len(values) > MAX_IN_VALUES:
            errors.append(f"'in' list too long on column {column!r} (max {MAX_IN_VALUES})")
            return None
        return {"column": column, "op": op, "value": values}
    if "value" not in f:
        errors.append(f"Filter on {column!r} is missing 'value'")
        return None
    return {"column": column, "op": op, "value": f["value"]}


def validate_query_spec(spec: dict) -> dict:
    """Validate a structured query spec produced by the query builder.

    Returns a cleaned spec containing only whitelisted fields, operators,
    aggregations and sort columns. Raises QueryValidationError listing every
    problem found if the spec references anything outside the whitelist.
    """
    if not isinstance(spec, dict):
        raise QueryValidationError("Query spec must be an object")

    errors: list[str] = []
    cleaned: dict[str, Any] = {
        "filters": [], "sort": None, "limit": None, "aggregation": None,
    }

    # Project-specific queries carry the resolver's canonical identifier at
    # the top level.  It is never accepted as free text from the LLM.  The
    # exact filter is added here so the executor cannot accidentally widen a
    # resolved lookup to the full portfolio.
    canonical_code = spec.get("project_code")
    if canonical_code is not None:
        if not isinstance(canonical_code, str) or not canonical_code.strip():
            errors.append("project_code must be a non-empty canonical identifier")
        else:
            canonical_code = canonical_code.strip()
            cleaned["project_code"] = canonical_code

    for f in spec.get("filters") or []:
        validated = _validate_filter(f, errors)
        if validated is not None:
            cleaned["filters"].append(validated)

    if canonical_code and isinstance(canonical_code, str):
        code_filters = [
            f for f in cleaned["filters"]
            if f["column"] == "project_code" and f["op"] == "=="
        ]
        if code_filters and any(str(f["value"]).casefold() != canonical_code.casefold() for f in code_filters):
            errors.append("project_code conflicts with its exact filter")
        elif not code_filters:
            cleaned["filters"].append({"column": "project_code", "op": "==", "value": canonical_code})

    sort = spec.get("sort")
    if sort:
        s_col = sort.get("column")
        s_dir = (sort.get("direction") or "DESC").upper()
        if s_col not in SORTABLE_COLUMNS:
            errors.append(f"Column not allowed for sorting: {s_col!r}")
        elif s_dir not in SORT_DIRECTIONS:
            errors.append(f"Sort direction not allowed: {s_dir!r}")
        else:
            cleaned["sort"] = {"column": s_col, "direction": s_dir}

    limit = spec.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            errors.append(f"Limit is not an integer: {limit!r}")
        else:
            if limit <= 0:
                errors.append(f"Limit must be positive: {limit!r}")
            else:
                cleaned["limit"] = min(limit, MAX_LIMIT)

    aggregation = spec.get("aggregation")
    if aggregation:
        a_func = aggregation.get("func")
        a_col = aggregation.get("column")
        if a_func not in AGGREGATIONS:
            errors.append(f"Aggregation function not allowed: {a_func!r}")
        elif a_func != "COUNT" and a_col not in AGGREGATABLE_COLUMNS:
            errors.append(f"Column not allowed for aggregation: {a_col!r}")
        else:
            cleaned["aggregation"] = {"func": a_func, "column": a_col}

    # limit + aggregation interaction: by default an aggregation always
    # covers every row matching the filters, never a limited subset -- a
    # portfolio-wide "total revenue" question must never silently become
    # "total revenue of the first N rows" just because a limit happened to
    # be attached to the same spec. A limited aggregation ("total revenue
    # of the top 10 projects by backlog") is only honored when the caller
    # explicitly asks for it via `aggregate_over_limited_rows: true` --
    # otherwise the limit is dropped for aggregation purposes (the spec
    # itself is not rejected, since dropping the limit is always the safe,
    # more-inclusive choice).
    explicit_top_n = bool(spec.get("aggregate_over_limited_rows"))
    if cleaned["aggregation"] and cleaned["limit"]:
        if explicit_top_n and cleaned["sort"] is not None:
            cleaned["aggregate_over_limited_rows"] = True
        elif explicit_top_n:
            cleaned["limit"] = None
            cleaned["aggregate_over_limited_rows"] = False
    elif explicit_top_n:
        cleaned["aggregate_over_limited_rows"] = False

    if errors:
        raise QueryValidationError("; ".join(errors))

    return cleaned
