"""
modules/query_compiler.py
--------------------------
Layer 3 of the semantic interpretation pipeline. Mechanical, not semantic:
by the time this runs, modules.semantic_interpreter (Layer 1) has already
classified intent and modules.entity_resolvers (Layer 2) has already
resolved every named entity to a real canonical DB value, or the
Interpretation would already carry requires_clarification=True and never
reach here. This module only assembles that already-resolved information
into a modules.query_schema-validated spec -- it never re-parses the
original question text, which is what let two independent parses of the
same sentence silently disagree (see the architecture plan).

Only PORTFOLIO_FILTER / PORTFOLIO_RANKING (-> a query_schema spec) and
GROUPED_ANALYTICS (-> a grouped_analytics entities dict) are compiled here.
Every other intent (KPI questions, project lookups, comparisons, contract
documents, executive analysis, small talk) is answered by the SAME
existing handlers as the original pipeline (modules.kpi_responder,
modules.project_entity_resolver, modules.comparison_engine,
modules.rag_engine, modules.executive_intelligence) -- this module does
not duplicate them, and modules.ai_engine._answer_inner_v2 dispatches to
those directly instead of routing them through here.
"""

from __future__ import annotations

from typing import Any

from modules import query_schema
from modules.intent_schema import GROUP_FIELDS, GROUPED_ANALYTICS, PORTFOLIO_FILTER, PORTFOLIO_RANKING
from modules.semantic_interpreter import Interpretation

_AGGREGATION_BY_METRIC = {
    "total_revenue": "sum_total_revenue",
    "total_cost": "sum_total_cost",
    "total_contract_value": "sum_total_contract_value",
    "pl": "sum_net_profit",
    "backlog": "sum_backlog",
}

# Entity kind -> the DB column it filters, in the fixed priority order
# filters are appended (stable, deterministic spec output).
_ENTITY_FILTER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "status"),
    ("departments", "dept"),
    ("business_units", "bu"),
    ("segments", "segment"),
    ("programs", "program"),
    ("managers", "project_manager"),
)


class CompileError(Exception):
    """Raised when an Interpretation cannot be compiled into a valid spec.
    Callers must treat this as a signal to fall back, never guess a spec."""


def _entity_filter(values: list[str], column: str) -> dict[str, Any]:
    if len(values) == 1:
        return {"column": column, "op": "==", "value": values[0]}
    return {"column": column, "op": "in", "value": list(values)}


def compile_query_spec(interpretation: Interpretation, *, default_rank_metric: str = "total_contract_value") -> dict:
    """Build a modules.query_schema-validated filter/sort/limit spec from
    already-resolved entities. Only valid for PORTFOLIO_FILTER/PORTFOLIO_RANKING."""
    if interpretation.intent not in {PORTFOLIO_FILTER, PORTFOLIO_RANKING}:
        raise CompileError(f"intent {interpretation.intent!r} is not spec-compilable")
    if interpretation.requires_clarification:
        raise CompileError("cannot compile a spec while clarification is still pending")

    entities = interpretation.entities
    filters: list[dict[str, Any]] = []
    for attr, column in _ENTITY_FILTER_COLUMNS:
        values = getattr(entities, attr)
        if values:
            filters.append(_entity_filter(values, column))

    spec: dict[str, Any] = {"filters": filters}
    operations = interpretation.operations

    if interpretation.intent == PORTFOLIO_RANKING:
        metric = entities.metrics[0] if entities.metrics else default_rank_metric
        direction = "ASC" if operations.sort == "asc" else "DESC"
        spec["sort"] = {"column": metric, "direction": direction}
        spec["limit"] = operations.limit or 1
    elif operations.limit:
        spec["limit"] = operations.limit

    try:
        return query_schema.validate_query_spec(spec)
    except query_schema.QueryValidationError as exc:
        raise CompileError(str(exc)) from exc


def compile_grouped_entities(interpretation: Interpretation) -> dict:
    """Build a modules.grouped_analytics.execute_grouped-compatible entities
    dict. Only valid for GROUPED_ANALYTICS."""
    if interpretation.intent != GROUPED_ANALYTICS:
        raise CompileError(f"intent {interpretation.intent!r} is not a grouped-analytics intent")
    if interpretation.requires_clarification:
        raise CompileError("cannot compile grouped entities while clarification is still pending")

    operations = interpretation.operations
    group_by = operations.group_by
    if group_by not in GROUP_FIELDS:
        raise CompileError(f"group_by {group_by!r} is not whitelisted")

    metric = interpretation.entities.metrics[0] if interpretation.entities.metrics else None
    aggregation = _AGGREGATION_BY_METRIC.get(metric, "count_distinct_projects")

    return {
        "group_by": group_by,
        "aggregation": aggregation,
        "sort_direction": "asc" if operations.sort == "asc" else "desc",
        "limit": operations.limit or 1,
    }
