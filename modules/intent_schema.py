"""
modules/intent_schema.py
Shared intent constants, Understanding dataclass, and whitelisted aggregation/group fields.
Single source of truth — nothing re-defines these elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Intent constants ─────────────────────────────────────────────────────────
PROJECT_SUMMARY        = "project_summary"
PROJECT_FOLLOWUP       = "project_followup"
PROJECT_COMPARISON     = "project_comparison"
PORTFOLIO_KPI          = "portfolio_kpi"
PROJECT_KPI            = "project_kpi"
PORTFOLIO_FILTER       = "portfolio_filter"
PORTFOLIO_RANKING      = "portfolio_ranking"
PORTFOLIO_SUMMARY      = "portfolio_summary"
EXEC_ATTENTION         = "executive_attention"
LIST_FOLLOWUP          = "list_followup"
CONTRACT_DOC           = "contract_document"
CONTRACT_VALUE         = "contract_value"
GROUPED_ANALYTICS      = "grouped_analytics"
SMALL_TALK             = "small_talk"
OUT_OF_SCOPE           = "out_of_scope"
CLARIFICATION_RESPONSE = "clarification_response"
PREVIOUS_RESULT_FOLLOWUP = "previous_result_follow_up"
CALCULATION_EXPLANATION  = "calculation_explanation"
EXECUTIVE_BRIEF          = "executive_brief"
EXECUTIVE_PRIORITY       = "executive_priority"
EXECUTIVE_RISK           = "executive_risk"
EXECUTIVE_OPPORTUNITY    = "executive_opportunity"
CONTROLLED_FALLBACK      = "controlled_fallback"

ALL_INTENTS = {
    PROJECT_SUMMARY, PROJECT_FOLLOWUP, PROJECT_COMPARISON,
    PORTFOLIO_KPI, PROJECT_KPI, PORTFOLIO_FILTER, PORTFOLIO_RANKING,
    PORTFOLIO_SUMMARY, EXEC_ATTENTION, LIST_FOLLOWUP,
    CONTRACT_DOC, CONTRACT_VALUE, GROUPED_ANALYTICS, SMALL_TALK, OUT_OF_SCOPE,
    CLARIFICATION_RESPONSE, PREVIOUS_RESULT_FOLLOWUP, CALCULATION_EXPLANATION,
    EXECUTIVE_BRIEF, EXECUTIVE_PRIORITY, EXECUTIVE_RISK, EXECUTIVE_OPPORTUNITY,
    CONTROLLED_FALLBACK,
}

# ── Whitelists ───────────────────────────────────────────────────────────────
AGGREGATIONS = frozenset({
    "count_distinct_projects",
    "sum_total_revenue",
    "sum_total_cost",
    "sum_total_contract_value",
    "sum_net_profit",
    "sum_backlog",
})

GROUP_FIELDS = frozenset({
    "project_manager", "bu", "segment", "dept", "status", "project_name_ar",
})

# ── Understanding dataclass ───────────────────────────────────────────────────
@dataclass
class Understanding:
    intent: str
    scope: str
    confidence: float = 1.0
    project_mentions: list[str] = field(default_factory=list)
    is_followup: bool = False
    requested_field: str | None = None
    requested_kpi: str | None = None
    list_followup_type: str | None = None
    comparison_projects: list[str] = field(default_factory=list)
    filter_intent: str | None = None
    ranking_intent: str | None = None
    group_by: str | None = None
    semantic_raw: str | None = None
    method: str = "unknown"
    entities: dict[str, Any] = field(default_factory=dict)


# ── Backward-compatibility aliases ───────────────────────────────────────────
# Some modules (query_schema, query_builder) import INTENTS or CONVERSATION_INTENTS
INTENTS = ALL_INTENTS
CONVERSATION_INTENTS = ALL_INTENTS


def validate_understanding(raw: dict) -> Understanding:
    """Build an Understanding from a raw dict, normalizing unknown intents."""
    intent = raw.get("intent", PROJECT_SUMMARY)
    if intent not in ALL_INTENTS:
        intent = PROJECT_SUMMARY
    return Understanding(
        intent=intent,
        scope=raw.get("scope", "unknown"),
        confidence=float(raw.get("confidence", 1.0)),
        project_mentions=raw.get("project_mentions") or [],
        is_followup=bool(raw.get("is_followup", False)),
        requested_field=raw.get("requested_field"),
        requested_kpi=raw.get("requested_kpi"),
        list_followup_type=raw.get("list_followup_type"),
        comparison_projects=raw.get("comparison_projects") or [],
        filter_intent=raw.get("filter_intent"),
        ranking_intent=raw.get("ranking_intent"),
        group_by=raw.get("group_by"),
        semantic_raw=raw.get("semantic_raw"),
        method=raw.get("method", "unknown"),
        entities=raw.get("entities") or {},
    )