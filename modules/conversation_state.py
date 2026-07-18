"""Canonical, atomic conversation-state transitions.

Handlers may still emit compatibility keys while they are migrated.  This
module is the sole production boundary that converts a completed turn into
the canonical state contract.  ``None`` never overwrites a valid reference.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SCOPES = {"portfolio", "project", "comparison", None}

CANONICAL_FIELDS = (
    "last_scope", "last_selected_project_id", "last_selected_project_name",
    "last_ranked_project_id", "last_ranked_project_name",
    "last_compared_project_ids", "last_compared_project_names",
    "pending_disambiguation_options", "pending_disambiguation_intent",
    "pending_comparison_side", "last_user_intent",
    "last_successful_result_type", "last_lookup_succeeded", "last_metric",
    "last_rank_direction",
)


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in values.items() if value is not None}


def set_portfolio_context(intent: str, result_type: str, metric: str | None = None) -> dict:
    return _without_none({"last_scope": "portfolio", "last_user_intent": intent,
                          "last_successful_result_type": result_type,
                          "last_metric": metric, "last_lookup_succeeded": True})


def set_active_project(project_id: str, name: str | None, intent: str,
                       result_type: str, metric: str | None = None) -> dict:
    return _without_none({"last_scope": "project", "last_selected_project_id": project_id,
                          "last_selected_project_name": name, "last_user_intent": intent,
                          "last_successful_result_type": result_type,
                          "last_lookup_succeeded": True, "last_metric": metric})


def set_ranked_project(project_id: str, name: str, intent: str, result_type: str,
                       metric: str | None, direction: str | None) -> dict:
    state = set_active_project(project_id, name, intent, result_type, metric)
    state.update(_without_none({"last_ranked_project_id": project_id,
                                "last_ranked_project_name": name,
                                "last_rank_direction": direction}))
    return state


def set_comparison_context(ids: list[str], names: list[str], intent: str,
                           metric: str | None = None) -> dict:
    return _without_none({"last_scope": "comparison", "last_compared_project_ids": ids,
                          "last_compared_project_names": names,
                          "last_user_intent": intent,
                          "last_successful_result_type": "comparison",
                          "last_lookup_succeeded": True, "last_metric": metric})


def set_pending_disambiguation(options: list[dict], intent: str,
                               comparison_side: int | None = None) -> dict:
    return _without_none({"pending_disambiguation_options": options,
                          "pending_disambiguation_intent": intent,
                          "pending_comparison_side": comparison_side})


def clear_pending_disambiguation() -> dict:
    return {"pending_disambiguation_options": [], "pending_disambiguation_intent": None,
            "pending_comparison_side": None}


def record_failed_lookup() -> dict:
    # Preserve the last valid project/portfolio/comparison reference.  The
    # failed text is never promoted into state.
    return {"last_lookup_succeeded": False}


def clear_project_context() -> dict:
    return {"last_selected_project_id": None, "last_selected_project_name": None,
            "last_ranked_project_id": None, "last_ranked_project_name": None}


def reset_conversation_context() -> dict:
    return {key: ([] if key in {"last_compared_project_ids", "last_compared_project_names",
                                "pending_disambiguation_options"} else None)
            for key in CANONICAL_FIELDS}


def transition_for_turn(current: dict, query_type: str, updates: dict | None) -> dict:
    """Translate one successful handler result into one atomic canonical update."""
    raw = updates or {}
    project_id = raw.get("last_selected_project_id") or raw.get("active_project_code")
    project_name = raw.get("last_selected_project_name") or raw.get("active_project_display_name")
    compared = raw.get("last_compared_project_ids") or raw.get("comparison_project_ids")
    result_type = raw.get("last_successful_result_type") or raw.get("last_result_type") or query_type
    metric = raw.get("last_metric") or raw.get("last_requested_metric")
    scope = raw.get("last_scope")
    legacy_scope = raw.get("last_result_scope")
    if scope is None and legacy_scope in {"portfolio", "list"}: scope = "portfolio"
    if scope is None and legacy_scope in {"project", "comparison"}: scope = legacy_scope

    state: dict[str, Any] = {"last_user_intent": query_type,
                             "last_successful_result_type": result_type}
    if compared:
        names = raw.get("last_compared_project_names") or []
        state.update(set_comparison_context(list(compared), list(names), query_type, metric))
    elif project_id:
        state.update(set_active_project(project_id, project_name, query_type, result_type, metric))
    elif scope == "portfolio":
        state.update(set_portfolio_context(query_type, result_type, metric))
    if raw.get("last_ranked_project_id"):
        direction = raw.get("last_rank_direction")
        state.update(set_ranked_project(raw["last_ranked_project_id"],
                                        raw.get("last_ranked_project_name") or project_name or raw["last_ranked_project_id"],
                                        query_type, result_type, metric, direction))
    pending = raw.get("pending_project_confirmation")
    if pending:
        state.update(set_pending_disambiguation(list(pending.get("candidates") or []),
                                                pending.get("kind") or query_type,
                                                pending.get("comparison_side")))
    elif "pending_project_confirmation" in raw:
        state.update(clear_pending_disambiguation())
    if raw.get("last_lookup_succeeded") is False:
        state.update(record_failed_lookup())
    return state
