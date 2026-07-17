"""Deterministic decomposition of one message into executable operations.

Understanding may use an LLM, but this plan contains only whitelisted project
identifiers and metric operations.  It never produces an answer or executes IO.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from modules.project_entity_resolver import resolve_project
from modules.semantic_dictionary import detect_requested_fields, normalize_text


@dataclass(frozen=True)
class PlanStep:
    operation: str
    project_code: str | None = None
    metrics: tuple[str, ...] = ()


@dataclass
class QueryPlan:
    steps: list[PlanStep] = field(default_factory=list)
    project_codes: list[str] = field(default_factory=list)

    @property
    def is_compound(self) -> bool:
        return len(self.steps) > 1 or any(len(step.metrics) > 1 for step in self.steps)


def build_plan(query: str, projects: list[dict], active_project_id: str | None = None) -> QueryPlan:
    """Build a safe plan from explicit entities plus all detected metrics."""
    resolution = resolve_project(query, projects)
    codes: list[str] = []
    if resolution.status == "matched" and resolution.canonical_project_code:
        codes.append(resolution.canonical_project_code)

    # The comparison resolver is deliberately reused: it understands two
    # names in one utterance and returns canonical database rows.
    from modules.comparison_engine import resolve_comparison_projects
    comparison_rows = resolve_comparison_projects(query, projects)
    for row in comparison_rows:
        code = row.get("project_code")
        if code and code not in codes:
            codes.append(code)

    if not codes and active_project_id:
        codes = [active_project_id]

    detected_metrics = [item.canonical for item in detect_requested_fields(query)]
    # Deterministic bilingual vocabulary supplements the semantic registry for
    # conjunction-heavy English requests; it adds operations, never answers.
    q_lower = query.casefold()
    supplements = (
        ("pl", ("net profit", " profit", "profit,")),
        ("profit_pct", ("margin", "profit percentage")),
        ("effective_end_date", ("end date", "finish date", "when does", "when will")),
        ("days_remaining", ("time left", "days left", "remaining time")),
        ("project_manager", ("manager", "project lead")),
    )
    for canonical, markers in supplements:
        if any(marker in q_lower for marker in markers):
            detected_metrics.append(canonical)
    metrics = tuple(dict.fromkeys(detected_metrics))
    steps: list[PlanStep] = []
    target = codes[0] if codes else None
    if target and metrics:
        steps.append(PlanStep("get_metrics", target, metrics))
    elif target:
        steps.append(PlanStep("get_summary", target))
    q = normalize_text(query)
    if len(codes) >= 2 or any(word in q for word in ("قارن", "compare")):
        steps.append(PlanStep("compare", metrics=metrics))
    return QueryPlan(steps=steps, project_codes=codes)
