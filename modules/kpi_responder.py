"""Deterministic answers for direct KPI and KPI-metadata questions.

Fix (see PRODUCTION_SECURITY_TODO.md history / audit): this module used to
always compute a named KPI over the *entire* portfolio, even when the
question named a specific project ("how much profit margin does project X
have?"). It would answer with the portfolio-wide number, worded exactly
like a scoped answer, with nothing to signal the mismatch and no
verification layer underneath it to catch it. That was the single most
dangerous "confidently wrong" answer path in the system.

Now, before computing anything, this module checks whether the question
plausibly names a project (after removing the KPI phrase and generic
question words). If it does, it resolves that reference through the same
deterministic modules.project_entity_resolver used everywhere else in the
system, and:
  - a confident single match  -> compute the KPI over that project only,
    and say explicitly it is a project-level number.
  - an ambiguous/needs-confirmation match -> ask, never guess.
  - no match at all           -> say plainly the project wasn't found;
    NEVER silently fall back to a portfolio-wide number instead.
An explicit "for the whole portfolio" phrase always short-circuits to the
portfolio-wide answer without attempting project resolution.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from modules.kpi_calculator import KPI_REGISTRY, compute_kpi
from modules.project_entity_resolver import (
    format_resolution_prompt,
    normalize_project_text,
    resolve_project,
)
from modules.project_repository import fetch_enriched_projects
from modules.semantic_dictionary import KPI_ALIASES, METHODOLOGY_MARKERS


_ALIASES = {name: set(aliases) for name, aliases in KPI_ALIASES.items()}
def _definite_article_variants(alias: str) -> set[str]:
    """An Arabic alias stored with the definite article ('الإيرادات') must
    also match the same question phrased without it ('إيرادات مشروع...').
    Real bug this fixes: 'كم إيرادات مشروع الباحث الاجتماعي؟' (no 'ال' on
    'إيرادات') failed to match the 'revenue' alias at all, so the question
    silently fell through to a plain project lookup instead of being
    recognized as a KPI question -- losing the scope-detection fix
    entirely, not just misapplying it."""
    words = alias.split()
    stripped = " ".join(w[2:] if w.startswith("ال") and len(w) > 2 else w for w in words)
    return {alias, stripped}


_ALIASES = {
    name: set().union(*(_definite_article_variants(normalize_project_text(alias)) for alias in aliases))
    for name, aliases in _ALIASES.items()
}

_EXPLANATION_MARKERS = METHODOLOGY_MARKERS
_DIRECT_MARKERS = tuple(normalize_project_text(value) for value in (
    "كم", "ما هو", "ما هي", "ما اجمالي", "اعطني اجمالي", "what is", "how much",
))
# An explicit "for the whole portfolio" phrase always means portfolio scope
# -- skip project-resolution entirely rather than risk a fuzzy match on
# these generic words.
_PORTFOLIO_SCOPE_MARKERS = tuple(normalize_project_text(value) for value in (
    "محفظة", "المحفظة", "المحفظه", "كل المشاريع", "جميع المشاريع",
    "للمحفظة", "للمحفظه", "الكلي", "الاجمالي", "إجمالي",
    "لكل المشاريع", "لجميع المشاريع",
    "portfolio", "entire portfolio", "overall",
))
# Statistical/quantifier modifier words. These change HOW a KPI is framed
# (e.g. "average profit margin") but are not, by themselves, an attempt to
# name a project -- a real bug this fixes: "كم متوسط هامش الربح؟" ("what's
# the AVERAGE profit margin") was being treated as a failed project-name
# reference (leftover word "متوسط") and answered with "project not found"
# instead of the portfolio figure. None of these ever count as project-name
# signal on their own.
_QUANTIFIER_MODIFIER_MARKERS = tuple(normalize_project_text(value) for value in (
    "متوسط", "معدل", "بالمتوسط", "في المتوسط", "نسبه", "تقريبا", "تقريبي",
    "تقديري", "عموما", "بشكل عام", "average", "mean", "roughly", "approximately",
))

_MONEY_KPIS = {
    "total_contract_value", "revenue", "cost", "backlog", "profit_loss",
    "amendments_total", "current_year_revenue", "current_year_cost",
}
_COUNT_KPIS = {
    "total_projects", "active_projects", "losing_projects", "contracts_expiring_soon",
    "completed_projects",
}

_ENGLISH_DISPLAY = {
    "total_projects": "Total projects", "total_contract_value": "Total contract value",
    "revenue": "Total portfolio revenue", "cost": "Total portfolio cost",
    "backlog": "Portfolio backlog", "profit_loss": "Net profit",
    "profit_margin": "Profit margin", "active_projects": "Active projects",
    "losing_projects": "Losing projects", "completed_projects": "Completed projects",
    "contracts_expiring_soon": "Contracts expiring soon",
    "amendments_total": "Contract amendments", "current_year_revenue": "Current-period revenue",
    "current_year_cost": "Current-period cost",
}


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in str(text or ""))

# Tokens that never count as "this looks like a project name" leftovers --
# generic question words. Used to decide whether there's enough real
# content left in the question, after removing the KPI phrase, to justify
# attempting project resolution at all.
_NON_PROJECT_TOKENS: set[str] = set()
for _markers in (_EXPLANATION_MARKERS, _DIRECT_MARKERS, _PORTFOLIO_SCOPE_MARKERS,
                  _QUANTIFIER_MODIFIER_MARKERS):
    for _marker in _markers:
        _NON_PROJECT_TOKENS.update(_marker.split())
# "project"/"the project" with common attached Arabic prefixes (ل/ب/و/ف + ال)
# -- these attach without a space, so they never match project_entity_resolver's
# own bare-word stopword list either.
_NON_PROJECT_TOKENS.update(normalize_project_text(w) for w in (
    "مشروع", "المشروع", "لمشروع", "للمشروع", "بمشروع", "بالمشروع",
    "ومشروع", "والمشروع", "فمشروع", "فالمشروع",
))
_MIN_PROJECT_REFERENCE_LENGTH = 3


def identify_kpi(query: str) -> str | None:
    normalized = normalize_project_text(query)
    # Longer aliases win so "هامش الربح" is never confused with "الربح".
    matches = []
    for name, aliases in _ALIASES.items():
        for alias in aliases:
            if alias and alias in normalized:
                matches.append((len(alias), name))
    return max(matches, default=(0, None))[1]


def question_kind(query: str) -> tuple[str, str] | None:
    normalized = normalize_project_text(query)
    kpi_name = identify_kpi(query)
    if not kpi_name and "كم خسر" in normalized:
        kpi_name = "profit_loss"
    if not kpi_name:
        return None
    if any(marker in normalized for marker in _EXPLANATION_MARKERS):
        return "explanation", kpi_name
    if any(marker in normalized for marker in _DIRECT_MARKERS):
        return "value", kpi_name
    return None


def _has_explicit_portfolio_marker(normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in _PORTFOLIO_SCOPE_MARKERS)


def _project_reference_residual(query: str, kpi_name: str) -> str:
    """Whatever's left of the question after removing the matched KPI
    alias and generic question words -- used only to decide whether it's
    worth attempting project resolution at all."""
    normalized = normalize_project_text(query)
    for alias in sorted(_ALIASES.get(kpi_name, ()), key=len, reverse=True):
        if alias and alias in normalized:
            normalized = normalized.replace(alias, " ")
            break
    tokens = [t for t in normalized.split() if t not in _NON_PROJECT_TOKENS]
    return " ".join(tokens).strip()


def answer_kpi_for_known_project(
    kpi_name: str, project_code: str, project_display_name: str,
    today: date | None = None, projects: list[dict[str, Any]] | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Compute and format a named KPI for an already-resolved project --
    used both by the normal project-scoped path below and by
    modules.ai_engine when replaying an original KPI question after the
    user resolves a pending ambiguity/confirmation ("نعم" / "الثاني"),
    so that confirming a project doesn't silently discard what was
    actually asked and fall back to a generic project summary instead."""
    metadata = KPI_REGISTRY[kpi_name]
    pool = projects if projects is not None else fetch_enriched_projects(today=today)
    scoped_pool = [p for p in pool if p.get("project_code") == project_code]
    value = compute_kpi(kpi_name, scoped_pool, today=today)
    arabic = _is_arabic(query) if query else True
    if arabic:
        answer = f"{metadata['display_name']} لمشروع «{project_display_name}» هو {_format_value(kpi_name, value, True)}."
    else:
        answer = f"{_ENGLISH_DISPLAY.get(kpi_name, kpi_name)} for {project_display_name} is {_format_value(kpi_name, value, False)}."
    return {
        "answer": answer,
        "kpi_name": kpi_name,
        "value": value,
        "scope": "project",
        "project_code": project_code,
        "source": {
            "table": metadata["source_table"],
            "columns": metadata["source_columns"],
            "filters": metadata["filters"],
            "formula": metadata["formula"],
        },
    }


def _format_value(kpi_name: str, value: Any, arabic: bool = True) -> str:
    if kpi_name == "profit_margin":
        return f"{float(value):,.2f}%"
    if kpi_name in _MONEY_KPIS:
        amount = float(value)
        absolute = abs(amount)
        if absolute >= 1_000_000_000:
            compact = f"{amount / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
            return f"{compact} مليار ريال" if arabic else f"SAR {compact}B"
        if absolute >= 10_000_000:
            compact = f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".")
            return f"{compact} مليون ريال" if arabic else f"SAR {compact}M"
        return f"{amount:,.2f} ريال" if arabic else f"SAR {amount:,.2f}"
    if kpi_name in _COUNT_KPIS:
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def answer_kpi_question(
    query: str,
    today: date | None = None,
    projects: list[dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Answer a direct KPI value/explanation question.

    `projects` lets a caller that already fetched the enriched portfolio
    (e.g. modules.ai_engine) pass it in instead of triggering a second
    fresh read. `context` is an optional modules.session_context entry --
    used only to resolve an explanation follow-up ("which column was that
    computed from?") that names no KPI at all, via `last_kpi_name`.
    """
    classification = question_kind(query)

    if classification is None and context and context.get("last_kpi_name"):
        normalized = normalize_project_text(query)
        if any(marker in normalized for marker in _EXPLANATION_MARKERS) and not identify_kpi(query):
            classification = ("explanation", context["last_kpi_name"])

    if classification is None:
        return None

    kind, kpi_name = classification
    metadata = KPI_REGISTRY[kpi_name]

    if kind == "explanation":
        columns = "، ".join(metadata["source_columns"])
        answer = (
            f"{metadata['display_name']}: {metadata['formula']}. "
            f"المصدر: جدول {metadata['source_table']}، الأعمدة: {columns}. "
            f"{metadata['description']}"
        )
        return {
            "answer": answer,
            "kpi_name": kpi_name,
            "value": None,
            "scope": "formula",
            "source": {
                "table": metadata["source_table"],
                "columns": metadata["source_columns"],
                "filters": metadata["filters"],
                "formula": metadata["formula"],
            },
        }

    # kind == "value" -- decide portfolio vs. project scope before computing
    # anything. This is the fix: never silently answer with a portfolio
    # number when the question named a project.
    normalized_query = normalize_project_text(query)
    pool = projects if projects is not None else fetch_enriched_projects(today=today)

    project_code: str | None = None
    project_display_name: str | None = None

    if not _has_explicit_portfolio_marker(normalized_query):
        residual = _project_reference_residual(query, kpi_name)
        if len(residual) >= _MIN_PROJECT_REFERENCE_LENGTH:
            # Two-pass resolution: try the raw query first (this is what
            # lets an exact project code like "PRJ-003" match -- resolve_project's
            # code stage needs the original punctuation intact). If that
            # doesn't produce a confident match, retry against `residual`
            # (KPI phrase and generic question words stripped out) -- for
            # name-based fuzzy/token matching, that leftover noise can be
            # the difference between a clean match and an unnecessary
            # "did you mean?" confirmation.
            resolution = resolve_project(query, pool)
            cleaned_resolution = resolve_project(residual, pool)
            raw_is_exact = (
                resolution.status == "matched"
                and resolution.candidates
                and resolution.candidates[0].match_type.startswith("exact_")
            )
            # Exact code/name resolution always wins. Otherwise the cleaned
            # project phrase is more authoritative than a noisy raw-query
            # fuzzy winner. In particular, never collapse a cleaned
            # ambiguity to the first project selected from the noisy query.
            if not raw_is_exact:
                if cleaned_resolution.status in {"matched", "ambiguous"}:
                    resolution = cleaned_resolution
                elif resolution.status == "no_match" and cleaned_resolution.status == "confirmation":
                    resolution = cleaned_resolution
            if resolution.status == "matched":
                project_code = resolution.canonical_project_code
                project_display_name = resolution.candidates[0].display_name
            elif resolution.status in {"ambiguous", "confirmation"}:
                return {
                    "answer": format_resolution_prompt(resolution),
                    "kpi_name": kpi_name,
                    "value": None,
                    "scope": "needs_clarification",
                    "pending_candidates": [
                        {"project_code": c.project_code, "display_name": c.display_name}
                        for c in resolution.candidates
                    ],
                    "source": None,
                }
            elif resolution.status == "no_match":
                # A project reference was clearly attempted (residual text
                # survived stripping all generic words) but didn't resolve
                # to any real project. Say so plainly -- never fall back to
                # the portfolio-wide number here, that would just be the
                # same "confidently wrong" bug in a different shape.
                return {
                    "answer": format_resolution_prompt(resolution),
                    "kpi_name": kpi_name,
                    "value": None,
                    "scope": "not_found",
                    "source": None,
                }

    if project_code is not None:
        return answer_kpi_for_known_project(
            kpi_name, project_code, project_display_name, today, pool, query=query,
        )

    value = compute_kpi(kpi_name, pool, today=today)
    arabic = _is_arabic(query)
    if arabic:
        answer = f"{metadata['display_name']} عبر المحفظة هو {_format_value(kpi_name, value, True)}."
    else:
        answer = f"{_ENGLISH_DISPLAY.get(kpi_name, kpi_name)} is {_format_value(kpi_name, value, False)}."
    return {
        "answer": answer,
        "kpi_name": kpi_name,
        "value": value,
        "scope": "portfolio",
        "project_code": None,
        "source": {
            "table": metadata["source_table"],
            "columns": metadata["source_columns"],
            "filters": metadata["filters"],
            "formula": metadata["formula"],
        },
    }