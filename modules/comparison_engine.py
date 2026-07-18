"""Deterministic two-project resolution and verified comparison wording."""

from __future__ import annotations

import re
from typing import Any

from modules.project_entity_resolver import normalize_project_text, resolve_project
from modules.project_name_index import resolve_multiple_project_phrases, MultipleProjectResolution
from modules.semantic_dictionary import FIELDS, detect_requested_field


_COMPARE_MARKERS = tuple(normalize_project_text(value) for value in (
    "قارن", "مقارنة", "مقارنه", "الفرق بين", "مقابل", "compare", "comparison", "versus", " vs ",
))
_FOLLOWUP_MARKERS = tuple(normalize_project_text(value) for value in (
    "ايهم", "أيهم", "بينهم", "المشروعين", "الاثنين", "كلاهما", "تكاليفهم",
    "ايراداتهم", "ربحهم", "هامشهم", "which one", "between them", "both projects",
))
_CLOSER_TO_COMPLETION = tuple(normalize_project_text(value) for value in (
    "ايهم اقرب للانتهاء", "أيهم أقرب للانتهاء", "ايهم اقرب للانجاز", "أيهم أقرب للإنجاز",
    "which is closer to completion", "which one is closer to completion", "which finishes sooner",
))
_SPLIT = re.compile(r"\s+(?:مقابل|مع|versus|vs\.?|and)\s+", re.IGNORECASE)
_ARABIC_BETWEEN = re.compile(r"(?:قارن\s+)?بين\s+(.+?)\s+و\s*(.+)$", re.IGNORECASE)


def is_comparison_request(query: str) -> bool:
    normalized = normalize_project_text(query)
    return any(marker.strip() in normalized for marker in _COMPARE_MARKERS)


def is_comparison_followup(query: str) -> bool:
    normalized = normalize_project_text(query)
    return any(marker in normalized for marker in _FOLLOWUP_MARKERS)


def is_comparison_winner_question(query: str) -> bool:
    normalized = normalize_project_text(query)
    return (
        any(marker in normalized for marker in ("ايهم", "مين", "which", "who"))
        and any(marker in normalized for marker in ("اعلي", "اقل", "اقرب", "higher", "lower", "closer", "sooner"))
    )


def extract_comparison_phrases(query: str) -> tuple[str, ...]:
    raw_query = str(query or "")
    arabic_between = _ARABIC_BETWEEN.search(raw_query)
    if arabic_between:
        return tuple(part.strip() for part in arabic_between.groups())
    parts = _SPLIT.split(raw_query, maxsplit=1)
    return tuple(part.strip() for part in parts) if len(parts) == 2 else ()


def resolve_comparison_entities(query: str, projects: list[dict[str, Any]]) -> MultipleProjectResolution:
    """Extract and resolve each side independently, preserving side status."""
    phrases = extract_comparison_phrases(query)
    return resolve_multiple_project_phrases(phrases, projects=projects)


def resolve_comparison_projects(query: str, projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return exactly two confidently identified projects, otherwise []."""
    multiple = resolve_comparison_entities(query, projects)
    if multiple.all_matched:
        codes = [resolution.matches[0].record.project_code for resolution in multiple.resolutions]
        resolved = [next((project for project in projects if project.get("project_code") == code), None) for code in codes]
        return resolved if all(resolved) and len({item.get("project_code") for item in resolved}) == 2 else []

    # Compatibility fallback for two exact identifiers when no separator
    # could be extracted. Normal comparison queries use the per-side path.
    normalized = normalize_project_text(query)
    positioned: list[tuple[int, dict[str, Any]]] = []
    for project in projects:
        identifiers = (
            project.get("project_code"), project.get("wbs_pc"), project.get("wbs"), project.get("pc"),
            project.get("project_name_ar"), project.get("project_name_en"),
        )
        positions = [normalized.find(normalize_project_text(value)) for value in identifiers if value]
        positions = [position for position in positions if position >= 0]
        if positions:
            positioned.append((min(positions), project))
    positioned.sort(key=lambda item: item[0])
    direct = [project for _, project in positioned]
    if len(direct) == 2:
        return direct
    if len(direct) > 2:
        return []

    # Partial names may still resolve safely when explicitly separated by a
    # comparison word. Each side must independently produce one match.
    raw_query = str(query or "")
    arabic_between = _ARABIC_BETWEEN.search(raw_query)
    parts = list(arabic_between.groups()) if arabic_between else _SPLIT.split(raw_query, maxsplit=1)
    resolved: list[dict[str, Any]] = []
    if len(parts) == 2:
        for part in parts:
            resolution = resolve_project(part, projects)
            if resolution.status != "matched":
                return []
            code = resolution.canonical_project_code
            row = next((project for project in projects if project.get("project_code") == code), None)
            if row is None or any(item.get("project_code") == code for item in resolved):
                return []
            resolved.append(row)
    return resolved if len(resolved) == 2 else []


def _money(value: Any, arabic: bool) -> str:
    if value is None:
        return "غير متوفر" if arabic else "Unavailable"
    amount = float(value)
    absolute = abs(amount)
    if absolute >= 1_000_000_000:
        rendered = f"{amount / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{rendered} مليار ريال" if arabic else f"SAR {rendered}B"
    if absolute >= 1_000_000:
        rendered = f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{rendered} مليون ريال" if arabic else f"SAR {rendered}M"
    return f"{amount:,.2f} ريال" if arabic else f"SAR {amount:,.2f}"


def _field_value(project: dict[str, Any], canonical: str) -> Any:
    return project.get("net_profit") if canonical == "pl" else project.get(canonical)


def _render_value(project: dict[str, Any], canonical: str, arabic: bool) -> str:
    value = _field_value(project, canonical)
    if value is None:
        return "غير متوفر" if arabic else "Unavailable"
    definition = FIELDS.get(canonical)
    if canonical == "pl" or (definition and definition.data_type == "money"):
        return _money(value, arabic)
    if definition and definition.data_type == "percentage":
        return f"{float(value):.1f}%"
    return str(value)


def comparison_field(query: str, fallback: str = "pl") -> str:
    normalized = normalize_project_text(query)
    if any(marker in normalized for marker in _CLOSER_TO_COMPLETION):
        return "effective_end_date"
    field = detect_requested_field(query)
    return field.canonical if field else fallback


def format_comparison_summary(projects: list[dict[str, Any]], query: str, current_field: str | None = None) -> str:
    """Concise verified summary of the active two-project comparison."""
    arabic = any("\u0600" <= char <= "\u06ff" for char in query)
    names = [(project.get("project_name_ar") if arabic else project.get("project_name_en")) or project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code") for project in projects]
    fields = []
    for field in (current_field, "total_revenue", "total_cost", "pl", "profit_pct", "effective_end_date"):
        if field and field not in fields and all(_field_value(project, field) is not None for project in projects):
            fields.append(field)
    lines = []
    for field in fields[:4]:
        left, right = _field_value(projects[0], field), _field_value(projects[1], field)
        if left == right:
            continue
        if field in {"effective_end_date", "end_date", "amended_end_date", "days_remaining"}:
            winner = 0 if left < right else 1
            relation = "أقرب للانتهاء" if arabic else "closer to completion"
        else:
            winner = 0 if float(left) > float(right) else 1
            relation = f"أعلى في {FIELDS[field].label_ar}" if arabic else f"higher in {FIELDS[field].label_en}"
        lines.append(f"«{names[winner]}» {relation} ({_render_value(projects[winner], field, arabic)}).")
    if not lines:
        return "لا يظهر فرق موثق في المؤشرات المتاحة للمشروعين." if arabic else "No verified difference appears in the available metrics."
    prefix = "الخلاصة: " if arabic else "Summary: "
    return prefix + " ".join(lines)


def format_comparison(projects: list[dict[str, Any]], query: str, field: str | None = None) -> str:
    arabic = any("\u0600" <= char <= "\u06ff" for char in query)
    fields = [field] if field else ["total_revenue", "total_cost", "pl", "profit_pct", "progress_completed"]
    names = [(project.get("project_name_ar") if arabic else project.get("project_name_en")) or project.get("project_name_ar") or project.get("project_name_en") or project.get("project_code") for project in projects]
    if arabic:
        header = f"| المؤشر | {names[0]} | {names[1]} |\n|---|---:|---:|"
        rows = []
        for canonical in fields:
            label = FIELDS[canonical].label_ar
            rows.append(f"| {label} | {_render_value(projects[0], canonical, True)} | {_render_value(projects[1], canonical, True)} |")
        return "مقارنة موثقة بين المشروعين:\n\n" + header + "\n" + "\n".join(rows)
    header = f"| Metric | {names[0]} | {names[1]} |\n|---|---:|---:|"
    rows = []
    for canonical in fields:
        label = FIELDS[canonical].label_en
        rows.append(f"| {label} | {_render_value(projects[0], canonical, False)} | {_render_value(projects[1], canonical, False)} |")
    return "Verified comparison:\n\n" + header + "\n" + "\n".join(rows)


def format_comparison_winner(projects: list[dict[str, Any]], query: str, field: str) -> str:
    arabic = any("\u0600" <= char <= "\u06ff" for char in query)
    available = [project for project in projects if _field_value(project, field) is not None]
    if len(available) != 2:
        return "لا تتوفر قيمة موثقة للمؤشر في كلا المشروعين." if arabic else "A verified value is not available for both projects."
    normalized = normalize_project_text(query)
    lower_wins = field in {"effective_end_date", "end_date", "amended_end_date", "days_remaining"} and any(
        marker in normalized for marker in ("اقرب", "أقرب", "closer", "sooner", "nearest")
    )
    left_value, right_value = (_field_value(project, field) for project in available)
    if left_value == right_value:
        label = FIELDS[field].label_ar if arabic else FIELDS[field].label_en
        value = _render_value(available[0], field, arabic)
        return (f"المشروعان متعادلين في {label} بقيمة {value}." if arabic else
                f"Both projects are tied on {label} at {value}.")
    winner = min(available, key=lambda project: _field_value(project, field)) if lower_wins else max(available, key=lambda project: _field_value(project, field))
    name = ((winner.get("project_name_ar") if arabic else winner.get("project_name_en")) or
            winner.get("project_name_ar") or winner.get("project_name_en") or winner.get("project_code"))
    label = FIELDS[field].label_ar if arabic else FIELDS[field].label_en
    value = _render_value(winner, field, arabic)
    if lower_wins:
        return (f"الأقرب للانتهاء هو مشروع «{name}»، وتاريخ انتهائه {value}." if arabic
                else f"{name} is closer to completion, with an end date of {value}.")
    return (f"الأعلى في {label} هو مشروع «{name}» بقيمة {value}." if arabic
            else f"{name} is higher on {label}, at {value}.")
