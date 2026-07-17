"""
modules/project_name_index.py
Dynamic project name index with alias generation.
Delegates to the legacy resolver; designed for future LRU caching and
richer alias expansion (nicknames, years, tokens) without changing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from modules.project_entity_resolver import (
    _resolve_project_legacy as _legacy,
    ProjectResolution,
    ProjectCandidate,
    normalize_project_text,
)


@dataclass(frozen=True)
class ProjectRecord:
    project_code: str
    official_ar: str
    official_en: str


@dataclass(frozen=True)
class IndexMatch:
    record: ProjectRecord
    score: float
    match_type: str


@dataclass(frozen=True)
class IndexResolution:
    status: str          # matched | ambiguous | confirmation | no_match
    matches: tuple[IndexMatch, ...]
    confidence: float

    @property
    def all_matched(self) -> bool:
        return self.status == "matched" and len(self.matches) == 1


@dataclass(frozen=True)
class SingleProjectResolution:
    status: str
    matches: tuple[IndexMatch, ...]
    confidence: float

    @property
    def all_matched(self) -> bool:
        return self.status == "matched"


@dataclass(frozen=True)
class MultipleProjectResolution:
    resolutions: tuple[SingleProjectResolution, ...]

    @property
    def all_matched(self) -> bool:
        return all(r.all_matched for r in self.resolutions)


def _to_record(candidate: ProjectCandidate) -> ProjectRecord:
    return ProjectRecord(
        project_code=candidate.project_code,
        official_ar=candidate.project_name_ar,
        official_en=candidate.project_name_en,
    )


def _to_index_resolution(res: ProjectResolution) -> IndexResolution:
    matches = tuple(
        IndexMatch(record=_to_record(c), score=c.score, match_type=c.match_type)
        for c in res.candidates
    )
    return IndexResolution(status=res.status, matches=matches, confidence=res.confidence)


def resolve_project_phrase(
    phrase: str,
    projects: Iterable[dict[str, Any]] | None = None,
) -> IndexResolution:
    """Resolve a single phrase against the project index."""
    if projects is None:
        from modules.project_repository import fetch_enriched_projects
        projects = fetch_enriched_projects()
    res = _legacy(phrase, list(projects))
    return _to_index_resolution(res)


def resolve_multiple_project_phrases(
    phrases: tuple[str, ...],
    projects: Iterable[dict[str, Any]] | None = None,
) -> MultipleProjectResolution:
    """Resolve multiple phrases independently (for comparison)."""
    if projects is None:
        from modules.project_repository import fetch_enriched_projects
        projects = list(fetch_enriched_projects())
    else:
        projects = list(projects)
    resolutions = []
    for phrase in phrases:
        res = _legacy(phrase, projects)
        matches = tuple(
            IndexMatch(record=_to_record(c), score=c.score, match_type=c.match_type)
            for c in res.candidates
        )
        resolutions.append(SingleProjectResolution(
            status=res.status, matches=matches, confidence=res.confidence,
        ))
    return MultipleProjectResolution(resolutions=tuple(resolutions))