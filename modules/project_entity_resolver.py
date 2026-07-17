"""Deterministic Arabic/English project entity resolution.

The language model never selects a database row.  This module resolves a
user-supplied project phrase against canonical project codes and names before
the structured query executor is allowed to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Literal, Mapping

from modules.semantic_dictionary import normalize_text

try:  # RapidFuzz is preferred in production; the fallback keeps tests portable.
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    _rapidfuzz = None


ResolutionStatus = Literal["matched", "confirmation", "ambiguous", "no_match"]

HIGH_FUZZY_THRESHOLD = 88.0
MEDIUM_FUZZY_THRESHOLD = 72.0
# Below this length, partial-substring scoring (partial_ratio, prefix,
# "partial_name" containment) is disabled: a short window inside a much
# longer unrelated name can align well by chance (e.g. "ناسا" spuriously
# matching an unrelated project), giving false confidence. Short phrases
# must clear a higher whole-string similarity bar instead (see
# SHORT_PHRASE_FUZZY_THRESHOLD / _similarity below).
SHORT_PHRASE_MIN_LENGTH = 6
SHORT_PHRASE_FUZZY_THRESHOLD = 90.0
MIN_FUZZY_GAP = 8.0
MAX_AMBIGUOUS_CANDIDATES = 20

# Words that describe the request rather than the entity.  They are removed
# only from the user phrase, never from stored database values.
_REQUEST_WORDS = {
    "اعطني", "اعطيني", "ابي", "ابغي", "اريد", "احتاج", "لو", "سمحت",
    "من", "فضلك", "ملخص", "لخص", "تفاصيل", "معلومات", "بيانات", "عن",
    "مشروع", "المشروع", "ومشروع", "وعن", "حاله", "وضع", "وش", "كيف", "صار", "عليه", "علومه",
    "ما", "هو", "هي", "لي", "اظهر", "اعرض",
    "show", "give", "me", "summary", "details", "information", "about",
    "project", "the", "please", "status", "of",
    # Financial / contract request words
    "كم", "ما", "اخبرني", "قل", "هل",
    "قيمه", "قيمة", "قيم",
    "عقد", "العقد", "عقده",
    "ايرادات", "إيرادات", "ايراد", "إيراد",
    "تكاليف", "تكلفة", "تكلفه",
    "ربح", "ربحه", "ربحها",
    "هامش",
    "ملخصه", "ملخصها",
    "تقدم", "التقدم", "نسبة", "نسبه",
}
_GENERIC_NAME_WORDS = {"مشروع", "المشروع", "project"}


def normalize_project_text(value: Any) -> str:
    """Compatibility wrapper around the application's single normalizer."""
    return normalize_text(value)


def extract_project_phrase(query: str) -> str:
    """Remove request-language tokens and retain the likely entity phrase."""
    normalized = normalize_project_text(query)
    tokens = [token for token in normalized.split() if token not in _REQUEST_WORDS]
    return " ".join(tokens).strip()


def _without_generic_words(value: str) -> str:
    return " ".join(token for token in value.split() if token not in _GENERIC_NAME_WORDS)


def _variants(value: Any) -> set[str]:
    normalized = normalize_project_text(value)
    if not normalized:
        return set()
    stripped = _without_generic_words(normalized)
    return {variant for variant in (normalized, stripped) if variant}


def _compact(value: str) -> str:
    return value.replace(" ", "")


def _fallback_partial_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if short in long:
        return 100.0
    window = len(short)
    scores = [
        SequenceMatcher(None, short, long[start:start + window]).ratio() * 100
        for start in range(max(1, len(long) - window + 1))
    ]
    return max(scores, default=0.0)


def _similarity_whole(left: str, right: str) -> float:
    """Whole-string similarity only -- no substring-window scoring. Safe to
    use on short phrases since it can't be fooled by a short window
    coincidentally aligning inside a much longer, unrelated string."""
    if _rapidfuzz is not None:
        return max(
            float(_rapidfuzz.ratio(left, right)),
            float(_rapidfuzz.token_set_ratio(left, right)),
        )
    ratio = SequenceMatcher(None, left, right).ratio() * 100
    token_ratio = SequenceMatcher(
        None, " ".join(sorted(left.split())), " ".join(sorted(right.split()))
    ).ratio() * 100
    return max(ratio, token_ratio)


def _similarity(left: str, right: str) -> float:
    """Full similarity including partial/substring-window scoring. Only
    call this when at least one side is long enough (>= SHORT_PHRASE_MIN_LENGTH)
    that a coincidental short-window alignment is implausible."""
    whole = _similarity_whole(left, right)
    if _rapidfuzz is not None:
        return max(whole, float(_rapidfuzz.partial_ratio(left, right)))
    return max(whole, _fallback_partial_ratio(left, right))


@dataclass(frozen=True)
class ProjectCandidate:
    project_code: str
    project_name_ar: str
    project_name_en: str
    score: float
    match_type: str

    @property
    def display_name(self) -> str:
        return self.project_name_ar or self.project_name_en or self.project_code


@dataclass(frozen=True)
class ProjectResolution:
    status: ResolutionStatus
    extracted_phrase: str
    candidates: tuple[ProjectCandidate, ...] = field(default_factory=tuple)
    confidence: float = 0.0

    @property
    def canonical_project_code(self) -> str | None:
        if self.status in {"matched", "confirmation"} and len(self.candidates) == 1:
            return self.candidates[0].project_code
        return None


def _candidate(project: Mapping[str, Any], score: float, match_type: str) -> ProjectCandidate:
    return ProjectCandidate(
        project_code=str(project.get("project_code") or "").strip(),
        project_name_ar=str(project.get("project_name_ar") or "").strip(),
        project_name_en=str(project.get("project_name_en") or "").strip(),
        score=round(score, 2),
        match_type=match_type,
    )


def _deduplicate(projects: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_code: dict[str, Mapping[str, Any]] = {}
    for project in projects:
        code = str(project.get("project_code") or "").strip()
        if code:
            by_code[code.casefold()] = project
    return list(by_code.values())


def _resolve_project_legacy(query: str, projects: Iterable[Mapping[str, Any]]) -> ProjectResolution:
    """Resolve a project query using ordered exact-to-fuzzy strategies."""
    portfolio = _deduplicate(projects)
    extracted = extract_project_phrase(query)
    raw_query = str(query or "")

    # A. Exact canonical identifier match anywhere in the question. WBS/PC
    # identifiers resolve to the canonical project_code used downstream.
    code_matches = []
    for project in portfolio:
        for key in ("project_code", "wbs_pc", "wbs", "pc"):
            identifier = str(project.get(key) or "").strip()
            if identifier and re.search(
                rf"(?<!\w){re.escape(identifier)}(?!\w)", raw_query, re.IGNORECASE
            ):
                code_matches.append(_candidate(project, 100, f"exact_{key}"))
                break
    if len(code_matches) == 1:
        return ProjectResolution("matched", extracted, tuple(code_matches), 100)
    if len(code_matches) > 1:
        return ProjectResolution("ambiguous", extracted, tuple(code_matches), 100)

    if not extracted:
        return ProjectResolution("no_match", extracted)

    project_variants: list[tuple[Mapping[str, Any], set[str]]] = []
    for project in portfolio:
        variants = _variants(project.get("project_name_ar")) | _variants(project.get("project_name_en"))
        if variants:
            project_variants.append((project, variants))

    # B. Exact normalized Arabic/English name, including harmless spacing.
    exact = []
    extracted_compact = _compact(extracted)
    for project, variants in project_variants:
        if extracted in variants or extracted_compact in {_compact(v) for v in variants}:
            exact.append(_candidate(project, 100, "exact_name"))
    if len(exact) == 1:
        return ProjectResolution("matched", extracted, tuple(exact), 100)
    if len(exact) > 1:
        return ProjectResolution("ambiguous", extracted, tuple(exact), 100)

    # C/D. Token containment, prefix and partial-name matching.  Multiple
    # matches are always surfaced instead of selecting an arbitrary row.
    query_tokens = set(extracted.split())
    partial = []
    for project, variants in project_variants:
        matched = False
        match_type = "token_containment"
        for variant in variants:
            variant_tokens = set(variant.split())
            compact_variant = _compact(variant)
            if query_tokens and query_tokens.issubset(variant_tokens):
                matched = True
            elif len(extracted_compact) >= SHORT_PHRASE_MIN_LENGTH and extracted_compact in compact_variant:
                matched = True
                match_type = "partial_name"
            elif len(extracted) >= SHORT_PHRASE_MIN_LENGTH and variant.startswith(extracted):
                matched = True
                match_type = "prefix"
            if matched:
                break
        if matched:
            partial.append(_candidate(project, 96, match_type))
    if len(partial) == 1:
        return ProjectResolution("matched", extracted, tuple(partial), 96)
    if len(partial) > 1:
        return ProjectResolution(
            "ambiguous", extracted, tuple(partial[:MAX_AMBIGUOUS_CANDIDATES]), 96
        )

    # E. Fuzzy matching. A strong unique winner executes directly; a medium
    # unique winner requires confirmation; close candidates remain ambiguous.
    # Short extracted phrases never get substring/partial scoring and must
    # clear a much higher whole-string bar -- this is what prevents a short,
    # unrelated string (e.g. "ناسا") from spuriously matching a real,
    # unrelated project via a coincidentally-aligned substring window.
    is_short_phrase = len(extracted) < SHORT_PHRASE_MIN_LENGTH
    effective_threshold = SHORT_PHRASE_FUZZY_THRESHOLD if is_short_phrase else MEDIUM_FUZZY_THRESHOLD
    score_fn = _similarity_whole if is_short_phrase else _similarity

    fuzzy = []
    for project, variants in project_variants:
        score = max((score_fn(extracted, variant) for variant in variants), default=0.0)
        if score >= effective_threshold:
            fuzzy.append(_candidate(project, score, "fuzzy"))
    fuzzy.sort(key=lambda item: (-item.score, item.display_name))
    if not fuzzy:
        return ProjectResolution("no_match", extracted)

    top = fuzzy[0]
    runner_up = fuzzy[1].score if len(fuzzy) > 1 else 0.0
    close = tuple(item for item in fuzzy if top.score - item.score < MIN_FUZZY_GAP)
    if top.score >= HIGH_FUZZY_THRESHOLD and len(close) == 1:
        return ProjectResolution("matched", extracted, (top,), top.score)
    if len(close) > 1:
        return ProjectResolution(
            "ambiguous", extracted, close[:MAX_AMBIGUOUS_CANDIDATES], top.score
        )
    if top.score - runner_up >= MIN_FUZZY_GAP:
        return ProjectResolution("confirmation", extracted, (top,), top.score)
    return ProjectResolution("no_match", extracted)


def resolve_project(query: str, projects: Iterable[Mapping[str, Any]]) -> ProjectResolution:
    """Resolve through the centralized database-driven project-name index."""
    from modules.project_name_index import resolve_project_phrase

    extracted = extract_project_phrase(query)
    indexed = resolve_project_phrase(extracted, projects=projects)
    candidates = tuple(
        ProjectCandidate(
            project_code=match.record.project_code,
            project_name_ar=match.record.official_ar,
            project_name_en=match.record.official_en,
            score=match.score,
            match_type=match.match_type,
        )
        for match in indexed.matches
    )
    return ProjectResolution(indexed.status, extracted, candidates, indexed.confidence)


def format_resolution_prompt(resolution: ProjectResolution) -> str:
    if resolution.status == "no_match":
        return "لم أجد مشروعًا مطابقًا للاسم المذكور في البيانات الحالية."
    if resolution.status == "confirmation":
        return f"هل تقصد مشروع «{resolution.candidates[0].display_name}»؟"
    if resolution.status == "ambiguous":
        normalized_names = [normalize_project_text(candidate.display_name) for candidate in resolution.candidates]
        names = "\n".join(
            f"{index + 1}. {candidate.display_name}"
            + (f" ({candidate.project_code})" if normalized_names.count(normalized_names[index]) > 1 else "")
            for index, candidate in enumerate(resolution.candidates)
        )
        return f"وجدت أكثر من مشروع قريب من الاسم. هل تقصد:\n{names}\n\nاختر الرقم أو اكتب الاسم الأقرب."
    raise ValueError("matched resolutions should be executed, not formatted as prompts")