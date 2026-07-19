"""
modules/entity_resolvers.py
----------------------------
Layer 2 of the semantic interpretation pipeline (see modules.semantic_interpreter).

Generic, data-driven resolution of a raw text mention to canonical DB
values, for every categorical column the copilot understands -- department,
status, business unit, segment, program, project manager, customer,
officer -- plus ordinal references into an arbitrary candidate list.

Every resolver shares one mechanism (`build_value_index`) and one fuzzy-match
safety net (modules.project_entity_resolver's thresholds/scoring, reused
here rather than re-tuned). Manual alias dictionaries (STATUS_ALIASES,
DEPT_ALIASES in modules.semantic_dictionary) are a small override layer on
top of the data-driven index, never the primary mechanism: they exist only
for columns whose Arabic vocabulary cannot be derived from the data itself
(status is a fixed short English enum; department names are English and
truncated at the source, with no Arabic column at all). Every other column
(bu, segment, program, project_manager, customer_id, officer_name) is
resolved purely from the live distinct values in the database -- nothing
about their valid values is hardcoded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

from modules.project_entity_resolver import (
    HIGH_FUZZY_THRESHOLD,
    MEDIUM_FUZZY_THRESHOLD,
    MIN_FUZZY_GAP,
    SHORT_PHRASE_FUZZY_THRESHOLD,
    SHORT_PHRASE_MIN_LENGTH,
    _similarity,
    _similarity_whole,
)
from modules.semantic_dictionary import DEPT_ALIASES, STATUS_ALIASES, normalize_text

ResolverStatus = Literal["exact", "fuzzy", "ambiguous", "none"]


@dataclass(frozen=True)
class ResolverResult:
    status: ResolverStatus
    values: tuple[str, ...] = ()
    confidence: float = 0.0
    candidates: tuple[str, ...] = ()

    @property
    def value(self) -> str | None:
        """Single resolved value, only when there is exactly one -- callers
        that need a clarification prompt instead should read `candidates`."""
        if self.status in {"exact", "fuzzy"} and len(self.values) == 1:
            return self.values[0]
        return None


_NONE = ResolverResult(status="none")


def _fuzzy_best(
    normalized_mention: str,
    candidates: Iterable[tuple[str, str]],
    *,
    threshold: float | None = None,
    score_fn=None,
) -> ResolverResult:
    """Shared fuzzy-scoring/tie-break logic. `candidates` is an iterable of
    (normalized_text_to_score_against, canonical_value_to_return).

    By default uses the same short-phrase-aware threshold/scorer as project
    name resolution (safe for arbitrarily long DB literal strings, where a
    short substring could spuriously align inside an unrelated long value).
    Callers matching against a small closed vocabulary of short target
    words (e.g. the 5 status labels) may pass an explicit `threshold`/
    `score_fn` instead -- the short-phrase substring-window risk that
    SHORT_PHRASE_FUZZY_THRESHOLD guards against doesn't apply when every
    candidate is itself short and whole-string compared.
    """
    if threshold is None or score_fn is None:
        is_short = len(normalized_mention) < SHORT_PHRASE_MIN_LENGTH
        threshold = SHORT_PHRASE_FUZZY_THRESHOLD if is_short else MEDIUM_FUZZY_THRESHOLD
        score_fn = _similarity_whole if is_short else _similarity
    scored = [
        (score_fn(normalized_mention, normalized), canonical)
        for normalized, canonical in candidates
    ]
    scored = [(score, canonical) for score, canonical in scored if score >= threshold]
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return _NONE
    top_score, top_value = scored[0]
    close = sorted({canonical for score, canonical in scored if top_score - score < MIN_FUZZY_GAP})
    if len(close) > 1:
        return ResolverResult(status="ambiguous", candidates=tuple(close), confidence=top_score / 100)
    if top_score >= HIGH_FUZZY_THRESHOLD:
        return ResolverResult(status="fuzzy", values=(top_value,), confidence=top_score / 100)
    return ResolverResult(status="ambiguous", candidates=(top_value,), confidence=top_score / 100)


@dataclass(frozen=True)
class ValueIndex:
    """A normalized-alias -> canonical-value lookup for one column, built
    from real, currently-live DB values, with an optional small manual
    override layer checked first."""

    column: str
    canonical_values: tuple[str, ...]
    manual_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def resolve(self, mention: str) -> ResolverResult:
        mention = (mention or "").strip()
        if not mention:
            return _NONE
        normalized_mention = normalize_text(mention)
        if not normalized_mention:
            return _NONE

        # 1. Manual override layer -- longest alias wins, exact confidence.
        # Only used where curated vocabulary exists; most columns have none.
        best_key, best_len = None, 0
        for canonical, aliases in self.manual_aliases.items():
            for alias in aliases:
                norm_alias = normalize_text(alias)
                if norm_alias and norm_alias in normalized_mention and len(norm_alias) > best_len:
                    best_key, best_len = canonical, len(norm_alias)
        if best_key is not None:
            return ResolverResult(status="exact", values=(best_key,), confidence=1.0)

        # 1b. Manual override typo tolerance -- a substring check alone
        # can't catch a misspelled/transposed curated alias (e.g. "الادراة"
        # for "الإدارة"); whole-word fuzzy match against the same curated
        # vocabulary, same as modules.entity_resolvers.resolve_status's
        # dedicated fallback, generalized here for every manual-alias column.
        if self.manual_aliases:
            fuzzy_manual = _fuzzy_best(
                normalized_mention,
                (
                    (normalize_text(alias), canonical)
                    for canonical, aliases in self.manual_aliases.items()
                    for alias in aliases
                ),
                threshold=MEDIUM_FUZZY_THRESHOLD,
                score_fn=_similarity_whole,
            )
            if fuzzy_manual.status != "none":
                return fuzzy_manual

        if not self.canonical_values:
            return _NONE

        # 2. Exact normalized match against a real, live DB value.
        exact = [v for v in self.canonical_values if normalize_text(v) == normalized_mention]
        if len(exact) == 1:
            return ResolverResult(status="exact", values=(exact[0],), confidence=1.0)
        if len(exact) > 1:
            return ResolverResult(status="ambiguous", candidates=tuple(sorted(exact)), confidence=1.0)

        # 3. Substring containment either direction -- short, truncated DB
        # literals (e.g. "BPO-Specialized Pr") commonly appear as a
        # substring of what the user typed, and vice versa.
        contains = [
            v for v in self.canonical_values
            if normalize_text(v) and (
                normalize_text(v) in normalized_mention or normalized_mention in normalize_text(v)
            )
        ]
        if len(contains) == 1:
            return ResolverResult(status="exact", values=(contains[0],), confidence=0.95)
        if len(contains) > 1:
            return ResolverResult(status="ambiguous", candidates=tuple(sorted(contains)), confidence=0.95)

        # 4. Fuzzy match -- reuses project_entity_resolver's exact
        # thresholds/scoring, one safety net across every entity type.
        return _fuzzy_best(
            normalized_mention,
            ((normalize_text(v), v) for v in self.canonical_values),
        )


def build_value_index(
    projects: Iterable[Mapping[str, Any]],
    column: str,
    manual_aliases: dict[str, tuple[str, ...]] | None = None,
) -> ValueIndex:
    """Build an index from the *live* distinct values of `column` across
    `projects`. Nothing about valid values is hardcoded -- if the
    underlying data changes, the index changes with it on the next call."""
    values = {str(p.get(column) or "").strip() for p in projects}
    values.discard("")
    return ValueIndex(column=column, canonical_values=tuple(sorted(values)), manual_aliases=manual_aliases or {})


# ── Named resolvers ──────────────────────────────────────────────────────
# One shape for every entity type. Columns backed by real data-driven
# diversity (bu/segment/program/manager/customer/officer) take `projects`
# and consult nothing but the live DB. `status` and `dept` additionally
# consult the small curated override dicts in modules.semantic_dictionary,
# because their Arabic vocabulary genuinely cannot be derived from the data.

def resolve_status(mention: str) -> ResolverResult:
    """Status is a fixed 5-value enum stored as English literals; Arabic
    vocabulary is manual-only. ValueIndex's manual-alias layer already
    includes fuzzy typo tolerance (e.g. a misspelled "نشيطة" still resolves
    to Ongoing), so this is a thin, empty-canonical-values wrapper around it."""
    return ValueIndex(column="status", canonical_values=(), manual_aliases=STATUS_ALIASES).resolve(mention)


def resolve_department(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "dept", manual_aliases=DEPT_ALIASES).resolve(mention)


def resolve_bu(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "bu").resolve(mention)


def resolve_segment(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "segment").resolve(mention)


def resolve_program(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "program").resolve(mention)


def resolve_manager(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "project_manager").resolve(mention)


def resolve_customer(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "customer_id").resolve(mention)


def resolve_officer(mention: str, projects: Iterable[Mapping[str, Any]]) -> ResolverResult:
    return build_value_index(projects, "officer_name").resolve(mention)


# ── Ordinal / positional references ─────────────────────────────────────
# Generalizes what modules.ai_engine previously kept as a private dict
# wired only to project-disambiguation lists (_ORDINAL_NORM/_LAST_NORM/
# _NEXT_NORM), so the same phrasing also resolves against any ordered
# candidate list -- e.g. a just-returned portfolio-filter result list.

_ORDINAL_WORDS: dict[str, int] = {
    "الاول": 0, "الأول": 0, "الاولى": 0, "الأولى": 0, "اول": 0, "أول": 0, "first": 0, "1": 0,
    "الثاني": 1, "الثانيه": 1, "الثانية": 1, "second": 1, "2": 1,
    "الثالث": 2, "الثالثه": 2, "الثالثة": 2, "third": 2, "3": 2,
    "الرابع": 3, "الرابعه": 3, "الرابعة": 3, "fourth": 3, "4": 3,
    "الخامس": 4, "الخامسه": 4, "الخامسة": 4, "fifth": 4, "5": 4,
    "السادس": 5, "السادسه": 5, "السادسة": 5, "sixth": 5, "6": 5,
    "السابع": 6, "السابعه": 6, "السابعة": 6, "seventh": 6, "7": 6,
    "الثامن": 7, "الثامنه": 7, "الثامنة": 7, "eighth": 7, "8": 7,
    "التاسع": 8, "التاسعه": 8, "التاسعة": 8, "ninth": 8, "9": 8,
    "العاشر": 9, "العاشره": 9, "العاشرة": 9, "tenth": 9, "10": 9,
}
_ORDINAL_NORM = {normalize_text(k): v for k, v in _ORDINAL_WORDS.items()}
_LAST_WORDS = {normalize_text(w) for w in ("الأخير", "الاخير", "آخر واحد", "اخر واحد", "last")}
_NEXT_WORDS = {normalize_text(w) for w in ("اللي بعده", "التالي", "next")}


def resolve_ordinal(mention: str, candidate_count: int, current_index: int | None = None) -> int | None:
    """Resolve an ordinal/positional reference ("الأول", "الثاني", "آخر
    واحد", "اللي بعده"...) against any ordered candidate list of length
    `candidate_count`. Returns a zero-based index, or None if `mention`
    isn't an ordinal reference or the index would be out of range.
    """
    if candidate_count <= 0:
        return None
    normalized = normalize_text(mention)
    if not normalized:
        return None
    tokens = set(normalized.split())
    if normalized in _LAST_WORDS or tokens & _LAST_WORDS:
        return candidate_count - 1
    if normalized in _NEXT_WORDS or tokens & _NEXT_WORDS:
        base = current_index if current_index is not None else -1
        return min(base + 1, candidate_count - 1)
    for word, idx in _ORDINAL_NORM.items():
        if (word in tokens or normalize_text("و" + word) in tokens) and 0 <= idx < candidate_count:
            return idx
    return None
