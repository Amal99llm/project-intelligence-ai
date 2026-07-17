"""
modules/followup_gate.py
------------------------
Phase 1 — Follow-up First Gate.

Semantic gate that fires BEFORE understanding.py and BEFORE any entity
resolution. If the gate fires, the question is treated as a project
follow-up immediately — no LLM, no pattern matching, no entity resolver.

Detection is semantic, NOT word-count based. A query passes the gate when:

  1. An active project exists in context (active_project_code)
  2. No new explicit project name appears in the query
  3. One or more of:
       a. Possessive/pronoun references  (وتكاليفه، ربحها، حالته...)
       b. Requested field detectable from the query
       c. Assessment/evaluation intent   (هل يحتاج، وش وضعه، يستاهل...)
       d. Elliptical question that only makes sense with a project in context
  4. NOT a portfolio-scope override      (إجمالي، جميع المشاريع...)
  5. NOT a list followup                 (already handled separately)
  6. NOT a comparison trigger

The gate returns a FollowupDecision with:
  - fires: bool
  - field: str | None   — canonical field if detectable
  - intent: str         — "field_lookup" | "assessment" | "status" | "general_followup"
  - confidence: float
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.semantic_dictionary import (
    normalize_text,
    detect_requested_field,
    FieldDefinition,
)


@dataclass(frozen=True)
class FollowupDecision:
    fires: bool
    field: str | None = None
    intent: str = "general_followup"
    confidence: float = 0.0
    reason: str = ""


_NO_FIRE = FollowupDecision(fires=False)


# ── Possessive / pronoun markers (Arabic + English) ──────────────────────────
# These are SUFFIXED pronouns or short possessive phrases that signal
# "the thing we were just talking about". No explicit project name needed.
_POSSESSIVE_SUFFIXES = tuple(normalize_text(v) for v in (
    # Arabic suffix forms (attached to field words)
    "وتكاليفه", "وتكاليفها", "وتكلفته", "وتكلفتها",
    "وربحه", "وربحها", "وخسارته", "وخسارتها",
    "وايراداته", "وايراداتها", "وإيراداته", "وإيراداتها",
    "وهامشه", "وهامشها", "ونسبته", "ونسبتها",
    "وعقده", "وعقدها", "وباكلوجه", "وباكلوجها",
    "وتقدمه", "وتقدمها", "وحالته", "وحالتها",
    "ومتى ينتهي", "وتاريخه", "وتاريخها",
    "ومديره", "ومديرها",
    # Without leading و
    "تكاليفه", "تكاليفها", "تكلفته", "تكلفتها",
    "ربحه", "ربحها", "خسارته", "خسارتها",
    "ايراداته", "ايراداتها", "إيراداته", "إيراداتها",
    "هامشه", "هامشها", "نسبته", "نسبتها",
    "عقده", "عقدها", "باكلوجه", "باكلوجها",
    "تقدمه", "تقدمها", "حالته", "حالتها",
    "وضعه", "وضعها",
    "مديره", "مديرها",
    # English possessives
    "its cost", "its revenue", "its profit", "its margin", "its status",
    "its progress", "its contract", "its backlog", "its manager", "its end date",
))

# ── Assessment / evaluation phrases ──────────────────────────────────────────
_ASSESSMENT_MARKERS = tuple(normalize_text(v) for v in (
    "هل يحتاج متابعة", "هل يحتاج", "يحتاج تدخل",
    "وش وضعه", "وش وضعها", "وش حاله", "وش حالها",
    "كيف وضعه", "كيف حاله", "ايش وضعه",
    "هل هو بخير", "هل هي بخير",
    "يستاهل انتباه", "يستحق انتباه",
    "كيف يسير", "كيف تسير",
    "هل هو في خطر", "في خطر",
    "needs attention", "how is it doing", "is it on track",
    "any concerns", "worth attention",
    # Opinion / assessment requests
    "وش رأيك", "وش رأيك فيه", "وش رأيك فيها",
    "ايش رأيك", "رأيك فيه", "رأيك فيها",
    "تقييمه", "تقييمها", "قيّمه", "قيّمها",
    "كيف تشوفه", "كيف تشوفها",
    "what do you think", "your assessment", "evaluate it",
    "وش تقول عنه", "وش تقول عنها",
    # Status/timeline checks
    "هل المشروع متاخر", "هل المشروع متأخر", "هل تأخر", "هل تاخر",
    "هل انتهى في وقته", "هل هو في الموعد", "on schedule", "is it delayed",
    "هل اكتمل", "هل خلص", "هل ينتهي",
    # Ownership
    "مالك المشروع", "صاحب المشروع", "العميل", "الجهة",
))

# ── Elliptical single-word/short questions ───────────────────────────────────
# Questions that by themselves are incomplete; they only make sense
# if there's an active project in context.
_ELLIPTICAL_MARKERS = tuple(normalize_text(v) for v in (
    "وربحيته", "وربحيتها",
    "وهامش الربح", "والهامش",
    "والتكاليف", "والإيرادات",
    "والعقد", "والباكلوج",
    "والتقدم", "والحالة",
    "نفس المشروع", "هذا المشروع", "المشروع نفسه",
    "same project", "this project", "that project",
))

# ── Portfolio scope overrides (gate must NOT fire) ────────────────────────────
_PORTFOLIO_OVERRIDES = tuple(normalize_text(v) for v in (
    "إجمالي", "الاجمالي", "الكلي",
    "جميع المشاريع", "كل المشاريع", "لجميع", "لكل",
    "المحفظة", "المحفظه", "portfolio",
    # Portfolio filter phrases — never project followups
    "أي المشاريع", "اي المشاريع", "مشاريع خسران", "مشاريع خاسره",
    "المشاريع الخاسره", "مشاريع تنتهي", "المشاريع التي",
    # "عندنا/عندكم X مشروع" = portfolio question
    "عندنا مشاريع", "عندكم مشاريع", "عندنا مشروع", "ليش عندنا",
    "ليش فيه مشاريع", "ليش في مشاريع",
))

# ── Comparison triggers (gate must NOT fire) ──────────────────────────────────
_CMP_OVERRIDES = tuple(normalize_text(v) for v in (
    "قارن", "مقارنة", "مقابل", "vs", "versus", "compare",
))

# ── New-project signals (gate must NOT fire) ──────────────────────────────────
# If the query contains words signalling intent to look at a DIFFERENT project,
# the gate should not fire even if there is an active project.
_NEW_PROJECT_SIGNALS = tuple(normalize_text(v) for v in (
    "مشروع آخر", "مشروع ثاني", "غيره", "غيرها",
    "another project", "different project",
))


def _has_any(q_norm: str, markers: tuple) -> bool:
    return any(m in q_norm for m in markers)


def check(query: str, ctx: dict) -> FollowupDecision:
    """
    Return a FollowupDecision.
    Call this BEFORE understanding.py — if fires=True, skip straight to
    the followup handler with the field/intent already resolved.
    """
    active_code = ctx.get("active_project_code") or ctx.get("last_project_code")
    if not active_code:
        return _NO_FIRE

    q_norm = normalize_text(query)

    # A project named in the current turn always outranks active context.
    if "مشروع" in q_norm.split() and _looks_like_new_project_named(query, q_norm):
        return _NO_FIRE

    # Hard stops — these override everything
    if _has_any(q_norm, _PORTFOLIO_OVERRIDES):
        return _NO_FIRE
    if _has_any(q_norm, _CMP_OVERRIDES):
        return _NO_FIRE
    if _has_any(q_norm, _NEW_PROJECT_SIGNALS):
        return _NO_FIRE

    # Check if a new explicit project name appears
    # We don't run the full resolver here (too expensive for a gate),
    # but we check if the query, after stripping known followup words,
    # has enough residual content to suggest a new project is named.
    # The full resolver will still run later if the gate fires and
    # the topic field is None — that's the safe fallback.
    has_possessive = _has_any(q_norm, _POSSESSIVE_SUFFIXES)
    has_assessment = _has_any(q_norm, _ASSESSMENT_MARKERS)
    has_elliptical = _has_any(q_norm, _ELLIPTICAL_MARKERS)

    # Detect requested field
    field_def: FieldDefinition | None = detect_requested_field(query)
    has_field = field_def is not None

    # Check for list followup scope (don't fire gate for list followups)
    if ctx.get("last_result_scope") == "list" or ctx.get("last_result_type") in {
        "portfolio_filter", "portfolio_ranking"
    }:
        # Only fire if possessive suffix makes it clearly about the active project
        if not has_possessive:
            return _NO_FIRE

    # Gate fires if any semantic signal is present
    if has_possessive:
        # Check if the possessive is actually an assessment phrase (وش وضعه / كيف حاله)
        _ASSESSMENT_POSSESSIVES = tuple(normalize_text(v) for v in (
            "وش وضعه","وش وضعها","كيف وضعه","كيف وضعها","ايش وضعه",
            "وش حاله","وش حالها","كيف حاله","كيف حالها",
        ))
        if any(m in q_norm for m in _ASSESSMENT_POSSESSIVES):
            return FollowupDecision(
                fires=True, field=None, intent="assessment",
                confidence=0.97, reason="possessive_assessment",
            )
        field_canonical = field_def.canonical if field_def else _infer_field_from_possessive(q_norm)
        return FollowupDecision(
            fires=True,
            field=field_canonical,
            intent="field_lookup" if field_canonical else "general_followup",
            confidence=0.97,
            reason="possessive_suffix",
        )

    if any(marker in q_norm for marker in tuple(normalize_text(v) for v in (
        "هل المشروع متأخر", "هل المشروع متاخر", "هل هو متأخر", "is it delayed",
    ))):
        return FollowupDecision(fires=True, field="days_remaining", intent="delay",
                                confidence=0.99, reason="explicit_delay_question")

    if has_assessment:
        return FollowupDecision(
            fires=True,
            field=None,
            intent="assessment",
            confidence=0.95,
            reason="assessment_marker",
        )

    if has_field and not _looks_like_new_project_named(query, q_norm):
        return FollowupDecision(
            fires=True,
            field=field_def.canonical,
            intent="field_lookup",
            confidence=0.9,
            reason="field_in_context",
        )

    # Even if _looks_like_new_project_named fired, check if it's actually
    # a project-related question word (not a real project name)
    _QUESTION_WORDS = tuple(normalize_text(v) for v in (
        "متى", "ما", "وش", "كيف", "هل", "لماذا", "ليش", "اين", "فين", "مين",
        "when", "what", "how", "why", "where", "who", "is", "are",
    ))
    if has_field:
        q_tokens = q_norm.split()
        non_question_tokens = [t for t in q_tokens
                               if t not in _QUESTION_WORDS
                               and t not in {normalize_text(v) for v in (
                                   "تاريخ","البداية","النهاية","الحالية","الحالي",
                                   "الابرز","ابرز","المخاطر","المشروع","مشروع",
                               )}]
        if len(non_question_tokens) < 3:
            # Not enough substance to be a real project name
            return FollowupDecision(
                fires=True,
                field=field_def.canonical,
                intent="field_lookup",
                confidence=0.88,
                reason="field_question_in_context",
            )

    if has_elliptical:
        return FollowupDecision(
            fires=True,
            field=field_def.canonical if field_def else None,
            intent="general_followup",
            confidence=0.85,
            reason="elliptical_marker",
        )

    return _NO_FIRE


def _infer_field_from_possessive(q_norm: str) -> str | None:
    """Map possessive suffix to canonical field without full field detection."""
    _MAP = {
        "تكاليف": "total_cost",
        "تكلفت":  "total_cost",
        "ايرادات": "total_revenue",
        "إيرادات": "total_revenue",
        "ربح":    "net_profit",
        "خسارت":  "net_profit",
        "هامش":   "profit_pct",
        "نسبت":   "profit_pct",
        "عقد":    "total_contract_value",
        "باكلوج": "backlog",
        "تقدم":   "progress_completed",
        "حالت":   "status",
        "وضع":    "status",
        "مدير":   "project_manager",
        "تاريخ":  "effective_end_date",
        "ينتهي":  "effective_end_date",
    }
    for key, canonical in _MAP.items():
        if key in q_norm:
            return canonical
    return None


def _looks_like_new_project_named(query: str, q_norm: str) -> bool:
    """
    Heuristic: does the query look like it names a NEW project?
    We use word count of the residual after removing common request words.
    Long residuals (>4 meaningful tokens) suggest a new project name.
    """
    from modules.project_entity_resolver import extract_project_phrase
    phrase = extract_project_phrase(query)
    tokens = [t for t in phrase.split() if len(t) > 2]
    # An explicit "مشروع <name>" needs only one meaningful name token.  The
    # previous four-token threshold caused active-project data leakage.
    if "مشروع" in q_norm.split():
        return len(tokens) >= 1
    return len(tokens) >= 4
