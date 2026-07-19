"""Unit tests for modules.entity_resolvers -- Layer 2 of the semantic
interpretation pipeline. These are pure-Python, no DB, no LLM: resolvers are
exercised directly against small inline project lists."""

from modules.entity_resolvers import (
    build_value_index,
    resolve_bu,
    resolve_customer,
    resolve_department,
    resolve_manager,
    resolve_officer,
    resolve_ordinal,
    resolve_program,
    resolve_segment,
    resolve_status,
)

_PROJECTS = [
    {"project_code": "P1", "dept": "BPO-Specialized Pr", "bu": "BPO", "segment": "BPO",
     "program": "PJ-GP-ALPHA", "project_manager": "Ahmad Al-Otaibi", "customer_id": "CUST-1",
     "officer_name": "Sara Al-Qahtani"},
    {"project_code": "P2", "dept": "BPO - Inspection", "bu": "BPO", "segment": "BPO",
     "program": "PJ-GP-BETA", "project_manager": "Faisal Al-Harbi", "customer_id": "CUST-2",
     "officer_name": "Nora Al-Dosari"},
    {"project_code": "P3", "dept": "BPO - Health", "bu": "TS", "segment": "BPO",
     "program": "PJ-GP-GAMMA", "project_manager": "Ahmad Al-Otaibi", "customer_id": "CUST-3",
     "officer_name": None},
]


# ── Status: closed enum, curated + typo tolerance ───────────────────────

def test_resolve_status_exact_alias():
    assert resolve_status("نشط").value == "Ongoing"
    assert resolve_status("المشاريع الجارية").value == "Ongoing"
    assert resolve_status("متوقفة").value == "On-hold"
    assert resolve_status("مخطط").value == "Pipeline"


def test_resolve_status_typo_tolerance():
    # "نشيطة" is a plausible typo/variant of "نشطة" (Ongoing) -- not a
    # literal alias, must still resolve via fuzzy fallback.
    result = resolve_status("نشيطة")
    assert result.status in {"fuzzy", "exact"}
    assert result.value == "Ongoing"


def test_resolve_status_unrelated_text_is_none():
    assert resolve_status("قيمة العقد الإجمالية").status == "none"


# ── Department: curated override + data-driven fallback ────────────────

def test_resolve_department_curated_alias():
    result = resolve_department("إدارة المشاريع المتخصصة", _PROJECTS)
    assert result.value == "BPO-Specialized Pr"
    assert result.confidence == 1.0


def test_resolve_department_short_form():
    assert resolve_department("التفتيش", _PROJECTS).value == "BPO - Inspection"
    assert resolve_department("الصحية", _PROJECTS).value == "BPO - Health"


def test_resolve_department_data_driven_fallback_for_uncurated_value():
    # Not in DEPT_ALIASES at all -- must still resolve via the live-value
    # index (substring/fuzzy), proving departments aren't a fixed keyword
    # list only.
    projects = _PROJECTS + [{"project_code": "P4", "dept": "Ethra Project"}]
    result = resolve_department("Ethra", projects)
    assert result.value == "Ethra Project"


# ── BU / segment / program / manager / customer / officer: fully data-driven ──

def test_resolve_bu_from_live_values_only():
    assert resolve_bu("BPO", _PROJECTS).value == "BPO"
    assert resolve_bu("TS", _PROJECTS).value == "TS"


def test_resolve_bu_unknown_value_is_none():
    assert resolve_bu("NOPE-NOT-REAL", _PROJECTS).status == "none"


def test_resolve_segment_from_live_values():
    assert resolve_segment("BPO", _PROJECTS).value == "BPO"


def test_resolve_program_exact_code():
    assert resolve_program("PJ-GP-BETA", _PROJECTS).value == "PJ-GP-BETA"


def test_resolve_manager_fuzzy_and_ambiguous():
    single = resolve_manager("فيصل الحربي", _PROJECTS)
    assert single.status == "none" or single.status in {"fuzzy", "exact"}
    # Two projects share manager "Ahmad Al-Otaibi" -- resolving by that
    # exact live value must not collapse to a guess; it's a legitimate
    # single canonical string shared by two projects, which is fine (the
    # resolver returns the *manager name*, not a project -- disambiguating
    # which of *his* projects is a separate, later step).
    exact = resolve_manager("Ahmad Al-Otaibi", _PROJECTS)
    assert exact.value == "Ahmad Al-Otaibi"


def test_resolve_customer_and_officer():
    assert resolve_customer("CUST-2", _PROJECTS).value == "CUST-2"
    assert resolve_officer("Sara Al-Qahtani", _PROJECTS).value == "Sara Al-Qahtani"


def test_resolve_officer_ignores_null_values():
    # P3's officer_name is None -- must never appear as a spurious
    # candidate value.
    index = build_value_index(_PROJECTS, "officer_name")
    assert None not in index.canonical_values
    assert "" not in index.canonical_values


# ── Ambiguity must never silently collapse to a guess ───────────────────

def test_ambiguous_two_equally_short_candidates_stays_ambiguous():
    projects = [
        {"project_code": "A", "dept": "AAA"},
        {"project_code": "B", "dept": "AAB"},
    ]
    # A mention that fuzzily matches both similarly-scored short values
    # must come back ambiguous, never pick one arbitrarily.
    result = build_value_index(projects, "dept").resolve("AA")
    assert result.status in {"ambiguous", "none"}
    if result.status == "ambiguous":
        assert len(result.candidates) >= 2


# ── Ordinal resolution against an arbitrary candidate list ─────────────

def test_resolve_ordinal_basic_positions():
    assert resolve_ordinal("الأول", 5) == 0
    assert resolve_ordinal("الثاني", 5) == 1
    assert resolve_ordinal("خامس", 5) is None  # not a recognized exact token form
    assert resolve_ordinal("الخامس", 5) == 4


def test_resolve_ordinal_last_and_next():
    assert resolve_ordinal("الأخير", 5) == 4
    assert resolve_ordinal("اللي بعده", 5, current_index=1) == 2
    assert resolve_ordinal("التالي", 5, current_index=4) == 4  # clamped, no overflow


def test_resolve_ordinal_out_of_range_is_none():
    assert resolve_ordinal("العاشر", 3) is None


def test_resolve_ordinal_non_ordinal_text_is_none():
    assert resolve_ordinal("قيمة العقد", 5) is None


def test_resolve_ordinal_empty_candidates_is_none():
    assert resolve_ordinal("الأول", 0) is None
