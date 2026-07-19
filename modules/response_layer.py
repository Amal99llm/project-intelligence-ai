"""
modules/response_layer.py
--------------------------
An addition to Layer 5 (response composition) for the new semantic
interpretation path: an internal-name leak guard applied to every answer
that path returns, before it reaches the caller.

This does NOT replace modules.response_formatter/modules.response_composer
(merging those is future work, deferred per the architecture plan) -- it
adds one new guarantee on top of whichever of them produced the text:
outbound text must never contain a raw DB column identifier (e.g.
"profit_pct", "total_contract_value") or a raw truncated department
literal (e.g. "BPO-Specialized Pr"), and an Arabic answer must never
contain a raw English status enum word (e.g. "Ongoing") instead of its
Arabic translation. A user has no way to interpret any of these, and last
turn's bug showed exactly this kind of leak reaching a real answer.

Detection is deliberately narrow to avoid false positives on ordinary
prose: only identifier-shaped tokens (containing an underscore, which
never appears in natural Arabic/English sentences) and the distinctive
truncated department strings are substring-checked; the 5 status enum
words are checked with word boundaries and only against Arabic text
(they're legitimate English words in an English-language answer).
"""

from __future__ import annotations

import re

from modules.semantic_dictionary import DEPT_ALIASES, FIELD_DEFINITIONS, STATUS_ALIASES


class InternalLeakError(Exception):
    """Raised when outbound text contains a raw internal identifier or a
    raw, untranslated DB literal that must have been converted to business
    language before reaching the user."""


def _is_arabic(text: str) -> bool:
    return any("؀" <= char <= "ۿ" for char in text)


# Identifier-shaped canonical field names (contain an underscore) --
# unambiguous: these never appear in natural prose in either language.
_IDENTIFIER_TOKENS: tuple[str, ...] = tuple(sorted(
    (field.canonical for field in FIELD_DEFINITIONS if "_" in field.canonical),
    key=len, reverse=True,
))

# Truncated department literals -- distinctive "BPO"/"TS"-prefixed source
# strings, never natural prose in any language.
_DEPT_LITERAL_TOKENS: tuple[str, ...] = tuple(sorted(DEPT_ALIASES.keys(), key=len, reverse=True))

# Raw English status enum words -- legitimate in an English answer, but a
# leak (an un-translated raw DB literal) in an Arabic one.
_STATUS_LITERAL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(status) for status in sorted(STATUS_ALIASES, key=len, reverse=True)) + r")\b"
)


def assert_no_internal_leakage(text: str) -> None:
    """Raise InternalLeakError if `text` leaks a raw internal identifier or
    an untranslated DB literal. No-op on falsy input."""
    if not text:
        return
    for token in _IDENTIFIER_TOKENS:
        if token in text:
            raise InternalLeakError(f"internal column identifier leaked into response: {token!r}")
    for token in _DEPT_LITERAL_TOKENS:
        if token in text:
            raise InternalLeakError(f"raw department DB literal leaked into response: {token!r}")
    if _is_arabic(text):
        match = _STATUS_LITERAL_PATTERN.search(text)
        if match:
            raise InternalLeakError(f"untranslated status literal leaked into Arabic response: {match.group()!r}")


_STATUS_LABELS_AR = {
    "Ongoing": "جاري", "Completed": "مكتمل", "Closed": "مغلق",
    "On-hold": "متوقف مؤقتًا", "On Hold": "متوقف مؤقتًا", "Pipeline": "قيد الإعداد",
}


def translate_group_label(column: str, raw_value: str) -> str:
    """Business-Arabic label for a raw grouped-analytics bucket value.
    Only `dept`/`status` need translation -- their DB literal is a raw
    English/truncated code (see modules.semantic_dictionary.DEPT_ALIASES/
    STATUS_ALIASES); other groupable columns (project_manager, bu, segment)
    already store human-readable values and pass through unchanged."""
    if column == "dept":
        return DEPT_ALIASES.get(raw_value, (raw_value,))[0]
    if column == "status":
        return _STATUS_LABELS_AR.get(raw_value, raw_value)
    return raw_value
