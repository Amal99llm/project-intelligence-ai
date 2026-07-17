"""
modules/session_context.py  — v2
Extended conversation state with full structured metadata per the
architectural spec. Stores ONLY structured metadata, never raw text or
financial figures.

New fields added over v1:
    active_project_code          -- current focused project
    active_project_display_name  -- display name for that project
    active_project_depth         -- how many turns deep in this project
    last_answered_fields         -- list of field canonicals answered so far
    conversation_phase           -- opening | deep_dive | assessment | summary
    last_result_type             -- project_summary | project_kpi | portfolio_kpi |
                                    portfolio_filter | portfolio_ranking | comparison |
                                    executive_attention | list_followup
    last_project_list            -- list of project_codes from last list result
    last_comparison              -- {codes: [c1,c2], field: str|None}
    last_executive_result        -- {intent: str, top_project: str|None}
    last_requested_metric        -- canonical field or KPI name last discussed
"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any

SESSION_TTL_SECONDS = 30 * 60

_lock = threading.Lock()
_store: dict[str, dict[str, Any]] = {}


def _empty_context() -> dict[str, Any]:
    return {
        # ── Legacy fields (keep for backward compatibility) ──────────────────
        "last_project_code": None,
        "last_project_display_name": None,
        "last_kpi_name": None,
        "last_requested_field": None,
        "last_result_scope": None,
        "last_list_project_codes": None,
        "last_list_intent": None,
        "last_list_filters": None,
        "last_list_sort": None,
        "pending_project_candidates": None,
        "pending_clarification_type": None,
        "pending_project_confirmation": None,
        "last_disambiguation_options": [],
        "last_disambiguation_query": None,
        "selected_disambiguation_index": None,
        # ── New structured metadata ──────────────────────────────────────────
        "active_project_code": None,
        "active_project_display_name": None,
        "active_project_depth": 0,
        "last_answered_fields": [],        # list[str] — canonical field names
        "conversation_phase": "opening",   # opening | deep_dive | assessment | summary
        "last_result_type": None,          # what the last answer was about
        "last_project_list": [],           # list[str] — project codes
        "last_comparison": None,           # {codes:[c1,c2], field:str|None}
        "last_executive_result": None,     # {intent:str, top_project:str|None}
        "last_requested_metric": None,     # last field/KPI discussed
        "_updated_at": time.time(),
    }


def new_session_id() -> str:
    return uuid.uuid4().hex


def _purge_expired_locked() -> None:
    now = time.time()
    expired = [
        sid for sid, entry in _store.items()
        if now - entry["_updated_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _store.pop(sid, None)


def get_context(session_id: str) -> dict[str, Any]:
    with _lock:
        _purge_expired_locked()
        entry = _store.get(session_id)
        if entry is None:
            entry = _empty_context()
            _store[session_id] = entry
        # Context contains lists/dicts.  A shallow copy lets callers mutate the
        # shared session state without taking the lock (for example by
        # appending to ``last_project_list``).  Return an isolated snapshot.
        return deepcopy(entry)


def update_context(session_id: str, **fields: Any) -> None:
    with _lock:
        _purge_expired_locked()
        entry = _store.setdefault(session_id, _empty_context())

        # ── Sync legacy ↔ new fields ─────────────────────────────────────────
        if "last_project_code" in fields and fields["last_project_code"]:
            fields.setdefault("active_project_code", fields["last_project_code"])
        if "last_project_display_name" in fields and fields["last_project_display_name"]:
            fields.setdefault("active_project_display_name", fields["last_project_display_name"])
        if "active_project_code" in fields and fields["active_project_code"]:
            fields.setdefault("last_project_code", fields["active_project_code"])
        if "active_project_display_name" in fields and fields["active_project_display_name"]:
            fields.setdefault("last_project_display_name", fields["active_project_display_name"])

        # ── Auto-advance conversation phase ──────────────────────────────────
        project_changed = False
        if "last_result_type" in fields:
            rtype = fields["last_result_type"]
            depth = entry.get("active_project_depth", 0)
            new_code = fields.get("active_project_code") or fields.get("last_project_code")
            old_code = entry.get("active_project_code") or entry.get("last_project_code")

            if new_code and new_code != old_code:
                project_changed = True
                # New project → reset depth and phase
                fields["active_project_depth"] = 1
                fields["conversation_phase"] = "opening"
                fields["last_answered_fields"] = []
            elif rtype in {"project_summary", "project_kpi"}:
                fields["active_project_depth"] = depth + 1
                if depth + 1 >= 3:
                    fields.setdefault("conversation_phase", "deep_dive")
            elif rtype in {"assessment", "executive_attention"}:
                fields.setdefault("conversation_phase", "assessment")

        # Apply assessment phase override AFTER reset (handles first-turn assessment)
        if fields.get("last_result_type") in {"assessment", "executive_attention"}:
            fields["conversation_phase"] = "assessment"

        # ── Track answered fields ─────────────────────────────────────────────
        if "last_requested_metric" in fields and fields["last_requested_metric"]:
            # A metric supplied on the first turn of a new project belongs to
            # that project only; do not resurrect the previous project's list.
            existing = [] if project_changed else list(entry.get("last_answered_fields") or [])
            metric = fields["last_requested_metric"]
            if metric not in existing:
                existing.append(metric)
            fields["last_answered_fields"] = existing[-8:]  # keep last 8

        # ── pending_project_confirmation helpers ──────────────────────────────
        if "pending_project_confirmation" in fields:
            pending = fields["pending_project_confirmation"]
            candidates = [dict(c) for c in pending.get("candidates", [])] if pending else None
            fields.setdefault("pending_project_candidates", candidates)
            fields.setdefault("pending_clarification_type", pending.get("kind") if pending else None)

        entry.update(fields)
        entry["_updated_at"] = time.time()


def clear_context(session_id: str) -> None:
    with _lock:
        _store.pop(session_id, None)


def _reset_all_for_tests() -> None:
    with _lock:
        _store.clear()
