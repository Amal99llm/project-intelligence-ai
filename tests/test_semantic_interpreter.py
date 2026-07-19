"""Unit tests for modules.semantic_interpreter -- Layer 1 of the semantic
interpretation pipeline. The LLM call is always mocked (same pattern as
tests/test_saudi_scenarios.py::test_invalid_llm_contract_is_rejected_to_safe_fallback)
so these tests never touch the network and are deterministic."""

import json
from unittest.mock import patch

from modules.intent_schema import CONTROLLED_FALLBACK, PORTFOLIO_FILTER, SMALL_TALK
from modules.semantic_interpreter import interpret

_PROJECTS = [
    {"project_code": "P1", "dept": "BPO-Specialized Pr", "bu": "BPO", "segment": "BPO",
     "program": "PJ-GP-ALPHA", "project_manager": "Ahmad Al-Otaibi", "status": "Ongoing"},
    {"project_code": "P2", "dept": "BPO - Inspection", "bu": "BPO", "segment": "BPO",
     "program": "PJ-GP-BETA", "project_manager": "Faisal Al-Harbi", "status": "Ongoing"},
]


def _fake_client(arguments: dict | str):
    payload = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)

    class Function:
        pass
    fn = Function()
    fn.arguments = payload

    class Call:
        pass
    call = Call()
    call.function = fn

    class Message:
        pass
    message = Message()
    message.tool_calls = [call]

    class Choice:
        pass
    choice = Choice()
    choice.message = message

    class Completions:
        @staticmethod
        def create(**kwargs):
            return type("Response", (), {"choices": [choice]})()

    return type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()


def _raw(**overrides) -> dict:
    base = {
        "intent": PORTFOLIO_FILTER,
        "scope": "portfolio",
        "entities": {
            "projects": [], "departments": [], "status": [], "programs": [],
            "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
        },
        "operations": {"filter": True, "group_by": None, "sort": None, "limit": None, "compare": False},
        "references": {"active_project": False, "previous_list": False, "comparison": False, "ordinal": None},
        "confidence": 0.95,
        "requires_clarification": False,
    }
    base.update(overrides)
    return base


# ── High confidence, clean entities: no clarification, mentions resolved ──

def test_high_confidence_resolves_status_and_department_to_db_literals():
    raw = _raw(entities={
        "projects": [], "departments": ["إدارة المشاريع المتخصصة"], "status": ["نشطة"],
        "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
    })
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("عطني المشاريع النشطة في إدارة المشاريع المتخصصة", {}, projects=_PROJECTS)
    assert result.intent == PORTFOLIO_FILTER
    assert result.confidence == 0.95
    assert not result.requires_clarification
    # The whole point: the Interpretation the caller sees holds the resolved
    # DB literal, never the raw Arabic phrase and never something the model
    # invented -- the model only ever produced the raw phrase.
    assert result.entities.departments == ["BPO-Specialized Pr"]
    assert result.entities.status == ["Ongoing"]


def test_model_never_needs_to_and_never_does_emit_a_db_literal():
    """The system prompt instructs the model to output raw text only; this
    test pins that the merge step is what performs resolution, not the
    model -- feeding a raw mention that looks nothing like the DB literal
    still resolves correctly only because entity_resolvers did the work."""
    raw = _raw(entities={
        "projects": [], "departments": ["التفتيش"], "status": ["الجارية"],
        "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
    })
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("مشاريع إدارة التفتيش الجارية", {}, projects=_PROJECTS)
    assert result.entities.departments == ["BPO - Inspection"]
    assert result.entities.status == ["Ongoing"]


# ── Confidence gate ───────────────────────────────────────────────────────

def test_low_confidence_requires_clarification():
    raw = _raw(confidence=0.3)
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("مشاريع غامضة جداً", {}, projects=_PROJECTS)
    assert result.requires_clarification
    assert result.confidence == 0.3
    assert result.clarification_question


def test_medium_confidence_with_unresolvable_entity_requires_clarification():
    raw = _raw(confidence=0.7, entities={
        "projects": [], "departments": ["إدارة غير موجودة إطلاقاً"], "status": [],
        "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
    })
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("مشاريع إدارة غير موجودة إطلاقاً", {}, projects=_PROJECTS)
    assert result.requires_clarification
    assert result.unresolved_entity is not None
    assert result.unresolved_entity.kind == "department"
    assert "إدارة غير موجودة إطلاقاً" in result.clarification_question


def test_medium_confidence_with_all_entities_resolved_does_not_clarify():
    raw = _raw(confidence=0.7, entities={
        "projects": [], "departments": ["التفتيش"], "status": [],
        "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
    })
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("مشاريع إدارة التفتيش", {}, projects=_PROJECTS)
    assert not result.requires_clarification
    assert result.entities.departments == ["BPO - Inspection"]


def test_high_confidence_unresolvable_entity_still_clarifies_never_guesses():
    """Even a confident overall classification must not silently drop or
    guess an entity that didn't resolve -- clarification is scoped to just
    that entity, not the whole question."""
    raw = _raw(confidence=0.95, entities={
        "projects": [], "departments": ["قسم لا يمكن حله"], "status": ["نشطة"],
        "programs": [], "business_units": [], "segments": [], "managers": [], "metrics": [], "dates": [],
    })
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("مشاريع قسم لا يمكن حله النشطة", {}, projects=_PROJECTS)
    assert result.requires_clarification
    assert result.unresolved_entity.kind == "department"
    # The entity that DID resolve is still available on the object.
    assert result.entities.status == ["Ongoing"]


# ── Failure handling: never raise, always defer ──────────────────────────

def test_llm_exception_returns_controlled_fallback_not_raise():
    def _raise(**kwargs):
        raise TimeoutError("boom")

    fake_client = type("Client", (), {
        "chat": type("Chat", (), {"completions": type("Completions", (), {"create": staticmethod(_raise)})()})()
    })()
    with patch("modules.semantic_interpreter._get_openai", return_value=fake_client):
        result = interpret("أي سؤال", {}, projects=_PROJECTS)
    assert result.intent == CONTROLLED_FALLBACK
    assert result.confidence == 0.0
    assert result.method == "llm_error"


def test_invalid_json_returns_controlled_fallback():
    client = _fake_client("not valid json {{{")
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("أي سؤال", {}, projects=_PROJECTS)
    assert result.intent == CONTROLLED_FALLBACK
    assert result.method == "llm_error"


def test_unsupported_intent_is_coerced_to_controlled_fallback():
    raw = _raw(intent="totally_made_up_intent")
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("سؤال عادي", {}, projects=_PROJECTS)
    assert result.intent == CONTROLLED_FALLBACK


def test_out_of_range_confidence_is_clamped():
    raw = _raw(confidence=5.0)
    client = _fake_client(raw)
    with patch("modules.semantic_interpreter._get_openai", return_value=client):
        result = interpret("سؤال عادي وواضح", {}, projects=_PROJECTS)
    assert result.confidence == 1.0


# ── Deterministic pre-pass: small talk never calls the LLM ──────────────

def test_small_talk_short_circuits_without_llm_call():
    calls = {"count": 0}

    def _get_openai_spy():
        calls["count"] += 1
        raise AssertionError("small talk must never reach the LLM")

    with patch("modules.semantic_interpreter._get_openai", side_effect=_get_openai_spy):
        result = interpret("السلام عليكم", {}, projects=_PROJECTS)
    assert result.intent == SMALL_TALK
    assert result.method == "deterministic_pre_pass"
    assert calls["count"] == 0
