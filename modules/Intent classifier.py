"""Semantic intent classification with safe deterministic fast paths."""

from __future__ import annotations

import json
import logging
import re

from openai import AzureOpenAI

import config
from modules import query_schema
from modules.project_entity_resolver import normalize_project_text
from modules.semantic_dictionary import (
    detect_requested_field, detect_semantic_intent, is_contract_document_question,
)

logger = logging.getLogger(__name__)
_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _openai_client


_PROJECT_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}-[A-Z]{2,}-[A-Z0-9]+\b", re.IGNORECASE)

_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Classify the user's question into exactly one BI query category.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": sorted(query_schema.INTENT_CATEGORIES)},
            },
            "required": ["intent"],
        },
    },
}

_SYSTEM_PROMPT = """Classify Arabic, Saudi-dialect, or English questions about a project portfolio.
Judge the full meaning, not one keyword. A mention of عقد is contract_analysis only when asking about
clauses, terms, penalties, obligations, or contract text. Financial questions mentioning contract
value remain financial_analysis. Understand Saudi phrases such as وش وضعه، مين ماسكه، كم جاب، أكثر
مشروع ربحان، قربت تخلص، وأغلى مشروع.

Categories:
- executive_kpi: portfolio-wide totals or named KPIs
- project_lookup: one named/coded project or a contextual question about one project
- financial_analysis: financial figures for a subset of projects
- ranking: highest, lowest, best, worst, biggest, or most profitable
- comparison: comparison of projects, units, or periods
- filtering: which projects meet a status, location, sector, date, or financial condition
- contract_analysis: actual clauses, terms, penalties, obligations, or contract-document content
- general_conversation: greetings or unrelated conversation

Respond only by calling classify_intent."""


def _saudi_fast_path(query: str) -> str | None:
    q = normalize_project_text(query)
    semantic = detect_semantic_intent(query)
    if is_contract_document_question(query):
        return query_schema.INTENT_CONTRACT_ANALYSIS
    if semantic and semantic.startswith("ranking_"):
        return query_schema.INTENT_RANKING
    if semantic in {"losing_projects", "profitable_projects", "expiring", "overdue", "health_projects", "riyadh_projects"}:
        return query_schema.INTENT_FILTERING
    if detect_requested_field(query) or any(marker in q for marker in (
        "وش وضع المشروع", "ملخص مشروع", "تفاصيل مشروع", "project summary",
    )):
        return query_schema.INTENT_PROJECT_LOOKUP
    return None


def classify(query: str) -> str:
    if _PROJECT_CODE_PATTERN.search(query or ""):
        return query_schema.INTENT_PROJECT_LOOKUP

    fast = _saudi_fast_path(query)
    if fast:
        return fast

    client = _get_openai()
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_intent"}},
        temperature=0,
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        logger.warning("Classifier returned no tool call; using general conversation")
        return query_schema.INTENT_GENERAL_CONVERSATION
    try:
        intent = json.loads(tool_calls[0].function.arguments).get("intent")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return query_schema.INTENT_GENERAL_CONVERSATION
    return intent if intent in query_schema.INTENT_CATEGORIES else query_schema.INTENT_GENERAL_CONVERSATION