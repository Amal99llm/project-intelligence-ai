"""
modules/query_builder.py
--------------------------
Section 17 Step 2 — Structured Query Builder.

Turns a natural-language question into a structured query spec using an
LLM function call constrained to modules.query_schema's whitelist. The
model only ever selects which columns/operators/values/sort/aggregation to
use -- it never computes a result. modules.query_schema.validate_query_spec
re-checks everything the model returns against the whitelist before
execution; anything outside the whitelist is rejected outright, never
guessed at or silently widened.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

from openai import AzureOpenAI

import config
from modules import query_schema
from modules.project_entity_resolver import normalize_project_text
from modules.semantic_dictionary import detect_portfolio_operation, detect_semantic_intent
from modules.contract_semantics import parse_future_period_days

logger = logging.getLogger(__name__)

_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        if not config.AZURE_OPENAI_KEY:
            raise QueryBuildError("Azure OpenAI is not configured")
        _openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _openai_client


class QueryBuildError(Exception):
    """Raised when the model fails to produce a spec that survives
    whitelist validation. Callers must treat this as a hard stop, never a
    reason to fall back to freeform LLM reasoning."""


_BUILD_TOOL = {
    "type": "function",
    "function": {
        "name": "build_query",
        "description": (
            "Turn the question into a structured filter/sort/aggregation spec over the "
            "projects table. Never compute a value yourself -- only select whitelisted "
            "columns, operators and literal values taken from the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "enum": sorted(query_schema.FILTERABLE_COLUMNS)},
                            "op": {"type": "string", "enum": sorted(query_schema.OPERATORS)},
                            "value": {
                                "description": "Literal value; use an array of literals only when op is 'in'.",
                                "type": ["number", "string", "array", "null"],
                            },
                            "value2": {"type": ["number", "string", "null"], "description": "Only used when op is 'between'."},
                        },
                        "required": ["column", "op"],
                    },
                },
                "sort": {
                    "type": ["object", "null"],
                    "properties": {
                        "column": {"type": "string", "enum": sorted(query_schema.SORTABLE_COLUMNS)},
                        "direction": {"type": "string", "enum": sorted(query_schema.SORT_DIRECTIONS)},
                    },
                },
                "limit": {"type": ["integer", "null"]},
                "aggregation": {
                    "type": ["object", "null"],
                    "properties": {
                        "func": {"type": "string", "enum": sorted(query_schema.AGGREGATIONS)},
                        "column": {"type": ["string", "null"], "enum": sorted(query_schema.AGGREGATABLE_COLUMNS) + [None]},
                    },
                },
                "aggregate_over_limited_rows": {
                    "type": "boolean",
                    "description": (
                        "Only set true when the question explicitly asks to aggregate over a "
                        "specific top/bottom N rows (e.g. 'total revenue of the top 10 projects "
                        "by backlog'). Requires both 'limit' and 'sort' to also be set. Leave "
                        "false/omitted for any portfolio-wide or filtered total -- a limit must "
                        "never silently shrink what an aggregation covers."
                    ),
                },
            },
            "required": ["filters"],
        },
    },
}

_SYSTEM_PROMPT = """Convert the user's question about a project portfolio into a structured query.
You never compute totals, filter data yourself, or invent values -- you only choose which whitelisted
columns/operators/values from the question belong in the structured spec; a separate deterministic
executor applies it against the real database. Today's date is provided for "days remaining" style
questions. If the question needs no filter (e.g. a portfolio-wide total), return an empty filters list.

For anything about a contract/project ending, expiring, or "within N days", always filter on the
precomputed 'days_remaining' column (e.g. between 0 and N) rather than 'end_date' or 'amended_end_date'
directly -- days_remaining already accounts for amendments and is relative to today's date. Only filter
on 'end_date'/'amended_end_date' directly if the question asks about a specific calendar date.

If the question asks for an aggregation (sum/count/avg/min/max) over the WHOLE filtered result -- e.g.
"total revenue", "how many losing projects" -- never attach a 'limit'; the aggregation must cover every
matching row. Only attach a 'limit' alongside an 'aggregation' when the question explicitly asks to
aggregate over a specific top/bottom N rows (e.g. "total revenue of the top 10 projects by backlog"),
and in that case also set 'aggregate_over_limited_rows' to true and provide a 'sort'.

Respond only by calling build_query."""


def _deterministic_conversational_spec(query: str, today_str: str) -> dict | None:
    """Safe specs for common Saudi executive questions.

    These are meaning-level phrases with a single unambiguous calculation;
    everything else continues through the constrained semantic builder.
    """
    q = normalize_project_text(query)
    today = date.fromisoformat(today_str)
    future_days = parse_future_period_days(query)
    if future_days and any(term in q for term in ("تنتهي", "ينتهي", "تخلص", "ending")):
        return query_schema.validate_query_spec({
            "filters": [{"column": "days_remaining", "op": "between", "value": 0, "value2": future_days}],
            "sort": {"column": "days_remaining", "direction": "ASC"}, "limit": 10,
        })
    if any(term in q for term in ("بدات هالسنه", "بدات هذه السنه", "started this year")):
        return query_schema.validate_query_spec({
            "filters": [{"column": "start_date", "op": "between",
                         "value": date(today.year, 1, 1), "value2": date(today.year, 12, 31)}],
            "sort": {"column": "start_date", "direction": "ASC"}, "limit": 10,
        })
    intent = detect_semantic_intent(query)
    operation = detect_portfolio_operation(query)
    if operation:
        filters = []
        if operation.get("status"):
            statuses = operation["status"]
            filters.append({
                "column": "status",
                "op": "==" if len(statuses) == 1 else "in",
                "value": statuses[0] if len(statuses) == 1 else statuses,
            })
        if operation["operation"] == "count":
            return query_schema.validate_query_spec({
                "filters": filters,
                "aggregation": {"func": "COUNT", "column": None},
            })
        return query_schema.validate_query_spec({
            "filters": filters,
            "sort": {"column": operation["metric"], "direction": operation["direction"]},
            "limit": 1,
        })
    filters = []
    if any(term in q for term in ("المشاريع الجاريه", "المشاريع الجارية", "مشروع جاري", "مشروع شغال", "المشاريع المستمره", "ongoing projects")):
        filters.append({"column": "status", "op": "==", "value": "Ongoing"})
    if any(term in q for term in ("خسائر", "خاسره", "خاسرة", "loss making", "loss-making")):
        filters.append({"column": "net_profit", "op": "<", "value": 0})
    if any(term in q for term in ("قرب انتهاء", "قريبه من expiry", "قريبة من expiry", "close to contract expiry", "close to expiry")):
        filters.append({"column": "days_remaining", "op": "between", "value": 0, "value2": 90})
    if any(term in q for term in ("المتاخره", "المتأخرة", "overdue")):
        filters.append({"column": "days_remaining", "op": "<", "value": 0})
    if any(term in q for term in ("عندها مخاطر", "فيها مخاطر", "recorded risk", "have risk")):
        filters.append({"column": "risk", "op": ">", "value": 0})
    if any(term in q for term in ("المكتمله", "المكتملة", "completed projects")):
        filters.append({"column": "status", "op": "in", "value": ["Completed", "Closed"]})
    contract_threshold = re.search(r"(?:قيمت\w*|worth|contract value)\s*(?:فوق|اكثر من|أكثر من|more than|over|above)\s*(?:sar\s*)?(\d+(?:\.\d+)?)\s*(مليون|million)?", q)
    if contract_threshold:
        multiplier = 1_000_000 if contract_threshold.group(2) else 1
        filters.append({"column": "total_contract_value", "op": ">", "value": float(contract_threshold.group(1)) * multiplier})
    if any(term in q for term in ("منخفضه مقارنه بالمده", "منخفضة مقارنة بالمدة", "behind schedule based on elapsed time", "behind plan")):
        filters.append({"column": "progress_gap", "op": ">=", "value": 20})
    margin_between_match = re.search(
        r"(?:هامش\w*|margin)\s*(?:بين|between)\s*(\d+(?:\.\d+)?)\s*%?\s*"
        r"(?:و\s*|and\s+|to\s+)(\d+(?:\.\d+)?)",
        q,
    )
    if margin_between_match:
        lower, upper = sorted(float(value) for value in margin_between_match.groups())
        filters.append({"column": "profit_pct", "op": "between", "value": lower, "value2": upper})
    margin_match = re.search(r"(?:هامش\w*|margin)\s*(?:اقل من|دون|تحت|below|less than)\s*(\d+(?:\.\d+)?)", q)
    if margin_match:
        filters.append({"column": "profit_pct", "op": "<", "value": float(margin_match.group(1))})
    if filters:
        spec = {"filters": filters}
        if margin_match or margin_between_match:
            spec["sort"] = {"column": "profit_pct", "direction": "ASC"}
            spec["limit"] = 10
        if any(term in q for term in ("كم", "عدد", "count")):
            spec["aggregation"] = {"func": "COUNT", "column": None}
        return query_schema.validate_query_spec(spec)
    if intent in {"ranking_profit", "ranking_best"}:
        return query_schema.validate_query_spec({
            "filters": [], "sort": {"column": "net_profit", "direction": "DESC"}, "limit": 1,
        })
    if intent in {"ranking_loss", "ranking_worst"}:
        return query_schema.validate_query_spec({
            "filters": [], "sort": {"column": "net_profit", "direction": "ASC"}, "limit": 1,
        })
    if intent == "ranking_profit_projects":
        return query_schema.validate_query_spec({
            "filters": [], "sort": {"column": "net_profit", "direction": "DESC"}, "limit": 10,
        })
    if intent == "losing_projects":
        return query_schema.validate_query_spec({
            "filters": [{"column": "net_profit", "op": "<", "value": 0}],
            "sort": {"column": "net_profit", "direction": "ASC"},
        })
    if intent == "profitable_projects":
        return query_schema.validate_query_spec({
            "filters": [{"column": "net_profit", "op": ">", "value": 0}],
            "sort": {"column": "net_profit", "direction": "DESC"},
        })
    if intent == "ranking_contract":
        return query_schema.validate_query_spec({
            "filters": [], "sort": {"column": "total_contract_value", "direction": "DESC"}, "limit": 1,
        })
    if intent == "expiring":
        return query_schema.validate_query_spec({
            "filters": [{"column": "days_remaining", "op": "between", "value": 1, "value2": 30}],
            "sort": {"column": "days_remaining", "direction": "ASC"},
        })
    if intent == "overdue":
        return query_schema.validate_query_spec({
            "filters": [
                {"column": "days_remaining", "op": "<", "value": 0},
                {"column": "status", "op": "in", "value": ["Ongoing", "On-hold", "Pipeline"]},
            ],
            "sort": {"column": "days_remaining", "direction": "ASC"},
        })
    if intent == "health_projects":
        return query_schema.validate_query_spec({
            "filters": [{"column": "project_name_ar", "op": "contains", "value": "الصحي"}],
        })
    if intent == "riyadh_projects":
        return query_schema.validate_query_spec({
            "filters": [{"column": "project_name_ar", "op": "contains", "value": "الرياض"}],
        })
    return None


def is_explicit_portfolio_filter(query: str, today_str: str) -> bool:
    """True when plural portfolio wording has a deterministic filter spec."""
    q = normalize_project_text(query)
    plural_subject = any(term in q for term in ("المشاريع", "مشاريع", "projects"))
    if not plural_subject:
        return False
    spec = _deterministic_conversational_spec(query, today_str)
    return bool(spec and spec.get("filters"))


def build_query(query: str, today_str: str) -> dict:
    deterministic = _deterministic_conversational_spec(query, today_str)
    if deterministic is not None:
        return deterministic
    client = _get_openai()
    response = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Today's date: {today_str}\nQuestion: {query}"},
        ],
        tools=[_BUILD_TOOL],
        tool_choice={"type": "function", "function": {"name": "build_query"}},
        temperature=0,
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise QueryBuildError("model returned no tool call")

    try:
        raw_spec = json.loads(tool_calls[0].function.arguments)
    except json.JSONDecodeError as exc:
        raise QueryBuildError(f"model returned invalid JSON: {exc}") from exc

    try:
        return query_schema.validate_query_spec(raw_spec)
    except query_schema.QueryValidationError as exc:
        logger.warning("Query spec rejected by whitelist: %s | raw=%s", exc, raw_spec)
        raise QueryBuildError(str(exc)) from exc
