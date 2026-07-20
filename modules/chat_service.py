"""Azure-assisted, trusted-tool-only Arabic executive chat engine v2."""
from __future__ import annotations

from datetime import date
import json
import re
from typing import Any

import config
from modules import project_tools, session_context
from modules.business_glossary import COMPOSITES, FIELD_MAP
from modules.conversation_state import v2_activate_project, v2_set_metric, v2_state
from modules.response_guard import (
    deterministic_answer, deterministic_tool_answer, validate, validate_generic,
)
from modules.time_utils import riyadh_today

YES = {"اي", "ايوه", "نعم", "صح", "تمام"}
ORDINALS = {"1": 0, "الأول": 0, "الاول": 0, "2": 1, "الثاني": 1, "3": 2, "الثالث": 2}
CORRECTIONS = ("مو", "مش", "ليس", "لا أقصد", "ما أقصد", "أقصد", "قصدي", "قلت", "بدل", "غير")
CANONICAL_FIELDS = sorted(set(FIELD_MAP) | {"effective_end_date"})

METRICS = [
    ("backlog", ("باقي إيراد", "باقي الايراد", "الإيراد المتبقي", "الايراد المتبقي",
                 "الإيرادات المتبقية", "فلوس بالمشروع", "الباقي المالي", "باقي من شغل",
                 "شغل العقد", "الأعمال المتبقية", "الباكلوق", "باكلوق", "remaining revenue", "remaining work")),
    ("revenue_and_cost", ("كم جاب وكم كلف", "جاب وكم كلف", "الإيراد والتكلفة")),
    ("total_revenue", ("إجمالي الإيرادات", "اجمالي الايرادات")),
    ("revenue", ("الإيراد", "كم جاب")),
    ("total_contract_value", ("إجمالي العقد", "اجمالي العقد", "العقد كاملة", "العقد كامل")),
    ("contract_value", ("قيمة العقد",)), ("contract_amendments", ("التعديلات", "تعديلات العقد")),
    ("etc_cost", ("باقي تكلفة", "تكلفة متوقعة")), ("total_cost", ("صرفنا", "إجمالي التكلفة", "كم كلف")),
    ("profit_margin", ("الهامش", "هامش", "نسبة الربح")), ("profit_loss", ("الربح", "p&l")),
    ("project_manager", ("مديره", "المدير", "مدير المشروع", "مين ماسكه", "ماسكه")),
    ("officer_name", ("المسؤول",)), ("progress", ("نسبة الإنجاز", "نسبة الانجاز")),
    ("effective_end_date", ("متى ينتهي", "متى يخلص", "النهاية", "ينتهي", "يخلص")),
]

_client = None


def _get_openai():
    global _client
    if _client is None:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            timeout=config.AI_REQUEST_TIMEOUT_SECONDS,
        )
    return _client


def _function(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


TOOL_DEFINITIONS = [
    _function("search_projects", "ابحث عن مشروع عندما يذكر المستخدم اسماً أو رمزاً جديداً. أعد مرشحين ولا تخمّن.", {
        "search_text": {"type": "string"}, "status": {"type": ["string", "null"]},
        "department": {"type": ["string", "null"]}, "program": {"type": ["string", "null"]},
        "manager": {"type": ["string", "null"]}, "category": {"type": ["string", "null"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["search_text"]),
    _function("get_project_fields", "اجلب حقولاً موثقة للمشروع النشط فقط.", {
        "project_identifier": {"type": "string"},
        "canonical_fields": {"type": "array", "items": {"type": "string", "enum": CANONICAL_FIELDS},
                             "minItems": 1, "uniqueItems": True}}, ["project_identifier", "canonical_fields"]),
    _function("filter_projects", "اعرض مشاريع حسب مرشحات موثقة.", {
        "filters": {"type": "object"}, "sort": {"type": ["object", "null"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ["filters"]),
    _function("aggregate_portfolio", "احسب تجميعاً للمحفظة داخل الأداة فقط.", {
        "metric": {"type": "string", "enum": CANONICAL_FIELDS},
        "aggregation": {"type": "string", "enum": ["sum", "avg", "min", "max", "count"]},
        "filters": {"type": ["object", "null"]}, "group_by": {"type": ["string", "null"]}},
        ["metric", "aggregation"]),
    _function("compare_projects", "قارن مشاريع موثقة وحقولاً محددة.", {
        "project_identifiers": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "canonical_fields": {"type": "array", "items": {"type": "string", "enum": CANONICAL_FIELDS}, "minItems": 1}},
        ["project_identifiers", "canonical_fields"]),
    _function("get_contract_context", "أجب عن سؤال عقدي لمشروع واحد موثق.", {
        "project_identifier": {"type": "string"}, "contract_question": {"type": "string"}},
        ["project_identifier", "contract_question"]),
]

SELECTOR_PROMPT = """أنت مفسر محادثة عربي سعودي. افهم اللهجة والأخطاء والتصحيحات والمتابعات.
اختر استدعاء أداة واحدة فقط من الأدوات المقدمة. لا تجب المستخدم، لا تحسب، لا تكتب SQL، ولا تنشئ قيمة أو مشروعاً.
الأولوية: تصحيح الرسالة الحالية، ثم مقياسها الصريح، ثم مشروعها الصريح، ثم المشروع النشط، ثم المقياس السابق.
إذا ذُكر مشروع جديد استخدم search_projects. إذا كان المشروع النشط هو المقصود استخدم get_project_fields واطلب فقط الحقول اللازمة.
عبارات الباقي المالي أو باقي الشغل تعني backlog. "كم جاب وكم كلف" تعني revenue وtotal_cost.
استخدم project_identifier الموجود بالحالة فقط؛ لا تخترع معرفاً."""

COMPOSER_PROMPT = """اكتب جواباً عربياً سعودياً طبيعياً ومختصراً من نتيجة الأداة الموثقة فقط.
لا تحسب ولا تضف رقماً أو سبباً أو توصية. اذكر اسم المشروع. استخدم ريال للقيم المالية و% للنسب.
لا تعرض أسماء الحقول الداخلية. أجب عن سؤال الرسالة الحالية تحديداً، لا عن المقياس السابق."""


def _minimal_state(state):
    return {key: state.get(key) for key in (
        "active_scope", "active_project_id", "active_project_name", "last_active_project_id",
        "active_metric", "previous_metric", "last_intent", "last_project_candidates", "active_comparison")}


def _select_tool(query, state):
    response = _get_openai().chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": SELECTOR_PROMPT},
                  {"role": "user", "content": json.dumps({"message": query, "state": _minimal_state(state)}, ensure_ascii=False)}],
        tools=TOOL_DEFINITIONS, tool_choice="required", parallel_tool_calls=False,
    )
    calls = response.choices[0].message.tool_calls or []
    if len(calls) != 1:
        raise ValueError("Azure must select exactly one trusted tool")
    call = calls[0]
    name = call.function.name
    if name not in {item["function"]["name"] for item in TOOL_DEFINITIONS}:
        raise ValueError("Unsupported tool")
    arguments = json.loads(call.function.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Invalid tool arguments")
    return name, arguments


def _compose(query, state, tool_result, fields, fallback):
    payload = {"message": query, "state": _minimal_state(state), "verified_tool_result": tool_result}
    response = _get_openai().chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "system", "content": COMPOSER_PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
    )
    answer = (response.choices[0].message.content or "").strip()
    return answer if validate(answer, tool_result, fields) else fallback


def _compose_generic(query, state, tool_name, tool_result, arguments):
    fallback = deterministic_tool_answer(tool_name, tool_result, arguments)
    payload = {"message": query, "state": _minimal_state(state), "verified_tool_result": tool_result}
    try:
        response = _get_openai().chat.completions.create(
            model=config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "system", "content": COMPOSER_PROMPT},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
        )
        text = (response.choices[0].message.content or "").strip()
        return text if validate_generic(text, tool_result, arguments.get("metric")) else fallback
    except Exception:
        return fallback


def _explicit_metric(message):
    norm = project_tools.normalize_arabic(message)
    positive = norm
    if any(word in message for word in CORRECTIONS) and ("أقصد" in message or "اقصد" in norm or "،" in message or "," in message):
        positive = project_tools.normalize_arabic(re.split(r"أقصد|اقصد|،|,", message)[-1])
    if "باقي" in norm and ("ينتهي" in norm or "يخلص" in norm or "النهايه" in norm) and not any(x in norm for x in ("ايراد", "فلوس", "مالي", "شغل")):
        return "days_remaining"
    for metric, phrases in METRICS:
        if any(project_tools.normalize_arabic(phrase) in positive for phrase in phrases):
            return metric
    if "ملخص" in norm or "باختصار" in norm:
        return "summary"
    if "وضعه المالي" in norm:
        return "financial"
    if "مخاطر" in norm:
        return "risks"
    if "وش وضعه" in norm:
        return "situation"
    return None


def _selection(message, candidates):
    norm = project_tools.normalize_arabic(message)
    if "اخر واحد" in norm and candidates:
        return candidates[-1]
    index = ORDINALS.get(message.strip())
    if index is None:
        index = ORDINALS.get(norm)
    if index is not None and index < len(candidates):
        return candidates[index]
    if norm in YES and len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if norm == project_tools.normalize_arabic(candidate["project_name"]):
            return candidate
    return None


def _clarify(candidates):
    return "تقصد أي مشروع؟\n" + "\n".join(f"{index}. {item['project_name']}" for index, item in enumerate(candidates, 1))


def _state_update(session_id, state):
    session_context.update_context(session_id, **state)


def _activate_search(query, state, arguments):
    allowed = {"search_text", "status", "department", "program", "manager", "category", "limit"}
    if set(arguments) - allowed or not str(arguments.get("search_text", "")).strip():
        raise ValueError("Invalid search call")
    candidates = project_tools.search_projects(**arguments)
    if not candidates:
        return state, None, {"answer": "ما لقيت مشروعًا موثقًا بهذا الوصف.", "query_type": "not_found"}
    strong = [candidate for candidate in candidates if candidate["score"] >= .88]
    if len(strong) == 1:
        chosen = strong[0]
        return v2_activate_project(state, chosen["project_id"], chosen["project_name"]), chosen, None
    if len(candidates) == 1:
        chosen = candidates[0]
        return v2_activate_project(state, chosen["project_id"], chosen["project_name"]), chosen, None
    state["last_project_candidates"] = candidates
    state["last_intent"] = "clarification"
    return state, None, {"answer": _clarify(candidates), "query_type": "clarification"}


def _verified_project_result(state, arguments):
    allowed = {"project_identifier", "canonical_fields"}
    if set(arguments) != allowed:
        raise ValueError("Invalid field call")
    identifier = str(arguments["project_identifier"])
    if identifier in {"active", "current", "المشروع النشط"}:
        identifier = state.get("active_project_id")
    if not identifier or identifier != state.get("active_project_id"):
        raise ValueError("Model may only read the verified active project")
    fields = arguments["canonical_fields"]
    if not isinstance(fields, list) or not fields or any(field not in CANONICAL_FIELDS for field in fields):
        raise ValueError("Invalid canonical fields")
    result = project_tools.get_project_fields(identifier, fields)
    if not result:
        raise ValueError("Unknown active project")
    return result, fields


def _dispatch_non_project(name, arguments):
    if name == "filter_projects":
        allowed = {"filters", "sort", "limit"}
        if set(arguments) - allowed:
            raise ValueError("Invalid filter call")
        return project_tools.filter_projects(**arguments)
    if name == "aggregate_portfolio":
        allowed = {"metric", "aggregation", "filters", "group_by"}
        if set(arguments) - allowed:
            raise ValueError("Invalid aggregate call")
        return project_tools.aggregate_portfolio(**arguments)
    if name == "compare_projects":
        if set(arguments) != {"project_identifiers", "canonical_fields"}:
            raise ValueError("Invalid comparison call")
        result = project_tools.compare_projects(**arguments)
        if not result or any(item is None for item in result):
            raise ValueError("Unverified comparison project")
        return result
    if name == "get_contract_context":
        if set(arguments) != {"project_identifier", "contract_question"}:
            raise ValueError("Invalid contract call")
        return project_tools.get_contract_context(**arguments)
    raise ValueError("Unsupported dispatcher")


def _answer_project(query, state, result, fields, today, correction, use_azure):
    days = None
    if fields == ["effective_end_date"] and _explicit_metric(query) == "days_remaining":
        end = result["fields"].get("effective_end_date")
        days = (end - today).days if isinstance(end, date) else None
        result["verified_calculations"] = {"days_remaining": days}
    fallback = deterministic_answer(result, fields, days_remaining=days, correction=correction)
    if use_azure:
        try:
            return _compose(query, state, result, fields, fallback)
        except Exception:
            return fallback
    return fallback


def _deterministic_turn(query, state, today):
    correction = any(word in query for word in CORRECTIONS)
    candidates = project_tools.search_projects(query, limit=5)
    chosen = None
    if candidates:
        strong = [candidate for candidate in candidates if candidate["score"] >= .88]
        if len(strong) == 1:
            chosen = strong[0]
        elif len(candidates) > 1:
            state["last_project_candidates"] = candidates
            return state, {"answer": _clarify(candidates), "query_type": "clarification"}
        elif candidates[0]["score"] >= .72:
            chosen = candidates[0]
    if chosen:
        state = v2_activate_project(state, chosen["project_id"], chosen["project_name"])
    metric = _explicit_metric(query) or state.get("active_metric")
    if not state.get("active_project_id"):
        return state, {"answer": "حدد لي المشروع المقصود، لو سمحت.", "query_type": "clarification"}
    if not metric:
        return state, {"answer": f"تم اختيار مشروع «{state['active_project_name']}». وش حاب تعرف عنه؟", "query_type": "project_selected"}
    state = v2_set_metric(state, metric)
    fields = COMPOSITES.get(metric, ["effective_end_date" if metric == "days_remaining" else metric])
    result = project_tools.get_project_fields(state["active_project_id"], fields)
    response = _answer_project(query, state, result, fields, today, correction, False)
    return state, {"answer": response, "query_type": "project_fields"}


def answer(query, session_id, today=None):
    today = today or riyadh_today()
    state = v2_state(session_context.get_context(session_id))
    norm = project_tools.normalize_arabic(query)
    pending = state.get("last_project_candidates") or []
    chosen = _selection(query, pending) if pending else None
    if chosen:
        state = v2_activate_project(state, chosen["project_id"], chosen["project_name"])
        _state_update(session_id, state)
        return {"answer": f"تم اختيار مشروع «{chosen['project_name']}». وش حاب تعرف عنه؟", "query_type": "project_selected"}
    if "ارجع للمحفظه" in norm or "ارجع للمحفظة" in query:
        state["active_scope"] = "portfolio"
        _state_update(session_id, state)
        return {"answer": "رجعنا للمحفظة. وش حاب تعرف عنها؟", "query_type": "portfolio"}
    if "ارجع" in norm and state.get("last_active_project_id"):
        previous = project_tools.get_project_fields(state["last_active_project_id"], ["project_name_ar"])
        previous_tokens = project_tools.normalize_arabic(previous["project_name"]).split() if previous else []
        named_previous = any(len(token.lstrip("ال")) >= 4 and token.lstrip("ال") in norm for token in previous_tokens)
        if previous and (named_previous or len(norm.split()) <= 2):
            state = v2_activate_project(state, previous["project_id"], previous["project_name"])
            _state_update(session_id, state)
            return {"answer": f"رجعنا لمشروع «{previous['project_name']}». وش حاب تعرف؟", "query_type": "project"}
    if norm.startswith("ارجع "):
        target = norm.removeprefix("ارجع ").lstrip("ل")
        matches = project_tools.search_projects(target, limit=5)
        if len(matches) == 1:
            state = v2_activate_project(state, matches[0]["project_id"], matches[0]["project_name"])
            _state_update(session_id, state)
            return {"answer": f"رجعنا لمشروع «{matches[0]['project_name']}». وش حاب تعرف؟", "query_type": "project"}
        if len(matches) > 1:
            state["last_project_candidates"] = matches
            _state_update(session_id, state)
            return {"answer": _clarify(matches), "query_type": "clarification"}

    azure_enabled = bool(config.AZURE_OPENAI_KEY)
    if azure_enabled:
        try:
            name, arguments = _select_tool(query, state)
            if name == "search_projects":
                state, chosen, immediate = _activate_search(query, state, arguments)
                if immediate:
                    _state_update(session_id, state)
                    return immediate
                metric = _explicit_metric(query)
                if metric:
                    state = v2_set_metric(state, metric)
                    fields = COMPOSITES.get(metric, ["effective_end_date" if metric == "days_remaining" else metric])
                    result = project_tools.get_project_fields(state["active_project_id"], fields)
                    text = _answer_project(query, state, result, fields, today, any(w in query for w in CORRECTIONS), True)
                    _state_update(session_id, state)
                    return {"answer": text, "query_type": "project_fields"}
                _state_update(session_id, state)
                return {"answer": f"تم اختيار مشروع «{chosen['project_name']}». وش حاب تعرف عنه؟", "query_type": "project_selected"}
            if name == "get_project_fields":
                result, fields = _verified_project_result(state, arguments)
                metric = next((key for key, value in COMPOSITES.items() if value == fields), fields[0])
                state = v2_set_metric(state, metric)
                text = _answer_project(query, state, result, fields, today, any(w in query for w in CORRECTIONS), True)
                state["last_intent"] = "project_fields"
                _state_update(session_id, state)
                return {"answer": text, "query_type": "project_fields"}
            verified = _dispatch_non_project(name, arguments)
            text = _compose_generic(query, state, name, verified, arguments)
            _state_update(session_id, state)
            return {"answer": text, "query_type": name}
        except Exception:
            pass

    state, response = _deterministic_turn(query, state, today)
    _state_update(session_id, state)
    return response
