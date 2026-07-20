"""Small trusted-tool surface for chat engine v2."""
from __future__ import annotations
import re, unicodedata
from difflib import SequenceMatcher
from modules import business_glossary, data_service

def normalize_arabic(text):
    text = unicodedata.normalize("NFKD", str(text or "")); text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.translate(str.maketrans("أإآٱى", "ااااي")); text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

ALIASES = {"الباحث": "الباحث", "العقاري": "العقار", "العقار": "العقار", "طريق مكة": "طريق مكة", "بطاقة الحج": "بطاقة الحج", "العلا": "العلا", "ناجز": "ناجز", "متم": "متم"}

def search_projects(search_text, status=None, department=None, program=None, manager=None, category=None, limit=5):
    query = normalize_arabic(search_text)
    if query.startswith("و") and " " not in query: query = query[1:]
    query = ALIASES.get(query, query)
    if not query: return []
    resolved = business_glossary.resolve_filters({"status": status, "department": department})
    status, department = resolved.get("status"), resolved.get("department")
    scored = []
    for row in data_service.list_search_rows({"status": status, "department": department, "program": program, "manager": manager, "category": category}):
        values = [row.get("project_name_ar"), row.get("project_name_en"), row.get("project_id"), row.get("wbs"), row.get("program")]
        norms = [normalize_arabic(v) for v in values if v]
        exact = max((1.0 if query == v else 0.96 if query in v else 0 for v in norms), default=0)
        fuzzy = max((SequenceMatcher(None, query, v).ratio() for v in norms), default=0)
        token = max((max(sum(1 for t in query.split() if t in v) / max(len(query.split()), 1),
                         sum(1 for t in v.split() if t in query) / max(len(v.split()), 1)) for v in norms), default=0)
        score = max(exact, fuzzy, token * .9)
        if score >= .54: scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1]["project_name"])); return [{**r, "score": round(s, 3)} for s, r in scored[:max(1, min(int(limit), 20))]]

def get_project_fields(project_identifier, canonical_fields): return data_service.get_fields(project_identifier, canonical_fields)
def filter_projects(filters, sort=None, limit=10):
    return data_service.filter_rows(business_glossary.resolve_filters(filters), sort, limit)
def aggregate_portfolio(metric, aggregation, filters=None, group_by=None):
    return data_service.aggregate(metric, aggregation, business_glossary.resolve_filters(filters), group_by)
def compare_projects(project_identifiers, canonical_fields):
    return [data_service.get_fields(i, canonical_fields) for i in project_identifiers]
def get_contract_context(project_identifier, contract_question):
    project = data_service.get_fields(project_identifier, ["project_name_ar"])
    if not project: raise ValueError("Unknown project")
    from modules.rag_engine import answer_contract_query
    return {"project_id": project_identifier, "project_name": project["project_name"], "answer": answer_contract_query(contract_question, project_identifier, project["project_name"])}