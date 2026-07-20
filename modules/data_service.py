"""Validated, deterministic database operations for chat engine v2."""
from __future__ import annotations
from sqlalchemy import func
from modules.database import BacklogProject, get_session
from modules.business_glossary import FIELD_MAP

ALLOWED_FILTERS = {"status": "status", "department": "dept", "program": "program", "manager": "project_manager", "category": "category"}

def _identity(row):
    return {"project_id": row.project_code, "project_name": row.project_name_ar or row.project_name_en or row.project_code}

def list_search_rows(filters=None):
    filters = filters or {}
    with get_session() as session:
        query = session.query(BacklogProject)
        for key, value in filters.items():
            if key not in ALLOWED_FILTERS: raise ValueError(f"Unsupported filter: {key}")
            if value is not None: query = query.filter(getattr(BacklogProject, ALLOWED_FILTERS[key]).ilike(f"%{value}%"))
        return [{**_identity(r), "project_name_ar": r.project_name_ar, "project_name_en": r.project_name_en,
                 "wbs": r.wbs, "program": r.program, "status": r.status, "department": r.dept,
                 "manager": r.project_manager, "category": r.category} for r in query.all()]

def get_fields(project_id, fields):
    invalid = [f for f in fields if f not in FIELD_MAP and f != "effective_end_date"]
    if invalid: raise ValueError(f"Unsupported fields: {invalid}")
    with get_session() as session:
        row = session.query(BacklogProject).filter(BacklogProject.project_code == str(project_id)).one_or_none()
        if row is None: return None
        values = {}
        for field in fields:
            if field == "effective_end_date": values[field] = row.amended_end_date or row.end_date
            elif field == "project_definition": values[field] = (row.raw_data or {}).get("Project Definition")
            else: values[field] = getattr(row, FIELD_MAP[field])
        return {**_identity(row), "fields": values}

def filter_rows(filters, sort=None, limit=10):
    rows = list_search_rows(filters)
    if sort:
        field, direction = sort.get("field"), sort.get("direction", "asc")
        if field not in FIELD_MAP: raise ValueError("Unsupported sort")
        detailed = [get_fields(r["project_id"], [field]) for r in rows]
        detailed.sort(key=lambda r: (r["fields"][field] is None, r["fields"][field]), reverse=direction == "desc")
        return detailed[:max(1, min(int(limit), 100))]
    return rows[:max(1, min(int(limit), 100))]

def aggregate(metric, aggregation, filters=None, group_by=None):
    if metric not in FIELD_MAP or aggregation not in {"sum", "avg", "min", "max", "count"}: raise ValueError("Unsupported aggregate")
    if group_by and group_by not in ALLOWED_FILTERS: raise ValueError("Unsupported group")
    column = getattr(BacklogProject, FIELD_MAP[metric]); operation = func.count(column) if aggregation == "count" else getattr(func, aggregation)(column)
    with get_session() as session:
        query = session.query(operation)
        for key, value in (filters or {}).items():
            if key not in ALLOWED_FILTERS: raise ValueError("Unsupported filter")
            query = query.filter(getattr(BacklogProject, ALLOWED_FILTERS[key]).ilike(f"%{value}%"))
        if group_by:
            group = getattr(BacklogProject, ALLOWED_FILTERS[group_by]); return [{"group": g, "value": v} for g, v in query.add_columns(group).group_by(group).all()]
        return {"value": query.scalar()}
