"""Whitelisted deterministic grouped portfolio aggregation."""

from __future__ import annotations

from typing import Any

from modules.intent_schema import AGGREGATIONS, GROUP_FIELDS


def _number(value: Any) -> float:
    try: return 0.0 if value is None else float(value)
    except (TypeError, ValueError): return 0.0


def execute_grouped(projects: list[dict[str, Any]], entities: dict[str, Any]) -> dict[str, Any]:
    group_by = entities.get("group_by")
    aggregation = entities.get("aggregation")
    if group_by not in GROUP_FIELDS or aggregation not in AGGREGATIONS:
        raise ValueError("grouped operation is not whitelisted")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for project in projects:
        key = project.get(group_by)
        if key not in (None, ""):
            buckets.setdefault(str(key), []).append(project)
    value_fields = {
        "sum_total_revenue": "total_revenue", "sum_total_cost": "total_cost",
        "sum_total_contract_value": "total_contract_value",
        "sum_net_profit": "net_profit", "sum_backlog": "backlog",
    }
    rows = []
    for key, members in buckets.items():
        if aggregation == "count_distinct_projects":
            value = len({p.get("project_code") for p in members if p.get("project_code")})
        else:
            value = sum(_number(p.get(value_fields[aggregation])) for p in members)
        rows.append({"group": key, "value": value, "project_codes": [p.get("project_code") for p in members]})
    descending = entities.get("sort_direction", "desc") == "desc"
    rows.sort(key=lambda row: ((-row["value"]) if descending else row["value"], row["group"]))
    return {
        "group_by": group_by, "aggregation": aggregation,
        "rows": rows[:entities.get("limit", 1)], "group_count": len(rows),
    }


def format_grouped(result: dict[str, Any]) -> str:
    if not result["rows"]:
        return "لا تتوفر مجموعات موثقة كافية للإجابة."
    winner = result["rows"][0]
    labels = {
        "project_manager": "مدير المشاريع", "bu": "وحدة الأعمال",
        "segment": "القطاع", "dept": "الإدارة", "status": "الحالة", "project_name_ar": "البرنامج",
    }
    agg = result["aggregation"]
    if agg == "count_distinct_projects":
        count = int(winner["value"])
        noun = "مشروع" if count == 1 else "مشاريع"
        return f"أعلى {labels[result['group_by']]} من حيث عدد المشاريع هو «{winner['group']}»، بعدد {count} {noun}."
    metric = {
        "sum_total_revenue": "إجمالي الإيرادات", "sum_total_cost": "إجمالي التكاليف",
        "sum_total_contract_value": "إجمالي قيمة العقود", "sum_net_profit": "إجمالي صافي الربح",
        "sum_backlog": "إجمالي الأعمال المتبقية",
    }[agg]
    return f"الأعلى في {metric} حسب {labels[result['group_by']]} هو «{winner['group']}»، بقيمة {winner['value']:,.2f} ريال."
