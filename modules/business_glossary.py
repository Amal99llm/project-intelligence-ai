"""Canonical business vocabulary for chat engine v2."""
from __future__ import annotations

FIELD_MAP = {
    "project_type": "project_type", "project_category": "category", "program": "program",
    "project_definition": "project_definition", "wbs_pc": "wbs_pc", "wbs": "wbs",
    "project_name_en": "project_name_en", "project_name_ar": "project_name_ar",
    "profit_center": "pc", "cost_center": "cc", "business_unit": "bu", "department": "dept",
    "segment": "segment", "officer_name": "officer_name", "customer_id": "customer_id",
    "support_document": "support_document", "start_date": "start_date", "end_date": "end_date",
    "amended_end_date": "amended_end_date", "project_manager": "project_manager",
    "contract_value": "contract_value", "contract_amendments": "amendment_crs",
    "total_contract_value": "total_contract_value", "previous_revenue": "previous_years_rev",
    "revenue": "revenue_current", "other_income": "other_income", "total_revenue": "total_revenue",
    "backlog": "backlog", "previous_cost": "previous_years_cost", "cost_of_revenue": "cost_of_revenue",
    "other_cost": "other_cost", "total_cost": "total_cost", "profit_until_2025": "pm_up_to_2025",
    "margin_until_2025": "pm_pct_up_to_2025", "gross_profit_2026": "gp_2026",
    "profit_2026": "pm_2026", "margin_2026": "pm_pct_2026", "profit_loss": "pl",
    "profit_margin": "profit_pct", "status": "status", "progress": "progress_completed",
    "purchase_orders": "po", "human_resources_cost": "hr", "external_cost": "other_external",
    "internal_cost": "other_internal", "risk": "risk", "contingency": "contingency",
    "planned_cost": "total_planned_cost", "planned_profit": "planned_profit",
    "planned_margin": "planned_pm_pct", "variance": "variance", "etc_cost": "etc_cost",
    "etc_revenue": "etc_revenue", "net": "net_etc", "net_percentage": "etc_pct", "note": "note",
    "accrued_revenue": "acc_rev", "performance_bond": "pb", "advance_payment": "adv",
    "accounts_receivable": "ar", "contract_assets": "contract_assets", "accounts_payable": "ap",
    "accrued_expenses": "acc_exp", "contract_liabilities": "contract_liabilities",
    "deferred_cost": "deferred_cost", "open_po": "open_po", "ecl_ar": "ecl_ar",
    "ecl_accrued_revenue": "ecl_acc_rev",
}

LABELS = {
    "project_name_ar": "اسم المشروع", "project_name_en": "اسم المشروع بالإنجليزية",
    "project_definition": "تعريف المشروع", "status": "حالة المشروع",
    "revenue": "الإيراد", "total_revenue": "إجمالي الإيرادات", "backlog": "قيمة الأعمال المتبقية",
    "contract_value": "قيمة العقد", "total_contract_value": "إجمالي قيمة العقد",
    "contract_amendments": "تعديلات العقد", "total_cost": "إجمالي التكلفة",
    "profit_loss": "الربح والخسارة", "profit_margin": "هامش الربح", "progress": "نسبة الإنجاز",
    "project_manager": "مدير المشروع", "officer_name": "المسؤول", "effective_end_date": "تاريخ الانتهاء",
    "etc_cost": "التكلفة المتوقعة المتبقية", "risk": "المخاطر", "contingency": "الاحتياطي",
    "variance": "الانحراف", "etc_revenue": "الإيراد المتوقع المتبقي", "net": "الصافي",
    "accounts_receivable": "الذمم المدينة", "accounts_payable": "الذمم الدائنة",
    "ecl_ar": "الخسائر الائتمانية للذمم", "ecl_accrued_revenue": "الخسائر الائتمانية للإيراد المستحق",
    "note": "الملاحظة",
}

CURRENCY_FIELDS = {"contract_value", "contract_amendments", "total_contract_value", "previous_revenue",
 "revenue", "other_income", "total_revenue", "backlog", "previous_cost", "cost_of_revenue", "other_cost",
 "total_cost", "profit_until_2025", "gross_profit_2026", "profit_2026", "profit_loss", "purchase_orders",
 "human_resources_cost", "external_cost", "internal_cost", "risk", "contingency", "planned_cost",
 "planned_profit", "variance", "etc_cost", "etc_revenue", "net", "accrued_revenue", "performance_bond",
 "advance_payment", "accounts_receivable", "contract_assets", "accounts_payable", "accrued_expenses",
 "contract_liabilities", "deferred_cost", "open_po", "ecl_ar", "ecl_accrued_revenue"}
PERCENT_FIELDS = {"profit_margin", "margin_until_2025", "margin_2026", "planned_margin", "net_percentage", "progress"}

COMPOSITES = {
 "summary": ["project_name_ar", "project_definition", "status", "progress", "project_manager", "total_contract_value", "revenue", "backlog", "profit_loss", "profit_margin", "risk", "effective_end_date"],
 "situation": ["status", "progress", "revenue", "backlog", "profit_loss", "profit_margin", "risk", "effective_end_date"],
 "financial": ["total_contract_value", "total_revenue", "backlog", "total_cost", "profit_loss", "profit_margin", "etc_cost", "etc_revenue", "net", "accounts_receivable", "accounts_payable"],
 "risks": ["risk", "contingency", "variance", "ecl_ar", "ecl_accrued_revenue", "note"],
 "revenue_and_cost": ["revenue", "total_cost"],
}

def column_for(field: str) -> str:
    if field == "effective_end_date": return "amended_end_date"
    if field not in FIELD_MAP: raise ValueError(f"Unsupported canonical field: {field}")
    return FIELD_MAP[field]
