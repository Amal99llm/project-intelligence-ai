"""Explicit boundaries for facts that are not present in verified project data."""

from __future__ import annotations

from dataclasses import dataclass

from modules.semantic_dictionary import normalize_text


@dataclass(frozen=True)
class BoundaryDecision:
    topic: str
    answer_ar: str
    answer_en: str


_BOUNDARIES = (
    (("رقم الجوال", "رقم جوال", "رقم الهاتف", "رقم هاتف", "ايميل", "بريد", "بيانات التواصل", "phone", "email", "contact details"),
     BoundaryDecision("contact", "بيانات التواصل مو موجودة في مصدر بياناتي — المشاريع والعقود فقط.", "Contact details are not in my data source.")),
    (("المقاول", "المقاولين", "contractor"),
     BoundaryDecision("contractor", "ما عندي بيانات مقاولين — بياناتي مرتبطة بالمشاريع والعقود فقط.", "Contractor info isn't in my data source.")),
    (("المورد", "الموردين", "supplier", "vendor"),
     BoundaryDecision("supplier", "بيانات الموردين مو متوفرة في مصدري الحالي.", "Supplier data isn't available in my current source.")),
    (("راتب", "رواتب", "salary", "compensation"),
     BoundaryDecision("salary", "بيانات الرواتب خارج نطاق بياناتي.", "Salary data is outside my scope.")),
    (("عدد الموظفين", "عدد موظفين", "كم موظف", "كم موظفين", "employee count", "number of employees"),
     BoundaryDecision("employee_count", "عدد الموظفين مو موجود في بيانات المشاريع عندي.", "Employee count isn't in my project data.")),
    (("سيرته", "السيره الذاتيه", "معلومات شخصيه", "biography", "personal information"),
     BoundaryDecision("personal", "ما عندي معلومات شخصية موثقة — أقدر أساعدك في بيانات المشاريع فقط.", "I don't have verified personal data.")),
)

import random

_PERSON_RESPONSES_AR = [
    "ما عندي معلومات عن هذا الشخص خارج سياق المشاريع — لو هو مدير مشروع أو موظف في المحفظة أخبرني المشروع وأساعدك.",
    "اسم الشخص مو موجود في بياناتي — بياناتي مرتبطة بالمشاريع والعقود. لو يشتغل على مشروع معين قل لي وين.",
    "ما أعرف عن هذا الشخص من بيانات المشاريع — لو تقصد مدير مشروع أو طرف في عقد، اذكر المشروع وأجيبك.",
]

_PERSON_RESPONSES_EN = [
    "I don't have information about this person outside project data. If they manage a project, let me know which one.",
    "That name isn't in my project data. If they're involved in a specific project, mention it and I'll help.",
]


def classify_boundary(query: str) -> BoundaryDecision | None:
    normalized = normalize_text(query)
    for phrases, decision in _BOUNDARIES:
        if any(normalize_text(phrase) in normalized for phrase in phrases):
            return decision
    # Person/identity questions
    _PERSON_TRIGGERS = ("تعرف ", "هل تعرف ", "مين ", "من هو ", "من هي ", "who is ", "do you know ", "tell me about ")
    if any(normalize_text(t) in normalized for t in _PERSON_TRIGGERS):
        # But NOT if it's asking about a project manager in context ("مين مديره")
        _PROJECT_CONTEXT = ("مديره", "مديرها", "مسؤوله", "ماسكه", "manager")
        if any(normalize_text(t) in normalized for t in _PROJECT_CONTEXT):
            return None  # Let the project lookup handle it
        return BoundaryDecision(
            "unknown_person",
            random.choice(_PERSON_RESPONSES_AR),
            random.choice(_PERSON_RESPONSES_EN),
        )
    return None


def boundary_answer(query: str, decision: BoundaryDecision) -> str:
    is_arabic = any("\u0600" <= char <= "\u06ff" for char in query)
    return decision.answer_ar if is_arabic else decision.answer_en