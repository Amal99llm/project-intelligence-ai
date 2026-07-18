"""Central schema-driven vocabulary for project and financial conversations."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal


DataType = Literal["text", "code", "money", "number", "percentage", "date", "days"]

_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCTUATION = re.compile(r"[^\w\s%&]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_REPEATED_ARABIC = re.compile(r"([\u0621-\u064A])\1{2,}")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي",
    "ؤ": "و", "ئ": "ي", "ـ": "",
})

# Narrow business-name typo corrections. These are token-level corrections,
# not global Arabic phonetic substitutions: unrelated words and short names
# therefore keep their original spelling and still face the normal resolver
# confidence safeguards.
_CONTROLLED_ARABIC_TYPOS = {
    "الباحص": "الباحث",
    "الباحس": "الباحث",
    "باحص": "باحث",
    "باحس": "باحث",
}


def normalize_digits(value: Any) -> str:
    """Convert Arabic-Indic and Persian digits without changing other text."""
    return unicodedata.normalize("NFKC", str(value or "")).translate(_DIGITS)


def normalize_text(value: Any, *, fold_ta_marbuta: bool = True) -> str:
    """Conservative Arabic/English normalization used only for detection."""
    text = normalize_digits(value).casefold()
    text = _DIACRITICS.sub("", text).translate(_LETTERS)
    if fold_ta_marbuta:
        text = text.replace("ة", "ه")
    text = _REPEATED_ARABIC.sub(r"\1\1", text)
    text = _PUNCTUATION.sub(" ", text.replace("_", " "))
    text = _SPACES.sub(" ", text).strip()
    return " ".join(_CONTROLLED_ARABIC_TYPOS.get(token, token) for token in text.split())


@dataclass(frozen=True)
class FieldDefinition:
    canonical: str
    source_column: str
    label_ar: str
    label_en: str
    aliases: tuple[str, ...]
    saudi_aliases: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()
    data_type: DataType = "text"
    unit: str | None = None
    project_specific: bool = True
    portfolio_aggregatable: bool = False
    calculated: bool = False
    show_by_default: bool = False
    methodology_on_request_only: bool = True

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        values = (self.label_ar, self.label_en, *self.aliases, *self.saudi_aliases, *self.abbreviations)
        return tuple(dict.fromkeys(normalize_text(value) for value in values if value))


def _f(canonical, source, ar, en, aliases=(), saudi=(), abbr=(), dtype="text", unit=None,
       aggregate=False, calculated=False, default=False):
    return FieldDefinition(
        canonical, source, ar, en, tuple(aliases), tuple(saudi), tuple(abbr), dtype, unit,
        True, aggregate, calculated, default, True,
    )


FIELD_DEFINITIONS = (
    _f("project_type", "Type", "نوع المشروع", "Project type", ("النوع", "type")),
    _f("category", "Category", "الفئة", "Category", ("تصنيف المشروع", "category")),
    _f("program", "Program", "البرنامج", "Program", ("اسم البرنامج", "program")),
    _f("project_code", "Project Definition", "رمز المشروع", "Project code", ("كود المشروع", "رقم المشروع", "project code"), ("ورقمه", "وش رقمه"), ("code",), "code", default=True),
    _f("wbs_pc", 'WBS "PC"', "رمز WBS PC", "WBS PC", ("wbs pc",), abbr=("wbspc",), dtype="code"),
    _f("wbs", "WBS", "رمز WBS", "WBS", ("work breakdown structure",), ("وال wbs", "وش ال wbs"), ("wbs",), "code"),
    _f("project_name_en", "Project Name (English)", "اسم المشروع بالإنجليزية", "English project name", ("english name", "project name english")),
    _f("project_name_ar", "Project Name (Arabic)", "اسم المشروع", "Arabic project name", ("اسم المشروع بالعربي", "project name"), default=True),
    _f("pc", "PC", "رمز PC", "PC", ("project control code",), abbr=("pc",), dtype="code"),
    _f("cc", "CC", "رمز CC", "CC", ("cost center", "مركز التكلفة"), abbr=("cc",), dtype="code"),
    _f("bu", "BU", "وحدة الأعمال", "Business unit", ("business unit", "قطاع الاعمال"), ("وال bu", "وش ال bu"), ("bu",), "text"),
    _f("dept", "Dept.", "الإدارة", "Department", ("القسم", "department", "dept")),
    _f("segment", "Segment", "القطاع", "Segment", ("segment", "الشريحة")),
    _f("officer_name", "Officer Name", "مسؤول المشروع", "Officer name", ("الأوفيسر", "الضابط", "officer", "officer name"), ("مين الاوفيسر",)),
    _f("customer_id", "Costumer ID", "رقم العميل", "Customer ID", ("العميل", "customer id", "client id", "مالك المشروع", "صاحب المشروع", "الجهة المستفيدة", "الجهة"), ("والعميل", "مالكه", "عميله", "جهته"), dtype="code"),
    _f("support_document", "Support Document", "المستند الداعم", "Support document", ("وثيقة الدعم", "support document")),
    _f("start_date", "Start Date", "تاريخ البداية", "Start date", ("بداية المشروع", "متى بدأ", "start date", "when did it start", "when does it start"), dtype="date"),
    _f("end_date", "End Date", "تاريخ الانتهاء الأصلي", "Original end date", ("original end date",), dtype="date"),
    _f("amended_end_date", "Amended End Date CR", "تاريخ الانتهاء المعدل", "Amended end date", ("التاريخ المعدل", "amended end date"), dtype="date"),
    _f("effective_end_date", "Amended End Date CR / End Date", "تاريخ الانتهاء", "Effective end date", ("النهاية", "end date", "completion date", "when does it end", "when will it end", "when does it expire"), ("متى ينتهي", "متى يخلص", "ومتى ينتهي"), dtype="date", calculated=True, default=True),
    _f("days_remaining", "Effective End Date", "الأيام المتبقية", "Days remaining", ("كم باقي عليه", "remaining days", "time left", "how much time is left", "remaining time"), ("وش باقي عليه",), dtype="days", calculated=True),
    _f("project_manager", "Project Manager Name", "مدير المشروع", "Project manager", ("المسؤول عن المشروع", "project manager", "pm name", "who manages", "who is the manager", "manager", "managed by"), ("مديره", "مين مديره", "مين ماسكه", "ومديره"), default=True),
    _f("contract_value", "Contract Value", "قيمة العقد الأساسية", "Base contract value", ("base contract value",), dtype="money", unit="SAR", aggregate=True),
    _f("amendment_crs", "Amendment (CRs)", "تعديلات العقد", "Contract amendments", ("طلبات التغيير", "amendments", "crs"), dtype="money", unit="SAR", aggregate=True),
    _f("total_contract_value", "Total Contract Value", "إجمالي قيمة العقد", "Total contract value", ("قيمة العقد", "قيمة المشروع", "contract value", "total contract value", "about the contract"), ("عقده", "كم عقده", "وعقده", "وضع العقد", "عن العقد"), dtype="money", unit="SAR", aggregate=True, default=True),
    _f("previous_years_rev", "Previous Years Revenue", "إيرادات السنوات السابقة", "Previous years revenue", ("previous revenue",), dtype="money", unit="SAR", aggregate=True),
    _f("revenue_current", "Revenue", "إيرادات الفترة الحالية", "Current-period revenue", ("current revenue", "revenue current"), dtype="money", unit="SAR", aggregate=True),
    _f("other_income", "Other Income", "إيرادات أخرى", "Other income", ("other income",), dtype="money", unit="SAR", aggregate=True),
    _f("total_revenue", "Total Revenue", "إجمالي الإيرادات", "Total revenue", ("الإيراد", "الإيرادات", "total revenue", "revenue"), ("كم جاب", "وش دخل", "وإيراداته"), dtype="money", unit="SAR", aggregate=True, default=True),
    _f("backlog", "Backlog", "الأعمال المتبقية", "Backlog", ("المتبقي المالي", "قيمة الأعمال المتبقية", "الباكلوق", "backlog"), ("والباكلوق",), dtype="money", unit="SAR", aggregate=True, default=True),
    _f("previous_years_cost", "Previous Years Cost", "تكاليف السنوات السابقة", "Previous years cost", ("previous cost",), dtype="money", unit="SAR", aggregate=True),
    _f("cost_of_revenue", "Cost of Revenue", "تكلفة الإيرادات", "Cost of revenue", ("cost of revenue",), dtype="money", unit="SAR", aggregate=True),
    _f("other_cost", "Other Cost", "تكاليف أخرى", "Other cost", ("other cost",), dtype="money", unit="SAR", aggregate=True),
    _f("total_cost", "Total Cost", "إجمالي التكاليف", "Total cost", ("التكلفة", "المصروف", "total cost", "cost", "التكاليف", "تكاليف"), ("تكلفته", "تكاليفه", "تكاليفها", "كم كلف", "وتكلفته", "وتكاليفه"), dtype="money", unit="SAR", aggregate=True, default=True),
    _f("pm_up_to_2025", "PM up to 2025", "ربح حتى 2025", "PM up to 2025", ("pm up to 2025",), dtype="money", unit="SAR", aggregate=True),
    _f("pm_pct_up_to_2025", "PM% up to 2025", "هامش الربح حتى 2025", "PM% up to 2025", ("margin up to 2025",), dtype="percentage"),
    _f("gp_2026", "GP 2026", "الربح الإجمالي 2026", "GP 2026", ("gross profit 2026",), abbr=("gp 2026",), dtype="money", unit="SAR", aggregate=True),
    _f("pm_2026", "PM 2026", "ربح 2026", "PM 2026", ("profit 2026",), dtype="money", unit="SAR", aggregate=True),
    _f("pm_pct_2026", "PM% 2026", "هامش ربح 2026", "PM% 2026", ("margin 2026",), dtype="percentage"),
    _f("pl", "P&L", "الربح والخسارة", "Profit and loss", ("صافي الربح", "الربح", "الخسارة", "profit", "loss"), ("ربحه", "كم ربح", "كم خسر", "وربحه", "هل هو خسران", "خسران"), ("p&l", "pl"), "money", "SAR", True, default=True),
    _f("profit_pct", "PM%", "هامش الربح", "Profit margin", ("الهامش", "نسبة الربح", "هامش ربح", "profit margin", "margin", "نسبه الربح"), ("ربحيته", "وهامشه", "هامشه", "هامشها", "كيف ربحيته", "وربحيته", "ربحيتها", "ربحيه"), ("pm%",), "percentage", calculated=True, default=True),
    _f("status", "Status", "الحالة", "Status", ("وضع المشروع", "status", "الحالة الحالية"), ("وش وضعه", "وش صار عليه", "وش علومه", "حالته الحالية", "وحالته", "حالته"), default=True),
    _f("progress_completed", "Progress Completed", "نسبة الإنجاز", "Progress completed", ("الإنجاز", "progress", "completion", "وين وصل", "فين وصل", "كم انجز"), ("وين وصلنا", "فين وصلنا", "تقدمه", "تقدمها", "وتقدمه", "كم خلص", "كم انتهى"), dtype="percentage", default=True),
    _f("po", "PO", "أوامر الشراء", "Purchase orders", ("purchase order",), abbr=("po",), dtype="money", unit="SAR", aggregate=True),
    _f("hr", "HR", "تكلفة الموارد البشرية", "HR cost", ("تكلفة الموظفين", "human resources"), abbr=("hr",), dtype="money", unit="SAR", aggregate=True),
    _f("other_external", "Other (External)", "تكاليف خارجية أخرى", "Other external cost", ("external cost",), dtype="money", unit="SAR", aggregate=True),
    _f("other_internal", "Other (Internal)", "تكاليف داخلية أخرى", "Other internal cost", ("internal cost",), dtype="money", unit="SAR", aggregate=True),
    _f("risk", "Risk", "المخاطر", "Risk", ("risk",), ("وش مخاطره", "هل فيه خطر"), dtype="number", default=True),
    _f("contingency", "Contingency", "الاحتياطي", "Contingency", ("contingency",), dtype="money", unit="SAR", aggregate=True),
    _f("total_planned_cost", "Total Planned Cost", "إجمالي التكلفة المخططة", "Total planned cost", ("planned cost", "التكلفة المخططة"), dtype="money", unit="SAR", aggregate=True),
    _f("planned_profit", "Planned Profit", "الربح المخطط", "Planned profit", ("planned profit",), dtype="money", unit="SAR", aggregate=True),
    _f("planned_pm_pct", "Planned PM%", "هامش الربح المخطط", "Planned profit margin", ("planned margin",), dtype="percentage"),
    _f("variance", "Var.", "الانحراف", "Variance", ("الفرق عن الخطة", "variance"), ("طيب وش الفرق", "قارنها بالخطة"), abbr=("var",), dtype="percentage"),
    _f("etc_cost", "ETC (Cost)", "التكلفة المتوقعة حتى الإكمال", "ETC cost", ("estimated cost to complete",), abbr=("etc cost",), dtype="money", unit="SAR", aggregate=True),
    _f("etc_revenue", "ETC (Revenue)", "الإيراد المتوقع حتى الإكمال", "ETC revenue", ("estimated revenue to complete",), abbr=("etc revenue",), dtype="money", unit="SAR", aggregate=True),
    _f("net_etc", "Net", "صافي ETC", "Net ETC", ("net etc",), dtype="money", unit="SAR", aggregate=True),
    _f("etc_pct", "%", "نسبة ETC", "ETC percentage", ("etc percentage",), dtype="percentage"),
    _f("note", "Note", "الملاحظات", "Notes", ("آخر ملاحظة", "note", "notes"), ("وش ملاحظاته", "وملاحظاته")),
    _f("acc_rev", "Acc Rev", "الإيراد المستحق", "Accrued revenue", ("الإيراد المستحق", "accrued revenue"), abbr=("acc rev",), dtype="money", unit="SAR", aggregate=True),
    _f("pb", "PB", "الفوترة المرحلية", "Progress billing", ("progress billing",), abbr=("pb",), dtype="money", unit="SAR", aggregate=True),
    _f("adv", "Adv", "دفعات مقدمة", "Advances", ("advance", "دفعة مقدمة"), abbr=("adv",), dtype="money", unit="SAR", aggregate=True),
    _f("ar", "AR", "الذمم المدينة", "Accounts receivable", ("المبالغ المستحقة", "accounts receivable"), abbr=("ar",), dtype="money", unit="SAR", aggregate=True),
    _f("contract_assets", "Contract Assets", "أصول العقود", "Contract assets", ("contract assets",), dtype="money", unit="SAR", aggregate=True),
    _f("ap", "AP", "الذمم الدائنة", "Accounts payable", ("accounts payable",), abbr=("ap",), dtype="money", unit="SAR", aggregate=True),
    _f("acc_exp", "Acc Exp", "المصروف المستحق", "Accrued expense", ("accrued expense",), abbr=("acc exp",), dtype="money", unit="SAR", aggregate=True),
    _f("contract_liabilities", "Contract Liabilities", "التزامات العقود", "Contract liabilities", ("contract liabilities",), dtype="money", unit="SAR", aggregate=True),
    _f("deferred_cost", "Deferred Cost", "التكلفة المؤجلة", "Deferred cost", ("deferred cost",), dtype="money", unit="SAR", aggregate=True),
    _f("open_po", "Open PO", "أوامر الشراء المفتوحة", "Open purchase orders", ("open purchase orders",), abbr=("open po",), dtype="money", unit="SAR", aggregate=True),
    _f("ecl_ar", "ECL (AR)", "الخسائر الائتمانية المتوقعة للذمم", "ECL AR", ("expected credit loss ar",), abbr=("ecl ar",), dtype="money", unit="SAR", aggregate=True),
    _f("ecl_acc_rev", "ECL (Acc Rev)", "الخسائر الائتمانية المتوقعة للإيراد المستحق", "ECL accrued revenue", ("expected credit loss accrued revenue",), abbr=("ecl acc rev",), dtype="money", unit="SAR", aggregate=True),
)

FIELDS = {field.canonical: field for field in FIELD_DEFINITIONS}

# Portfolio KPI vocabulary belongs beside the field vocabulary so routing,
# calculation explanation, and project-field lookup share one language map.
KPI_ALIASES = {
    "profit_margin": ("هامش الربح", "نسبة هامش الربح", "profit margin"),
    "profit_loss": ("صافي الربح", "الربح والخسارة", "p&l", "profit loss", "net profit"),
    "total_contract_value": ("إجمالي قيمة العقود", "قيمة العقود", "قيمة العقد", "total contract value", "contract value"),
    "revenue": ("إجمالي الإيرادات", "الإيرادات", "كم جاب", "total revenue", "revenue"),
    "cost": ("إجمالي التكاليف", "التكاليف", "total cost", "cost"),
    "backlog": ("backlog", "باك لوق", "الأعمال المتبقية"),
    "total_projects": ("إجمالي المشاريع", "عدد المشاريع", "total projects"),
    "active_projects": ("المشاريع النشطة", "active projects"),
    "losing_projects": ("المشاريع الخاسرة", "losing projects"),
    "completed_projects": ("المشاريع المنتهية", "المشاريع المكتملة", "المشاريع المغلقة", "completed projects", "finished projects", "closed projects"),
    "contracts_expiring_soon": ("العقود المنتهية قريبًا", "العقود التي تنتهي قريبًا", "contracts expiring soon"),
    "amendments_total": ("تعديلات العقود", "إجمالي تعديلات العقود"),
    "current_year_revenue": ("إيرادات الفترة الحالية", "current revenue"),
    "current_year_cost": ("تكاليف الفترة الحالية", "current cost"),
}


def _alias_in_query(alias: str, normalized_query: str) -> bool:
    if not alias:
        return False
    if " " in alias or len(alias) > 3:
        if alias in normalized_query:
            return True
    tokens = normalized_query.split()
    for token in tokens:
        variants = {token}
        if len(token) > 3 and token[0] in "وفبل":
            variants.add(token[1:])
        if alias in variants:
            return True
    return False


def detect_requested_fields(query: str) -> list[FieldDefinition]:
    normalized = normalize_text(query)
    matches: list[tuple[int, FieldDefinition]] = []
    for field in FIELD_DEFINITIONS:
        best = max((len(alias) for alias in field.normalized_aliases if _alias_in_query(alias, normalized)), default=0)
        if best:
            matches.append((best, field))
    matches.sort(key=lambda item: (-item[0], item[1].canonical))
    # Keep explicitly distinct fields, but suppress shorter aliases embedded
    # in a stronger semantic match (e.g. contract_value inside total_contract_value).
    strongest = matches[0][0] if matches else 0
    selected = [field for length, field in matches if length >= strongest - 2] if matches else []
    by_name = {field.canonical: field for field in FIELD_DEFINITIONS}
    # Coordinated questions explicitly request distinct concepts; preserving
    # both is more important than suppressing a shorter embedded alias.
    concepts = {
        "start_date": ("البدايه", "البداية", "متى بدا", "متى بدأ"),
        "effective_end_date": ("النهايه", "النهاية", "متى ينتهي", "متى يخلص"),
        "project_manager": ("مديره", "مدير المشروع", "مين مدير"),
        "status": ("حالته", "الحاله", "الحالة"),
        "progress_completed": ("نسبه انجازه", "نسبة إنجازه", "نسبه الانجاز", "نسبة الإنجاز"),
        "backlog": ("اعمال متبقيه", "أعمال متبقية", "باقي فيه", "backlog", "باك لوق"),
        "pl": ("ربحه", "وربحه", "صافي ربحه", "كم ربح"),
        "profit_pct": ("هامشه", "وهامشه", "نسبه ربحه", "نسبة ربحه"),
        "total_cost": ("صرفنا فيه", "كم صرفنا", "المصروف عليه"),
        "total_revenue": ("دخلنا منه", "وش دخلنا", "ايراده"),
    }
    for canonical, aliases in concepts.items():
        if any(normalize_text(alias) in normalized for alias in aliases) and by_name[canonical] not in selected:
            selected.append(by_name[canonical])
    return selected


def detect_requested_field(query: str) -> FieldDefinition | None:
    fields = detect_requested_fields(query)
    return fields[0] if fields else None


METHODOLOGY_MARKERS = tuple(normalize_text(value) for value in (
    "كيف حسبتها", "كيف حسبته", "من أي عمود", "وش المعادلة", "اشرح الحساب",
    "وضح الحساب", "how was it calculated", "which column", "what is the formula",
    "كيف حسبت", "كيف تحسب", "كيف تم احتساب", "ما المعادلة", "ما هي المعادلة",
    "مصدر الرقم", "مصدر المؤشر", "source column", "formula", "how is", "how was", "calculated",
    "explain the calculation", "why is this value correct", "ليش الرقم صحيح",
))


def is_methodology_question(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in METHODOLOGY_MARKERS)


CONTRACT_DOCUMENT_MARKERS = tuple(normalize_text(value) for value in (
    "شروط الدفع", "شروط الفسخ", "غرامة", "الالتزامات", "بنود العقد", "بند العقد",
    "sla", "مدة الإشعار", "ضمان", "شروط التمديد", "payment terms", "termination clause",
    "penalty", "obligations", "notice period", "warranty", "extension terms",
    "اعطني عقد مشروع", "أعطني عقد مشروع", "متى ينتهي العقد", "contract document",
))


def is_contract_document_question(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in CONTRACT_DOCUMENT_MARKERS)


SUMMARY_MARKERS = tuple(normalize_text(value) for value in (
    "ملخص", "زبدة المشروع", "المختصر", "أهم الأرقام", "وش لازم أعرف عنه",
    "summary", "executive summary", "key figures",
))


def is_summary_request(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker in normalized for marker in SUMMARY_MARKERS)


INTENT_PATTERNS = {
    "ranking_profit": ("اكثر مشروع ربحان", "اعلي مشروع ربحا", "most profitable project"),
    "ranking_loss": ("اكثر مشروع خسران", "اكبر مشروع خاسر", "most losing project"),
    "ranking_contract": ("اكبر عقد", "اعلى عقد", "اكبر مشروع", "اكبر قيمة", "مين اكبر مشروع", "اغلي مشروع", "biggest contract", "biggest project", "highest contract value"),
    "ranking_best": ("افضل مشروع", "وش افضل مشروع", "best project"),
    "ranking_worst": ("اسوا مشروع", "وش اسوا مشروع", "worst project"),
    "ranking_profit_projects": ("اعلي المشاريع ربحا", "وش اعلي المشاريع ربحا", "top profitable projects"),
    "losing_projects": ("المشاريع الخاسره", "مشاريع خاسره", "losing projects", "loss making projects", "المشاريع خسرانه", "مشاريع خسرانه", "خسرانه", "خسرانة", "اي المشاريع خسران", "وش المشاريع الخسران", "مشروع خسران", "ليش خسران", "ليش عندنا مشاريع خسران", "ليش فيه مشاريع"),
    "profitable_projects": ("المشاريع الرابحه", "مشاريع رابحه", "profitable projects", "المشاريع الرابحة", "مشاريع رابحة", "اي المشاريع رابحه"),
    "expiring": ("قربت تخلص", "قربت تنتهي", "اللي قربت تنتهي", "تنتهي قريبا", "expiring soon", "العقود اللي تنتهي", "عقود تنتهي", "اي عقود تنتهي قريبا", "ينتهي عقودها", "عقودها تنتهي", "وش العقود قاربت تخلص", "قاربت تنتهي", "راح تخلص", "مقاربه تنتهي"),
    "overdue": ("المشاريع المتاخره", "مشاريع متاخره", "overdue projects"),
    "health_projects": ("مشاريع الصحه",),
    "riyadh_projects": ("مشاريع الرياض",),
    "executive_attention": ("وش يحتاج متابعه", "وش اكثر شيء مقلق", "وش اللي يحتاج متابعه", "وين المخاطر", "المشاريع المتعثره", "what needs attention", "اكثر شي مقلق", "ايش يحتاج متابعه", "اشد المشاريع خطورة", "وش اخطر المشاريع", "اي مشاريع تحتاج تدخل", "يحتاج تدخل", "تحتاج تدخل", "يحتاج انتباه", "تحتاج انتباه", "الاولويات", "ايش الاوليات", "مشاريع بحاجة متابعة", "ايش الاولويات", "وش يستدعي انتباه"),
    "portfolio_summary": ("كيف وضع المحفظه", "وش وضع المحفظه", "وش عندنا", "وش اخبار المشاريع", "عطني الزبده", "اعطني الزبده", "ملخص تنفيذي", "اهم شيء اعرفه", "خمس دقايق", "portfolio summary", "ملخص المحفظه", "ملخص المحفظة", "وضع المحفظه", "احوال المحفظه", "نبذة عن المحفظه", "ملخص الاعمال", "overview", "ايش عندنا من مشاريع", "وش عندنا من مشاريع", "اعطني قائمة المشاريع", "عرض كل المشاريع", "وش المشاريع الموجوده"),
}

LIST_FOLLOWUP_MARKERS = tuple(normalize_text(value) for value in (
    "ايش هذي", "وش هذي", "اشرح", "اشرحها", "وش تقصد", "ليش",
    "which one", "what are these", "tell me more", "explain",
    "ايهم اعلى", "ايهم اكبر", "ايهم اقل", "ايهم افضل", "ايهم اسوا",
    "ايهم اكثر", "ايهم اقرب", "which is highest", "which is biggest",
))


def is_previous_list_followup(query: str) -> bool:
    normalized = normalize_text(query)
    return any(marker == normalized or marker in normalized for marker in LIST_FOLLOWUP_MARKERS)


SMALL_TALK_PATTERNS = {
    "salam":     ("السلام عليكم", "السلام عليكم ورحمة الله", "السلام", "سلام"),
    "reply_salam": ("وعليكم السلام", "عليكم السلام"),
    "greeting":  ("اهلا", "أهلاً", "هلا", "مرحبا", "مرحباً", "hello", "hi", "hey", "يهلا", "اهلين", "يسلمو", "مرحبتين", "حياك", "حياكم"),
    "morning":   ("صباح الخير", "صباح النور", "good morning"),
    "evening":   ("مساء الخير", "مساء النور", "good evening"),
    "wellbeing": ("كيف حالك", "كيفك", "شلونك", "كيف الحال", "how are you", "عامل ايه"),
    "thanks":    ("شكرا", "شكراً", "يسلموا", "thank you", "thanks", "ثانكس"),
    "bye":       ("مع السلامه", "في امان الله", "نشوفك على خير", "اشوفك", "باي", "الله يحفظك",
                  "يعطيك العافيه", "مشكور", "مشكور ما قصرت", "خلاص يعطيك العافيه",
                  "وداعا", "الى اللقاء", "bye", "goodbye"),
}


def detect_small_talk(query: str) -> str | None:
    normalized = normalize_text(query)
    # Exact meanings win before containment.  Without this, "السلام" inside
    # "مع السلامة" is incorrectly returned as a greeting.
    for kind, patterns in SMALL_TALK_PATTERNS.items():
        if normalized in {normalize_text(pattern) for pattern in patterns}:
            return kind
    for kind, patterns in SMALL_TALK_PATTERNS.items():
        if any(normalize_text(pattern) in normalized for pattern in sorted(patterns, key=len, reverse=True)):
            return kind
    return None


def detect_portfolio_operation(query: str) -> dict | None:
    """Extract the common portfolio operation independently of full phrases.

    The router and query builder share this contract so count/rank semantics
    cannot drift between the two stages.
    """
    q = normalize_text(query)
    project_subject = any(token in q for token in ("مشروع", "مشاريع", "المشاريع", "project"))
    requested = detect_requested_field(query)
    if requested is None and any(token in q for token in ("اعمال متبقيه", "باقي فيه", "باك لوق")):
        requested = FIELDS["backlog"]

    status = None
    if any(token in q for token in ("جاري", "جاريه", "شغال", "مستمر", "ongoing", "active")):
        status = ["Ongoing"]
    elif any(token in q for token in ("مكتمل", "مكتمله", "منتهي", "منتهيه", "completed", "closed")):
        status = ["Completed", "Closed"]

    if project_subject and any(token == q or f"{token} " in q for token in ("كم", "عدد", "how many", "count")):
        return {"operation": "count", "status": status}

    # "أقل مشروع" is a rank; "هامشه أقل من 5" is a numeric filter.
    threshold_comparison = bool(re.search(r"(?:اقل|اعلي|اكثر|ادني)\s+من\s+\d", q))
    direction = None
    if not threshold_comparison and any(token in q for token in ("اصغر", "اقل", "ادنى", "ادني", "lowest", "smallest")):
        direction = "ASC"
    elif not threshold_comparison and any(token in q for token in ("اكبر", "اعلى", "اعلي", "اكثر", "highest", "largest", "biggest")):
        direction = "DESC"
    if direction:
        if not project_subject and requested is None:
            return None
        metric = requested.canonical if requested else "total_contract_value"
        if any(token in q for token in ("هامش", "نسبه ربح", "نسبة ربح", "margin")):
            metric = "profit_pct"
        elif any(token in q for token in ("ربحيه", "ربحية", "ربح", "profit")):
            metric = "pl"
        return {"operation": "rank", "direction": direction, "metric": metric, "status": status}
    return None


def detect_semantic_intent(query: str) -> str | None:
    normalized = normalize_text(query)
    for intent, patterns in INTENT_PATTERNS.items():
        if any(normalize_text(pattern) in normalized for pattern in patterns):
            return intent
    return None
