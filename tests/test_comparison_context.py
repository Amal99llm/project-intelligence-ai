from modules.ai_engine import _handle_comparison
from modules.comparison_engine import format_comparison_winner


PROJECTS = [
    {"project_code": "A", "project_name_ar": "الأول", "project_name_en": "First", "pl": 10, "net_profit": 10,
     "profit_pct": 5, "total_revenue": 100, "total_cost": 90, "progress_completed": 50, "effective_end_date": "2027-01-01"},
    {"project_code": "B", "project_name_ar": "الثاني", "project_name_en": "Second", "pl": 20, "net_profit": 20,
     "profit_pct": 8, "total_revenue": 120, "total_cost": 100, "progress_completed": 60, "effective_end_date": "2026-12-01"},
]


def test_comparison_followup_reuses_saved_pair(today):
    text, kind, _, updates = _handle_comparison(
        "which one has the higher margin?", today,
        {"comparison_project_ids": ["A", "B"]}, PROJECTS, {},
    )
    assert kind == "project_comparison"
    assert "Second" in text
    assert updates["comparison_project_ids"] == ["A", "B"]


def test_failed_comparison_never_degrades_to_single_project_summary(today):
    text, kind, _, _ = _handle_comparison("compare unknown with First", today, {}, PROJECTS, {})
    assert kind == "project_comparison"
    assert "طرفي المقارنة" in text or "أي واحد تقصد" in text


def test_comparison_tie_is_reported_without_arbitrary_winner():
    tied = [dict(PROJECTS[0], profit_pct=8), PROJECTS[1]]
    answer = format_comparison_winner(tied, "which has the higher margin?", "profit_pct")
    assert "Both projects are tied" in answer
