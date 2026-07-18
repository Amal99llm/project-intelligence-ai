from modules import query_builder, query_executor


def test_compound_filter_planner_keeps_every_condition():
    spec = query_builder.build_query("Which loss-making projects are close to contract expiry?", "2026-07-15")
    assert {item["column"] for item in spec["filters"]} == {"net_profit", "days_remaining"}

    spec = query_builder.build_query("Which overdue projects have recorded risk?", "2026-07-15")
    assert {item["column"] for item in spec["filters"]} == {"days_remaining", "risk"}

    spec = query_builder.build_query("Which completed projects are worth more than SAR 50 million?", "2026-07-15")
    assert {item["column"] for item in spec["filters"]} == {"status", "total_contract_value"}


def test_progress_gap_filter_is_deterministic():
    spec = query_builder.build_query("Which projects are behind schedule based on elapsed time?", "2026-07-15")
    assert spec["filters"] == [{"column": "progress_gap", "op": ">=", "value": 20}]
    rows = [{"project_code": "A", "progress_gap": 30}, {"project_code": "B", "progress_gap": None}]
    result = query_executor.execute(spec, projects=rows)
    assert [row["project_code"] for row in result["rows"]] == ["A"]


def test_rank_limit_includes_all_projects_tied_at_boundary():
    spec = {"filters": [], "sort": {"column": "net_profit", "direction": "DESC"}, "limit": 1, "aggregation": None}
    rows = [{"project_code": "A", "net_profit": 10}, {"project_code": "B", "net_profit": 10}, {"project_code": "C", "net_profit": 5}]
    result = query_executor.execute(spec, projects=rows)
    assert [row["project_code"] for row in result["rows"]] == ["A", "B"]


def test_arabic_digits_drive_portfolio_filters_and_limits():
    margin = query_builder.build_query("وش المشاريع اللي هامشها أقل من ٥٪؟", "2026-07-15")
    assert {"column": "profit_pct", "op": "<", "value": 5.0} in margin["filters"]
    expiry = query_builder.build_query("وش المشاريع اللي تنتهي خلال ٩٠ يوم؟", "2026-07-15")
    assert expiry["filters"][0]["value2"] == 90
