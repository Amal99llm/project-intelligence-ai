import os
import sys
import tempfile
from datetime import date, timedelta

import pytest


def pytest_configure(config):
    """Redirect pytest's own tmp_path/tmpdir base away from the default
    "<TEMP>/pytest-of-<user>" directory. On this machine that directory has
    a corrupted ACL (confirmed via `icacls`/`Get-Acl`, both denied even to
    its owning account) -- a pre-existing OS-level issue unrelated to this
    project, but pytest's tmp_path cleanup logic tries to list it on every
    run and raises PermissionError before any test even starts. Using a
    project-specific, freshly-created directory sidesteps that corrupted
    path entirely without needing admin rights to repair/delete it."""
    if not config.option.basetemp:
        base = os.path.join(tempfile.gettempdir(), "pytest-project-intelligence-ai")
        os.makedirs(base, exist_ok=True)
        config.option.basetemp = base


# Point at a throwaway temp SQLite file BEFORE any project module is
# imported anywhere in the test session, so modules.database's
# `engine = create_engine(config.DB_URL, ...)` never touches real project
# data. Each test gets full isolation via the seeded_db fixture below,
# which drops and recreates all tables per test.
_TMP_DB_PATH = os.path.join(tempfile.gettempdir(), "elm_test_only.db")
os.environ["DB_URL"] = f"sqlite:///{_TMP_DB_PATH}"
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Base, engine, get_session, BacklogProject  # noqa: E402
from modules import session_context  # noqa: E402


TODAY = date(2026, 7, 15)


def _row(**kwargs):
    defaults = dict(
        project_name_en="", project_name_ar="", bu="Digital", segment="Public",
        status="Ongoing", progress_completed=0.5,
        start_date=TODAY - timedelta(days=200), end_date=TODAY + timedelta(days=100),
        amended_end_date=None,
        contract_value=0, amendment_crs=0, total_contract_value=0,
        previous_years_rev=0, revenue_current=0, other_income=0, total_revenue=0,
        previous_years_cost=0, cost_of_revenue=0, other_cost=0, total_cost=0,
        backlog=0, pl=0, planned_profit=0, planned_pm_pct=0, variance=0,
        pm_pct_up_to_2025=0, gp_2026=0, pm_pct_2026=0, risk=0,
        acc_rev=0, pb=0, adv=0, ar=0, contract_assets=0, ap=0, acc_exp=0,
        contract_liabilities=0, deferred_cost=0, open_po=0, ecl_ar=0,
        ecl_acc_rev=0, etc_cost=0, etc_revenue=0,
        customer_id=None, project_manager=None, officer_name=None, note=None,
    )
    defaults.update(kwargs)
    return defaults


# A small, fixed, isolated portfolio -- NOT real project data.
TEST_PROJECTS = [
    _row(
        project_code="PRJ-001", project_name_ar="الباحث الاجتماعي الثاني",
        project_name_en="Social Researcher II", project_manager="Manager A",
        total_revenue=1_000_000, total_cost=800_000, pl=190_000,  # ~1 SAR-scale rounding vs computed 200,000 in spirit; here a deliberately larger gap to exercise the variance flag
        total_contract_value=1_200_000, backlog=200_000,
        end_date=TODAY + timedelta(days=60),
    ),
    _row(
        project_code="PRJ-002", project_name_ar="مشروع النور للطاقة",
        project_name_en="Al Noor Energy Project", project_manager="Manager B",
        total_revenue=500_000, total_cost=650_000, pl=-150_000,
        total_contract_value=700_000, backlog=50_000,
        end_date=TODAY + timedelta(days=10),  # expiring soon
    ),
    _row(
        project_code="PRJ-003", project_name_ar="برنامج التطوير الأول",
        project_name_en="Development Program I", project_manager="Manager C",
        total_revenue=2_000_000, total_cost=1_500_000, pl=500_000,
        total_contract_value=2_500_000, backlog=300_000,
        end_date=TODAY + timedelta(days=400),
    ),
    _row(
        project_code="PRJ-004", project_name_ar="برنامج التطوير الثاني",
        project_name_en="Development Program II", project_manager="Manager D",
        total_revenue=1_800_000, total_cost=1_600_000, pl=200_000,
        total_contract_value=2_000_000, backlog=100_000,
        end_date=TODAY + timedelta(days=500),
    ),
    _row(
        project_code="PRJ-005", project_name_ar="مشروع الرصد البيئي",
        project_name_en="Environmental Monitoring Project", project_manager="Manager E",
        total_revenue=300_000, total_cost=300_000, pl=0,
        total_contract_value=300_000, backlog=0,
        status="Completed", end_date=TODAY - timedelta(days=365),  # long-expired, completed
    ),
]


@pytest.fixture()
def seeded_db():
    """Fresh, isolated schema + fixed test portfolio for one test."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with get_session() as session:
        for row in TEST_PROJECTS:
            session.add(BacklogProject(**row))
        session.commit()
    session_context._reset_all_for_tests()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture()
def today():
    return TODAY
