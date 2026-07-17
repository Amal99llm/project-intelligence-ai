"""
modules/database.py — v2
Database schema redesigned around Backlog as primary source.
All Backlog columns stored — ready for future sheets integration.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Date, DateTime, Text, JSON
)
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime
import config

engine = create_engine(config.DB_URL, echo=False)


class Base(DeclarativeBase):
    pass


class BacklogProject(Base):
    __tablename__ = "backlog_projects"

    id                   = Column(Integer, primary_key=True)
    project_code         = Column(String(100), unique=True, nullable=False, index=True)
    project_name_en      = Column(String(500))
    project_name_ar      = Column(String(500))
    project_type         = Column(String(50))
    category             = Column(String(100))
    program              = Column(String(100))
    wbs_pc               = Column(String(100))
    wbs                  = Column(String(100))
    pc                   = Column(String(100))
    cc                   = Column(String(100))
    bu                   = Column(String(100))
    dept                 = Column(String(100))
    segment              = Column(String(100))
    officer_name         = Column(String(200))
    customer_id          = Column(String(200))
    support_document     = Column(String(200))
    project_manager      = Column(String(200))
    start_date           = Column(Date)
    end_date             = Column(Date)
    amended_end_date     = Column(Date)
    status               = Column(String(50))
    progress_completed   = Column(Float)
    contract_value       = Column(Float, default=0.0)
    amendment_crs        = Column(Float, default=0.0)
    total_contract_value = Column(Float, default=0.0)
    previous_years_rev   = Column(Float, default=0.0)
    revenue_current      = Column(Float, default=0.0)
    other_income         = Column(Float, default=0.0)
    total_revenue        = Column(Float, default=0.0)
    backlog              = Column(Float, default=0.0)
    previous_years_cost  = Column(Float, default=0.0)
    cost_of_revenue      = Column(Float, default=0.0)
    other_cost           = Column(Float, default=0.0)
    total_cost           = Column(Float, default=0.0)
    pm_up_to_2025        = Column(Float, default=0.0)
    pm_pct_up_to_2025    = Column(Float, default=0.0)
    gp_2026              = Column(Float, default=0.0)
    pm_2026              = Column(Float, default=0.0)
    pm_pct_2026          = Column(Float, default=0.0)
    pl                   = Column(Float, default=0.0)
    net_profit           = Column(Float, default=0.0)
    profit_pct           = Column(Float, default=0.0)
    po                   = Column(Float, default=0.0)
    hr                   = Column(Float, default=0.0)
    other_external       = Column(Float, default=0.0)
    other_internal       = Column(Float, default=0.0)
    risk                 = Column(Float, default=0.0)
    contingency          = Column(Float, default=0.0)
    total_planned_cost   = Column(Float, default=0.0)
    planned_profit       = Column(Float, default=0.0)
    planned_pm_pct       = Column(Float, default=0.0)
    variance             = Column(Float, default=0.0)
    etc_cost             = Column(Float, default=0.0)
    etc_revenue          = Column(Float, default=0.0)
    net_etc              = Column(Float, default=0.0)
    etc_pct              = Column(Float, default=0.0)
    acc_rev              = Column(Float, default=0.0)
    pb                   = Column(Float, default=0.0)
    adv                  = Column(Float, default=0.0)
    ar                   = Column(Float, default=0.0)
    contract_assets      = Column(Float, default=0.0)
    ap                   = Column(Float, default=0.0)
    acc_exp              = Column(Float, default=0.0)
    contract_liabilities = Column(Float, default=0.0)
    deferred_cost        = Column(Float, default=0.0)
    open_po              = Column(Float, default=0.0)
    ecl_ar               = Column(Float, default=0.0)
    ecl_acc_rev          = Column(Float, default=0.0)
    note                 = Column(Text)
    raw_data             = Column(JSON)
    synced_at            = Column(DateTime, default=datetime.utcnow)
    created_at           = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id            = Column(Integer, primary_key=True)
    timestamp     = Column(DateTime, default=datetime.utcnow)
    user_id       = Column(String(100), default="anonymous")
    query_text    = Column(Text, nullable=False)
    query_type    = Column(String(50))
    response_text = Column(Text)
    source        = Column(String(50))
    ip_address    = Column(String(50))


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def log_query(query_text: str, query_type: str, response_text: str,
              user_id: str = "anonymous", source: str = "flask_ui",
              ip_address: str = ""):
    with get_session() as session:
        session.add(AuditLog(
            user_id=user_id, query_text=query_text,
            query_type=query_type, response_text=response_text[:2000],
            source=source, ip_address=ip_address,
        ))
        session.commit()