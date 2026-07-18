"""Application clock for all user-facing date calculations."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

RIYADH_TIMEZONE = ZoneInfo("Asia/Riyadh")


def riyadh_now() -> datetime:
    return datetime.now(RIYADH_TIMEZONE)


def riyadh_today() -> date:
    return riyadh_now().date()
