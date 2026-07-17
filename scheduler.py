"""
scheduler.py
------------
Background sync scheduler.
Runs data_loader.sync_all() automatically every N hours.

Usage:
  python scheduler.py          ← runs standalone (recommended for production)
  imported by app.py           ← runs in background thread (dev/simple deploy)

Change SYNC_INTERVAL_HOURS in .env to control frequency.
"""

import logging
import threading
import time
import os

logger = logging.getLogger(__name__)

SYNC_INTERVAL_HOURS = int(os.environ.get("SYNC_INTERVAL_HOURS", "6"))


def _run_sync():
    """Import here to avoid circular imports at module load."""
    from modules.data_loader import sync_all
    from modules.database    import get_session, AuditLog

    logger.info("⏰ Scheduled sync starting...")
    try:
        summary = sync_all()
        errors  = summary.get("errors", [])
        status  = "completed" if not errors else f"completed_with_{len(errors)}_errors"

        # Log sync event to audit table
        with get_session() as session:
            session.add(AuditLog(
                user_id       = "scheduler",
                query_text    = "automatic_sync",
                query_type    = "system_sync",
                response_text = str(summary),
                source        = "scheduler",
            ))
            session.commit()

        logger.info("⏰ Sync %s. Excel: %d files, PDF: %d files, Errors: %d",
                    status,
                    len(summary.get("excel", {})),
                    len(summary.get("pdf",   {})),
                    len(errors))
    except Exception as e:
        logger.error("⏰ Sync failed: %s", e)


def start_background_scheduler():
    """
    Start a background daemon thread that syncs every SYNC_INTERVAL_HOURS.
    Called from app.py so it runs alongside Flask.
    Thread is daemon=True so it stops when Flask stops.
    """
    def loop():
        # First sync immediately on startup
        _run_sync()
        while True:
            time.sleep(SYNC_INTERVAL_HOURS * 3600)
            _run_sync()

    thread = threading.Thread(target=loop, name="DataSyncScheduler", daemon=True)
    thread.start()
    logger.info("⏰ Background scheduler started — syncing every %dh", SYNC_INTERVAL_HOURS)
    return thread


# ── Standalone mode ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Running one-time sync...")
    _run_sync()
    logger.info("Done. Run again or set up a cron job:")
    logger.info("  Windows Task Scheduler: python scheduler.py")
    logger.info("  Linux cron: 0 */6 * * * cd /path/to/project && python scheduler.py")