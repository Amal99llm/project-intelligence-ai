"""
modules/data_loader.py
----------------------
Automatic Data Source Layer.

Scans the data/ folder and loads everything into SQLite + ChromaDB.
No manual upload needed — just place files in data/ and run sync.

To change data source in the future:
  - Replace _scan_excel_files() with a database connector
  - Replace _scan_pdf_files() with an API connector
  - The rest of the system stays unchanged.
"""

import logging
from pathlib import Path
from datetime import datetime

import config
from modules.ingestion import read_excel, read_pdf
from modules.processor import process_excel_file, process_pdf_contract
from modules.database  import get_session, AuditLog

logger = logging.getLogger(__name__)

# ── Data folder ───────────────────────────────────────────────────────────────

DATA_DIR = config.BASE_DIR / "data"


# ── Source Registry (swap connectors here for future data sources) ────────────

class DataSourceRegistry:
    """
    Registry of available data source connectors.
    Each connector must implement: scan() -> list[Path] or similar.

    Current connectors:
      - ExcelFolderConnector  : reads .xlsx/.xls from data/
      - PDFFolderConnector    : reads .pdf from data/contracts/

    Future connectors (add here without touching anything else):
      - PostgreSQLConnector   : reads from live DB tables
      - SAPConnector          : reads from SAP RFC
      - APIConnector          : reads from REST endpoint
    """

    def __init__(self):
        self._connectors = {}

    def register(self, name: str, connector):
        self._connectors[name] = connector
        logger.info("DataSource registered: %s", name)

    def get(self, name: str):
        return self._connectors.get(name)

    def all(self):
        return self._connectors.items()


# ── Connectors ────────────────────────────────────────────────────────────────

class ExcelFolderConnector:
    """Reads all Excel files from data/ folder."""

    def __init__(self, folder: Path = DATA_DIR):
        self.folder = folder

    def scan(self) -> list[Path]:
        files = list(self.folder.glob("*.xlsx")) + list(self.folder.glob("*.xls"))
        logger.info("ExcelFolderConnector: found %d files", len(files))
        return files

    def load(self, path: Path) -> dict:
        return read_excel(path)


class PDFFolderConnector:
    """Reads all PDF files from data/contracts/ folder."""

    def __init__(self, folder: Path = DATA_DIR / "contracts"):
        self.folder = folder

    def scan(self) -> list[Path]:
        files = list(self.folder.glob("*.pdf"))
        logger.info("PDFFolderConnector: found %d files", len(files))
        return files

    def load(self, path: Path) -> str:
        return read_pdf(path)


# ── Future connector stubs (uncomment and implement when ready) ───────────────

# class PostgreSQLConnector:
#     """Replace ExcelFolderConnector when company DB is ready."""
#     def __init__(self, connection_string: str):
#         self.conn_str = connection_string
#
#     def scan(self):
#         return ["projects_table", "revenue_table", "kpi_table"]
#
#     def load(self, table_name: str) -> pd.DataFrame:
#         import pandas as pd
#         from sqlalchemy import create_engine
#         engine = create_engine(self.conn_str)
#         return pd.read_sql_table(table_name, engine)


# class APIConnector:
#     """Replace ExcelFolderConnector when ERP API is ready."""
#     def __init__(self, base_url: str, api_key: str):
#         self.base_url = base_url
#         self.api_key  = api_key
#
#     def scan(self):
#         return ["/api/projects", "/api/revenue", "/api/kpis"]
#
#     def load(self, endpoint: str) -> dict:
#         import requests
#         resp = requests.get(
#             f"{self.base_url}{endpoint}",
#             headers={"Authorization": f"Bearer {self.api_key}"}
#         )
#         return resp.json()


# ── Sync Engine ───────────────────────────────────────────────────────────────

# Build the registry with current connectors
registry = DataSourceRegistry()
registry.register("excel_folder", ExcelFolderConnector())
registry.register("pdf_folder",   PDFFolderConnector())


def sync_all() -> dict:
    """
    Main sync function — called on startup and by scheduler.
    Loads all data sources into SQLite + ChromaDB.
    Returns summary dict.
    """
    summary = {
        "started_at": datetime.now().isoformat(),
        "excel":      {},
        "pdf":        {},
        "errors":     [],
    }

    # ── Excel files ───────────────────────────────────────────────────────────
    excel_connector = registry.get("excel_folder")
    for path in excel_connector.scan():
        try:
            sheets  = excel_connector.load(path)
            results = process_excel_file(sheets)
            summary["excel"][path.name] = results
            logger.info("Synced Excel: %s", path.name)
        except Exception as e:
            msg = f"Excel {path.name}: {e}"
            summary["errors"].append(msg)
            logger.error(msg)

    # ── PDF contracts ─────────────────────────────────────────────────────────
    pdf_connector = registry.get("pdf_folder")
    for path in pdf_connector.scan():
        try:
            text         = pdf_connector.load(path)
            # Derive project_code and contract_ref from filename
            # Expected format: PRJ001_CNT2024.pdf  or  contract_name.pdf
            parts        = path.stem.replace("-", "_").split("_")
            project_code = parts[0] if len(parts) > 1 else "UNKNOWN"
            contract_ref = path.stem
            chunks       = process_pdf_contract(text, path, project_code, contract_ref)
            summary["pdf"][path.name] = f"{chunks} chunks indexed"
            logger.info("Synced PDF: %s → %d chunks", path.name, chunks)
        except Exception as e:
            msg = f"PDF {path.name}: {e}"
            summary["errors"].append(msg)
            logger.error(msg)

    summary["finished_at"] = datetime.now().isoformat()
    # Centralized here so scheduled and manual syncs invalidate identically.
    try:
        from modules.ai_engine import invalidate_cache
        invalidate_cache()
    except Exception as exc:
        summary["errors"].append(f"Cache invalidation: {exc}")
        logger.error("Post-sync cache invalidation failed: %s", exc)
    logger.info("Sync complete: %s", summary)
    return summary


def get_last_sync_time() -> str | None:
    """Return timestamp of last successful sync from audit log."""
    with get_session() as session:
        entry = (
            session.query(AuditLog)
            .filter_by(query_type="system_sync")
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        return entry.timestamp.isoformat() if entry else None
