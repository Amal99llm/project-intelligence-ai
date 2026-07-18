"""
modules/processor.py — v5
Backlog-only processor. Reads ALL columns, stores everything.
Other sheets: preserved in code, skipped in current phase.
"""

import logging
from pathlib import Path
from datetime import date, datetime

import pandas as pd
try:
    import chromadb
except ImportError:  # Structured BI and Flask imports must work without optional RAG extras.
    chromadb = None
from openai import AzureOpenAI

import config
from modules.database import get_session, BacklogProject

logger = logging.getLogger(__name__)

_chroma_client = None
_chroma_collection = None
_openai_client = None


def _get_chroma():
    global _chroma_client, _chroma_collection
    if chromadb is None:
        raise RuntimeError("ChromaDB is required to process contract documents")
    if _chroma_collection is None:
        _chroma_client     = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return _chroma_collection


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_KEY,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _openai_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s(val) -> str:
    if val is None: return ""
    s = str(val).strip()
    return "" if s in ("nan", "None", "NaT", "") else s


def _f(val) -> float:
    try:
        if pd.isna(val): return 0.0
    except: pass
    try:
        return float(str(val).replace(",", "").replace(" ", "").replace("%", "").strip())
    except: return 0.0


def _d(val):
    if val is None: return None
    try:
        if pd.isna(val): return None
    except: pass
    try:
        r = pd.to_datetime(val)
        return None if pd.isna(r) else r.date()
    except: return None


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _json_safe(val):
    """Make value JSON serializable."""
    if val is None: return None
    try:
        if pd.isna(val): return None
    except: pass
    if isinstance(val, (date, datetime)): return str(val)
    if isinstance(val, float):
        import math
        if math.isnan(val) or math.isinf(val): return None
    return val


# ══════════════════════════════════════════════════════════════════════════════
# BACKLOG PROCESSOR — Primary source
# ══════════════════════════════════════════════════════════════════════════════

def process_backlog(df: pd.DataFrame) -> int:
    df = _strip(df)
    now = datetime.utcnow()
    count = 0

    with get_session() as session:
        for _, row in df.iterrows():

            # ── Project code ──────────────────────────────────────────────
            code = (
                _s(row.get("Project Definition")) or
                _s(row.get("WBS \"PC\""))          or
                _s(row.get("PC"))
            )
            if not code:
                continue

            # ── Names ──────────────────────────────────────────────────────
            name_en = (
                _s(row.get("Project Name\n(English)")) or
                _s(row.get("Project Name\r\n(English)")) or
                _s(row.get("Project Name (English)")) or
                _s(row.get("Project Name"))
            )
            name_ar = (
                _s(row.get("Project Name\n(Arabic)")) or
                _s(row.get("Project Name\r\n(Arabic)")) or
                _s(row.get("Project Name (Arabic)"))
            )

            # ── Dates ──────────────────────────────────────────────────────
            end_date = (
                _d(row.get("Amended End Date\nCR")) or
                _d(row.get("Amended End Date\r\nCR")) or
                _d(row.get("Amended End Date CR")) or
                _d(row.get("End Date"))
            )

            # ── Store ALL raw columns as JSON ──────────────────────────────
            raw = {str(k): _json_safe(v) for k, v in row.items()}

            # ── Upsert ────────────────────────────────────────────────────
            existing = session.query(BacklogProject).filter_by(project_code=code).first()
            obj = existing or BacklogProject(project_code=code)

            # Identity
            obj.project_name_en      = name_en
            obj.project_name_ar      = name_ar
            obj.project_type         = _s(row.get("Type"))
            obj.category             = _s(row.get("Category"))
            obj.program              = _s(row.get("Program"))
            obj.wbs_pc               = _s(row.get("WBS \"PC\""))
            obj.wbs                  = _s(row.get("WBS"))
            obj.pc                   = _s(row.get("PC"))
            obj.cc                   = _s(row.get("CC"))
            obj.bu                   = _s(row.get("BU"))
            obj.dept                 = _s(row.get("Dept."))
            obj.segment              = _s(row.get("Segment"))
            obj.officer_name         = _s(row.get("Officer Name"))
            obj.customer_id          = _s(row.get("Costumer ID"))
            obj.support_document     = _s(row.get("Support Document"))
            obj.project_manager      = _s(row.get("Project Manager Name"))

            # Dates
            obj.start_date           = _d(row.get("Start Date"))
            obj.end_date             = end_date
            obj.amended_end_date     = (
                _d(row.get("Amended End Date\nCR")) or
                _d(row.get("Amended End Date CR"))
            )
            obj.status               = _s(row.get("Status"))
            obj.progress_completed   = _f(row.get("Progress Completed"))

            # Contract value
            obj.contract_value       = _f(row.get("Contract Value"))
            obj.amendment_crs        = _f(row.get("Amendment (CRs)"))
            obj.total_contract_value = _f(row.get("Total Contract Value"))

            # Revenue
            obj.previous_years_rev   = _f(row.get("Previous Years Revenue"))
            obj.revenue_current      = _f(row.get("Revenue"))
            obj.other_income         = _f(row.get("Other Income"))
            obj.total_revenue        = _f(row.get("Total Revenue"))
            obj.backlog              = _f(row.get("Backlog"))

            # Cost
            obj.previous_years_cost  = _f(row.get("Previous Years Cost"))
            obj.cost_of_revenue      = _f(row.get("Cost of Revenue "))  or _f(row.get("Cost of Revenue"))
            obj.other_cost           = _f(row.get("Other Cost"))
            obj.total_cost           = _f(row.get("Total Cost"))

            # Profitability
            obj.pm_up_to_2025        = _f(row.get("PM up to 2025"))
            obj.pm_pct_up_to_2025    = _f(row.get("PM% up to 2025"))
            obj.gp_2026              = _f(row.get("GP 2026"))
            obj.pm_2026              = _f(row.get("PM  2026"))
            obj.pm_pct_2026          = _f(row.get("PM%  2026"))
            obj.pl                   = _f(row.get("P&L ")) or _f(row.get("P&L"))
            obj.net_profit           = _f(row.get("Net"))
            obj.profit_pct           = _f(row.get("%"))

            # Cost breakdown
            obj.po                   = _f(row.get("PO"))
            obj.hr                   = _f(row.get("HR"))
            obj.other_external       = _f(row.get("Other (External)"))
            obj.other_internal       = _f(row.get("Other (Internal)"))
            obj.risk                 = _f(row.get("Risk"))
            obj.contingency          = _f(row.get("Contingency"))
            obj.total_planned_cost   = _f(row.get("Total Planned Cost"))
            obj.planned_profit       = _f(row.get("Planned Profit"))
            obj.planned_pm_pct       = _f(row.get("Planned PM%"))
            obj.variance             = _f(row.get("Var."))
            obj.etc_cost             = _f(row.get("ETC (Cost) ")) or _f(row.get("ETC (Cost)"))
            obj.etc_revenue          = _f(row.get("ETC (Revenue)"))
            obj.net_etc              = _f(row.get("Net"))
            obj.etc_pct              = _f(row.get("%"))

            # Financial position (for future sheet integration)
            obj.acc_rev              = _f(row.get("Acc Rev"))
            obj.pb                   = _f(row.get("PB"))
            obj.adv                  = _f(row.get("Adv"))
            obj.ar                   = _f(row.get("AR"))
            obj.contract_assets      = _f(row.get("Contract Assets"))
            obj.ap                   = _f(row.get("AP"))
            obj.acc_exp              = _f(row.get("Acc Exp"))
            obj.contract_liabilities = _f(row.get("Contract Liabilities"))
            obj.deferred_cost        = _f(row.get("Deferred Cost"))
            obj.open_po              = _f(row.get("Open PO"))
            obj.ecl_ar               = _f(row.get("ECL (AR)"))
            obj.ecl_acc_rev          = _f(row.get("ECL (Acc Rev)"))
            obj.note                 = _s(row.get("Note"))

            # Raw JSON — all columns preserved
            obj.raw_data             = raw
            obj.synced_at            = now

            if not existing:
                session.add(obj)
            count += 1

        session.commit()

    logger.info("Backlog: upserted %d projects", count)
    return count


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

# Future sheets — kept as stubs, activated in next phases
FUTURE_SHEETS = {
    "accrued revenue":       "Phase 2 — Accrued Revenue integration",
    "progress billing":      "Phase 2 — Progress Billing integration",
    "account receivable":    "Phase 2 — AR integration",
    "advance from customer": "Phase 2 — Advance integration",
    "deferred cost":         "Phase 2 — Deferred Cost integration",
}

SKIP_SHEETS = {
    "cover page", "pending subjects", "definitions", "summery",
    "accrued data", "progress billing data",
    "account receivable data", "advance from customer data",
}


def process_excel_file(sheets: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for sheet_name, df in sheets.items():
        name_lower = sheet_name.strip().lower()

        if name_lower in SKIP_SHEETS:
            results[sheet_name] = "skipped (info sheet)"
            continue
        if df.empty:
            results[sheet_name] = "skipped (empty)"
            continue
        if "backlog" in name_lower:
            try:
                count = process_backlog(df)
                results[sheet_name] = f"✅ {count} projects loaded"
            except Exception as e:
                results[sheet_name] = f"❌ {e}"
                logger.error("Backlog failed: %s", e)
            continue

        # Future sheets — preserved, not processed yet
        for keyword, phase in FUTURE_SHEETS.items():
            if keyword in name_lower:
                results[sheet_name] = f"preserved — {phase}"
                break
        else:
            results[sheet_name] = "skipped (unrecognised)"

    return results


# ── PDF ───────────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunks.append(text[start:end])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def _embed(texts: list[str]) -> list[list[float]]:
    client   = _get_openai()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def process_pdf_contract(text: str, file_path: Path,
                          project_code: str, contract_ref: str) -> int:
    chunks     = _chunk_text(text)
    collection = _get_chroma()
    embeddings = _embed(chunks)
    ids        = [f"{contract_ref}_chunk_{i}" for i in range(len(chunks))]
    metadatas  = [{"project_code": project_code, "contract_ref": contract_ref,
                   "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings,
                      documents=chunks, metadatas=metadatas)
    logger.info("Contract '%s': %d chunks indexed", contract_ref, len(chunks))
    return len(chunks)
