"""
modules/ingestion.py
--------------------
Step 6: Data Ingestion Layer.
Reads Excel and PDF files, validates them, returns clean raw data.
To add a new source (CSV, API, SAP) — add a new function here only.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pypdf

import config

logger = logging.getLogger(__name__)


# ── File Validation ───────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lstrip(".").lower()
    return ext in config.ALLOWED_EXTENSIONS


def _safe_filename(filename: str) -> str:
    """Strip directory traversal and dangerous chars."""
    name = Path(filename).name
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    return safe or "unnamed_file"


def validate_upload(file_obj) -> tuple[bool, str]:
    """
    Validate a Flask FileStorage object before saving.
    Returns (is_valid, error_message).
    """
    if not file_obj or file_obj.filename == "":
        return False, "No file selected"

    if not _allowed_file(file_obj.filename):
        return False, f"File type not allowed. Allowed: {config.ALLOWED_EXTENSIONS}"

    # Check size: read then seek back
    file_obj.seek(0, 2)
    size = file_obj.tell()
    file_obj.seek(0)
    if size > config.MAX_UPLOAD_BYTES:
        return False, f"File too large (max {config.MAX_UPLOAD_BYTES // 1_048_576} MB)"

    return True, ""


def save_upload(file_obj) -> Path:
    """Save validated file to uploads dir with a hash-prefixed name."""
    safe_name = _safe_filename(file_obj.filename)
    prefix    = hashlib.md5(safe_name.encode()).hexdigest()[:8]
    dest      = config.UPLOAD_DIR / f"{prefix}_{safe_name}"
    file_obj.save(dest)
    logger.info("File saved: %s", dest)
    return dest


# ── Excel Ingestion ───────────────────────────────────────────────────────────

def read_excel(file_path: Path) -> dict[str, pd.DataFrame]:
    """
    Read all sheets from an Excel file.
    Returns dict: { sheet_name: DataFrame }
    Raises ValueError on corrupt or empty files.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        sheets: dict[str, pd.DataFrame] = pd.read_excel(
            file_path,
            sheet_name=None,       # read all sheets
            engine="openpyxl",
            dtype=str,             # read everything as str first — processor cleans types
        )
    except Exception as e:
        raise ValueError(f"Cannot read Excel file: {e}") from e

    if not sheets:
        raise ValueError("Excel file has no sheets")

    # Drop fully-empty sheets
    sheets = {k: v for k, v in sheets.items() if not v.empty}
    if not sheets:
        raise ValueError("All sheets in Excel file are empty")

    logger.info("Excel loaded: %s | sheets: %s", file_path.name, list(sheets.keys()))
    return sheets


# ── PDF Ingestion ─────────────────────────────────────────────────────────────

def read_pdf(file_path: Path) -> str:
    """
    Extract full text from a PDF contract.
    Returns plain text string.
    Raises ValueError on corrupt or empty PDFs.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        reader = pypdf.PdfReader(str(file_path))
        pages  = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise ValueError(f"Cannot read PDF file: {e}") from e

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("PDF appears to be empty or scanned (no extractable text)")

    logger.info("PDF loaded: %s | pages: %d | chars: %d",
                file_path.name, len(pages), len(text))
    return text
