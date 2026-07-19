"""
config.py
---------
Single source of truth for all configuration.
All secrets loaded from environment variables — never hardcoded.
Swap DB_URL or AI_PROVIDER here to migrate without touching other files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (local dev only — production uses real env vars)
load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
UPLOAD_DIR  = BASE_DIR / "uploads"
CHROMA_DIR  = BASE_DIR / "chroma_db"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-in-production")

# ── Local run mode ────────────────────────────────────────────────────────────
# APP_HOST defaults to 127.0.0.1 (localhost-only) for the local demo use
# case -- nothing else on the same network can reach the app at all. Set
# APP_HOST=0.0.0.0 explicitly (e.g. in .env) only if you need to open the
# app to another device on the same LAN; app.py prints which mode is active
# on startup and never changes this value on its own. See
# PRODUCTION_SECURITY_TODO.md before ever using 0.0.0.0 outside a trusted,
# private network.
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))

# Allowed file types for upload (enforced in ingestion layer)
ALLOWED_EXTENSIONS = {"xlsx", "xls", "pdf"}
MAX_UPLOAD_BYTES   = 20 * 1024 * 1024  # 20 MB hard limit

# ── Database ──────────────────────────────────────────────────────────────────
# SQLite for now.  To migrate to PostgreSQL, change this one line:
#   DB_URL = "postgresql+psycopg2://user:pass@host/dbname"
DB_URL = os.environ.get("DB_URL", f"sqlite:///{BASE_DIR}/database.db")

# ── AI Provider ───────────────────────────────────────────────────────────────
# OpenAI for MVP.  To switch to local LLM (Ollama/Llama), only ai_engine.py changes.
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT", "https://elm-openai.openai.azure.com/"
)
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
)
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
AI_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("AI_REQUEST_TIMEOUT_SECONDS", "8"))

# ── Semantic interpreter (opt-in, off by default) ───────────────────────────
# Gates modules.ai_engine's new centralized semantic-interpretation path
# (modules.semantic_interpreter / entity_resolvers / query_compiler). When
# false (the default -- every existing deployment and test run), behavior is
# byte-for-byte identical to before this flag existed: modules.ai_engine.answer
# always uses the original _answer_inner path. When true, _answer_inner_v2 is
# attempted first and falls back to _answer_inner on any error, timeout, or
# low-confidence/unsupported result -- never a hard failure on its own.
SEMANTIC_INTERPRETER_ENABLED = os.environ.get(
    "SEMANTIC_INTERPRETER_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "contracts"

# ── RAG Settings ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks to preserve context
TOP_K_RESULTS = 5     # number of chunks returned per query

# ── Rate Limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT = "30 per minute"

# ── Telegram (Phase 2) ────────────────────────────────────────────────────────
# Uncomment when implementing Phase 2
# TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ── Audit Log ─────────────────────────────────────────────────────────────────
AUDIT_LOG_FILE = BASE_DIR / "audit.log"

# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> list[str]:
    """Return list of critical warnings on startup."""
    warnings = []
    if not AZURE_OPENAI_KEY and not OPENAI_API_KEY:
        warnings.append(
            "AZURE_OPENAI_KEY and fallback OPENAI_API_KEY are not set — AI features will fail"
        )
    if SECRET_KEY == "change-this-in-production":
        warnings.append("SECRET_KEY is using default value — set it in .env")
    return warnings
