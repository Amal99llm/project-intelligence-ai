"""
app.py — Elm Project Intelligence Platform
Flask application with all routes.

SCOPE UPDATE (see git history / PRODUCTION_SECURITY_TODO.md for the prior
state): this build is now deployed to a shared Railway URL so a specific
manager can test it remotely, for a limited window -- it is no longer the
local-only demo described in the original note below. Authentication has
been added accordingly (see `require_auth` below). Everything else that was
already hardened in the local-only pass is unchanged: input validation,
exception handling (no raw tracebacks to the client), a configurable bind
address, and a signed, server-side session id (never the client-supplied
X-User-ID header) used to key modules.session_context's per-session
conversation state.

AUTH MODEL: single shared username/password (HTTP Basic Auth) read from the
APP_USER / APP_PASS environment variables (set in Railway → Variables,
never committed to source). If either is missing, every protected route
denies access by default rather than silently opening up. This is
intentionally simple -- it is meant for a short, single-reviewer trial, not
multi-user access control. /health stays open (no data, needed for
Railway's own health checks).
"""

import sys
import os
import logging
import uuid
from functools import wraps

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")

import bleach
from flask import Flask, render_template, request, jsonify, session as flask_session, Response
from modules.time_utils import riyadh_today

import config
from modules.ingestion   import validate_upload, save_upload, read_excel, read_pdf
from modules.processor   import process_excel_file, process_pdf_contract
from modules.ai_engine   import answer
from modules.data_loader import sync_all, get_last_sync_time
from modules.llm_health import check_openai_client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.AUDIT_LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

GENERIC_ERROR_MESSAGE = "حدث خطأ أثناء معالجة الطلب. حاول إعادة صياغة السؤال أو المحاولة لاحقًا."

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

for warning in config.validate_config():
    logger.warning("CONFIG WARNING: %s", warning)

check_openai_client(logger)

from scheduler import start_background_scheduler
if os.environ.get("DISABLE_BACKGROUND_SCHEDULER", "").lower() not in {"1", "true", "yes"}:
    start_background_scheduler()


def _get_session_id() -> str:
    """Server-generated, signed-cookie session id used only as a key into
    modules.session_context's small structured per-session state (last
    project/KPI discussed, pending confirmation). Never trust the
    client-supplied X-User-ID header for this -- that header is
    unauthenticated and trivially spoofable, so it must never be used to
    key anything that carries state between requests."""
    if "sid" not in flask_session:
        flask_session["sid"] = uuid.uuid4().hex
        flask_session.permanent = True
    return flask_session["sid"]


# ── Auth (added for the shared Railway trial — see module docstring) ─────────

def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "unknown")


def _credentials_ok(username: str, password: str) -> bool:
    expected_user = os.environ.get("APP_USER")
    expected_pass = os.environ.get("APP_PASS")
    if not expected_user or not expected_pass:
        # Fail closed: misconfiguration must never silently mean "open to everyone".
        logger.warning("APP_USER/APP_PASS not set — denying all access until configured.")
        return False
    return username == expected_user and password == expected_pass


def _unauthorized() -> Response:
    return Response(
        "🔒 تسجيل الدخول مطلوب للوصول لهذا التطبيق.",
        401,
        {"WWW-Authenticate": 'Basic realm="Restricted Access"'},
    )


def require_auth(view_func):
    """HTTP Basic Auth gate for the shared trial deployment. Every attempt
    (missing, wrong, or successful) is logged through the app's existing
    logger/audit file, alongside the client IP and requested path."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        ip = _client_ip()

        if not auth:
            logger.warning("AUTH: no credentials | ip=%s | path=%s", ip, request.path)
            return _unauthorized()

        if not _credentials_ok(auth.username, auth.password):
            logger.warning("AUTH: failed login | ip=%s | username_tried=%s | path=%s",
                            ip, auth.username, request.path)
            return _unauthorized()

        logger.info("AUTH: ok | ip=%s | user=%s | path=%s", ip, auth.username, request.path)
        return view_func(*args, **kwargs)

    return wrapped


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
@require_auth
def index():
    try:
        last_sync = get_last_sync_time()
    except Exception:
        logger.exception("Failed to load last sync time for index page")
        last_sync = None
    return render_template("index.html", last_sync=last_sync)


@app.route("/ask", methods=["POST"])
@require_auth
def ask():
    try:
        raw_query = request.json.get("query", "") if request.is_json else request.form.get("query", "")
    except Exception:
        return jsonify({"error": "طلب غير صالح."}), 400

    query = bleach.clean(raw_query or "", tags=[], strip=True).strip()
    if not query:
        return jsonify({"error": "الرجاء كتابة سؤال."}), 400

    session_id = _get_session_id()
    try:
        if config.CHAT_ENGINE_V2_ENABLED:
            from modules.chat_service import answer as answer_v2
            result = answer_v2(query=query, session_id=session_id)
        else:
            result = answer(
                query      = query,
                user_id    = request.headers.get("X-User-ID", "anonymous"),
                source     = "flask_ui",
                ip_address = request.remote_addr,
                session_id = session_id,
            )
    except Exception:
        # modules.ai_engine.answer() already catches its own internal
        # exceptions and returns the Arabic fallback message -- this is a
        # last-resort net in case something above that layer still raises.
        logger.exception("Unhandled error in /ask")
        return jsonify({"answer": GENERIC_ERROR_MESSAGE, "query_type": "error"}), 200
    return jsonify(result)


@app.route("/upload", methods=["POST"])
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم إرفاق أي ملف."}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "لم يتم إرفاق أي ملف."}), 400

    try:
        valid, err = validate_upload(file)
    except Exception:
        logger.exception("Upload validation failed unexpectedly")
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500
    if not valid:
        return jsonify({"error": err or "الملف غير صالح."}), 400

    try:
        file_path = save_upload(file)
        ext       = file_path.suffix.lower()
        if ext in {".xlsx", ".xls"}:
            sheets  = read_excel(file_path)
            results = process_excel_file(sheets)
            return jsonify({"message": "تمت معالجة ملف Excel بنجاح", "details": results})
        elif ext == ".pdf":
            project_code = bleach.clean(request.form.get("project_code", "UNKNOWN"), tags=[], strip=True)
            contract_ref = bleach.clean(request.form.get("contract_ref", file_path.stem), tags=[], strip=True)
            text_content = read_pdf(file_path)
            chunks       = process_pdf_contract(text_content, file_path, project_code, contract_ref)
            return jsonify({"message": f"تمت معالجة العقد: تم فهرسة {chunks} مقطعًا"})
        return jsonify({"error": "نوع ملف غير مدعوم."}), 400
    except Exception as e:
        logger.error("Upload failed: %s", e, exc_info=True)
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500


@app.route("/sync", methods=["POST"])
@require_auth
def manual_sync():
    try:
        summary = sync_all()
        from modules.ai_engine import invalidate_cache
        invalidate_cache()
        return jsonify({
            "message":    "اكتملت المزامنة",
            "excel_files": len(summary.get("excel", {})),
            "pdf_files":   len(summary.get("pdf", {})),
            "errors":      summary.get("errors", []),
            "finished_at": summary.get("finished_at"),
        })
    except Exception as e:
        logger.error("Manual sync failed: %s", e, exc_info=True)
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500


@app.route("/sync/status")
@require_auth
def sync_status():
    try:
        return jsonify({"last_sync": get_last_sync_time()})
    except Exception:
        logger.exception("Failed to read sync status")
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500


@app.route("/api/me")
@require_auth
def api_me():
    """Return current user info from session or SSO headers."""
    user = {
        "username":     flask_session.get("username", ""),
        "display_name": flask_session.get("display_name", ""),
        "name":         flask_session.get("name", ""),
        "role":         flask_session.get("role", ""),
        "title":        flask_session.get("title", ""),
    }
    # Fallback: SSO / reverse-proxy headers
    if not any(user.values()):
        user["display_name"] = (
            request.headers.get("X-Display-Name") or
            request.headers.get("X-User-Name") or
            request.headers.get("X-Remote-User") or
            ""
        )
    return jsonify(user)


@app.route("/api/projects")
@require_auth
def api_projects():
    """Return projects and the canonical database-backed dashboard KPIs.

    Row shaping lives in modules.project_repository and KPI math lives in
    modules.kpi_calculator's registry — this route only fetches and
    serializes, so the dashboard, the chatbot and this API can never
    disagree about what a KPI means.
    """
    from modules.kpi_calculator import calculate_executive_kpis
    from modules.project_repository import fetch_enriched_projects

    try:
        today = riyadh_today()
        projects = fetch_enriched_projects(today=today)
        metrics = calculate_executive_kpis(projects, today=today)
        return jsonify({"projects": projects, "count": metrics["total_projects"], "metrics": metrics})
    except Exception:
        logger.exception("Failed to load /api/projects")
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500


@app.route("/report/<report_type>")
@require_auth
def get_report(report_type: str):
    from modules.ai_engine import generate_report
    allowed = {"executive", "risk", "collection", "backlog"}
    if report_type not in allowed:
        return jsonify({"error": "نوع تقرير غير معروف."}), 400
    try:
        text = generate_report(report_type)
        return jsonify({"answer": text, "query_type": "report_query"})
    except Exception:
        logger.exception("Failed to generate report: %s", report_type)
        return jsonify({"error": GENERIC_ERROR_MESSAGE}), 500


@app.route("/health")
def health():
    # Intentionally NOT behind require_auth: Railway (and any load balancer)
    # needs this reachable to confirm the service is alive, and it exposes
    # no project data — just a static status/version string.
    return jsonify({"status": "ok", "version": "2.0.0"})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = config.APP_HOST
    port = config.APP_PORT

    if not os.environ.get("APP_USER") or not os.environ.get("APP_PASS"):
        print("[WARNING] APP_USER / APP_PASS are not set — every protected route will "
              "return 401 until both are configured (see Railway → Variables).")

    if host == "127.0.0.1":
        print(f"[Local-only mode] Listening on http://127.0.0.1:{port}/ -- "
              f"not reachable from any other device on the network.")
    elif host == "0.0.0.0":
        print(f"[Shared trial mode] Listening on 0.0.0.0:{port} -- protected by HTTP Basic "
              f"Auth (APP_USER/APP_PASS). Anyone with those credentials and the public URL "
              f"can reach every route, including /api/projects (full portfolio data). "
              f"Only share the credentials with the intended reviewer, and rotate/remove "
              f"them once the trial window ends.")
    else:
        print(f"[Custom bind] Listening on {host}:{port}.")

    app.run(host=host, port=port, debug=debug, use_reloader=False)
