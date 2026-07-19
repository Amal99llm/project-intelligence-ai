"""
security.py
------------
وحدة جاهزة للحماية والتسجيل (Logging) — استخدمها كما هي بدون تعديل.

توفر:
1. requires_auth: decorator يحمي أي route بـ Basic Auth (اسم مستخدم/كلمة مرور).
2. init_security(app): يفعّل تسجيل كل طلب يدخل على السيرفر (ناجح أو فاشل أو مرفوض)
   في ملف access.log وأيضًا على الشاشة (يظهر في Railway Logs مباشرة).

طريقة الاستخدام في app.py:

    from security import requires_auth, init_security

    app = Flask(__name__)
    init_security(app)          # سطر واحد يفعّل التسجيل لكل الطلبات

    @app.route("/")
    @requires_auth              # يحمي هذا الـ route بكلمة مرور
    def index():
        ...
"""

import os
import logging
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from flask import request, Response, g


# =========================
# 1) إعداد التسجيل (Logging)
# =========================

logger = logging.getLogger("access")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # يطبع في الشاشة (Railway يلتقط هذا تلقائيًا في تبويب Logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(console_handler)

    # يحفظ أيضًا في ملف محلي access.log (يدور تلقائيًا بعد 1MB، يحتفظ بآخر 3 نسخ)
    try:
        file_handler = RotatingFileHandler(
            "access.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(file_handler)
    except OSError:
        # لو نظام الملفات على المنصة السحابية للقراءة فقط، تجاهل ملف اللوق وخلّك على الشاشة فقط
        pass


def _get_client_ip():
    """
    يجيب IP الحقيقي للمستخدم حتى خلف بروكسي Railway،
    عبر هيدر X-Forwarded-For إذا وُجد.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def init_security(app):
    """
    فعّل هذه الدالة مرة واحدة بعد إنشاء app.
    تسجّل كل طلب يدخل السيرفر: من (IP)، أي رابط، أي طريقة، وكم استغرق، وأي كود رد.
    """

    @app.before_request
    def _log_start():
        g._start_time = time.time()

    @app.after_request
    def _log_request(response):
        duration_ms = int((time.time() - getattr(g, "_start_time", time.time())) * 1000)
        ip = _get_client_ip()
        user = getattr(g, "auth_user", "-")
        logger.info(
            f"IP={ip} | USER={user} | {request.method} {request.path} "
            f"| STATUS={response.status_code} | {duration_ms}ms | UA={request.headers.get('User-Agent', '-')}"
        )
        return response

    logger.info("=== تم تشغيل نظام التسجيل (Logging) بنجاح ===")


# =========================
# 2) الحماية (Basic Auth)
# =========================

def _check_credentials(username, password):
    expected_user = os.environ.get("APP_USER")
    expected_pass = os.environ.get("APP_PASS")

    if not expected_user or not expected_pass:
        # لو ما ضبطت المتغيرات في Railway، امنع الوصول بالكامل بدل ما تفتحه بالخطأ للجميع
        logger.warning("APP_USER أو APP_PASS غير مضبوطة في متغيرات البيئة! الوصول مرفوض افتراضيًا.")
        return False

    return username == expected_user and password == expected_pass


def _unauthorized_response():
    return Response(
        "🔒 تسجيل الدخول مطلوب للوصول لهذا التطبيق.",
        401,
        {"WWW-Authenticate": 'Basic realm="Restricted Access"'},
    )


def requires_auth(f):
    """
    ضع هذا الـ decorator فوق أي route تبي تحميه بكلمة مرور.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        ip = _get_client_ip()

        if not auth:
            logger.warning(f"محاولة دخول بدون بيانات اعتماد | IP={ip} | PATH={request.path}")
            return _unauthorized_response()

        if not _check_credentials(auth.username, auth.password):
            logger.warning(
                f"محاولة دخول فاشلة | IP={ip} | USERNAME_TRIED={auth.username} | PATH={request.path}"
            )
            return _unauthorized_response()

        # نجح تسجيل الدخول — نحفظ اسم المستخدم عشان يظهر في سطر اللوق العام
        g.auth_user = auth.username
        logger.info(f"دخول ناجح | IP={ip} | USER={auth.username} | PATH={request.path}")

        return f(*args, **kwargs)

    return decorated