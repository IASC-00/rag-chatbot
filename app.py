import hmac
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from ingest import ingest as run_ingest
from query import pipe
from ratelimit import RateLimiter

load_dotenv()

app = Flask(__name__)
# Behind Caddy every request arrives from the proxy container, so without this the
# per-address limiter below is ONE bucket shared by every visitor. Caddy sets
# X-Forwarded-For itself; trust exactly one hop. Set TRUST_PROXY=0 only when the
# app is reachable directly (then a client could forge the header).
if os.environ.get("TRUST_PROXY", "1") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
# A question is a sentence, not a file. Anything bigger is refused with 413 before
# it is parsed.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000

_CHAT_LIMITER = None


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment; blank or malformed falls back.

    `KEY=` is a common way to mean "unset" in a .env file, and a bare int() on
    that raises at import time.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _const_eq(a: str, b: str) -> bool:
    """Constant-time comparison that tolerates non-ASCII input.

    `hmac.compare_digest` raises TypeError on a str containing non-ASCII. The
    value compared below is a request header, so it is attacker-controlled: one
    accented character turned an unauthorized ingest attempt into an unhandled
    500 rather than a clean 401. Encoding to bytes first accepts any input and
    keeps the timing property.
    """
    return hmac.compare_digest(
        a.encode("utf-8", "surrogateescape"), b.encode("utf-8", "surrogateescape")
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


def _chat_limiter():
    """Per-process limiter for /chat, built once from the environment."""
    global _CHAT_LIMITER
    if _CHAT_LIMITER is None:
        _CHAT_LIMITER = RateLimiter(
            max_requests=_env_int("CHAT_MAX_REQUESTS", 30),
            window_seconds=_env_int("CHAT_WINDOW_SECONDS", 3600),
        )
    return _CHAT_LIMITER


@app.route("/chat", methods=["POST"])
def chat():
    """POST /chat — ask the RAG pipeline. Body: {"query": "..."}

    Two controls, both off-by-default-safe:

    CHAT_TOKEN — when set, an X-Chat-Token header matching it is required. When
    UNSET the endpoint is open, which is the current public-demo behaviour and
    stays the default, so setting nothing changes nothing.

    Rate limiting is always on, because every call here spends money at
    OpenRouter and the endpoint is reachable without credentials. Defaults are
    deliberately generous (30 requests/hour/address) so a demo visitor never
    hits them.
    """
    expected = os.environ.get("CHAT_TOKEN", "")
    if expected and not _const_eq(request.headers.get("X-Chat-Token", ""), expected):
        return jsonify({"error": "Unauthorized"}), 401

    if not _chat_limiter().allow(request.remote_addr or "unknown"):
        return jsonify({"error": "Rate limit exceeded. Try again later."}), 429

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_text = data["query"]
    # A number or list here used to reach Chroma and come back as a 500.
    if not isinstance(query_text, str) or not query_text.strip():
        return jsonify({"error": "'query' must be a non-empty string"}), 400
    if len(query_text) > MAX_QUERY_CHARS:
        return jsonify({"error": f"Question is too long ({MAX_QUERY_CHARS} characters max)."}), 400
    try:
        response = pipe(query_text)
        return jsonify({"query": query_text, "response": response}), 200
    except Exception:
        # Detail stays in the server log; the caller gets nothing internal. In
        # practice this is the upstream model being busy or withdrawn, so say so.
        log.exception("chat failed")
        busy = "The assistant is busy right now. Please try again in a minute."
        return jsonify({"error": busy}), 503


@app.route("/ingest", methods=["POST"])
def trigger_ingest():
    """POST /ingest — rebuild the stores. Requires the X-Ingest-Token header."""
    expected = os.environ.get("INGEST_TOKEN", "")
    if not expected:
        return jsonify({"error": "Ingest is disabled"}), 503
    if not _const_eq(request.headers.get("X-Ingest-Token", ""), expected):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        return jsonify({"status": "ok", **run_ingest()}), 200
    except Exception:
        log.exception("ingest failed")
        return jsonify({"error": "Internal error"}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(debug=debug, port=port)
