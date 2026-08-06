import hmac
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from ingest import ingest as run_ingest
from query import pipe

load_dotenv()

app = Flask(__name__)
log = logging.getLogger(__name__)


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


@app.route("/chat", methods=["POST"])
def chat():
    """POST /chat — ask the RAG pipeline. Body: {"query": "..."}"""
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_text = data["query"]
    try:
        response = pipe(query_text)
        return jsonify({"query": query_text, "response": response}), 200
    except Exception:
        # Detail stays in the server log; the caller gets nothing internal.
        log.exception("chat failed")
        return jsonify({"error": "Internal error"}), 500


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
