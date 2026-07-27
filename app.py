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
    if not hmac.compare_digest(request.headers.get("X-Ingest-Token", ""), expected):
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
