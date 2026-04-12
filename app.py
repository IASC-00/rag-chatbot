import os
from flask import Flask, request, jsonify, send_from_directory
from query import pipe
from ingest import ingest as run_ingest
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat - Ask a question to the RAG pipeline.
    Expects json: {"query": "how does X work?"}
    """
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_text = data["query"]
    try:
        response = pipe(query_text)
        return jsonify({"query": query_text, "response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ingest", methods=["POST"])
def trigger_ingest():
    """POST /ingest - Re-run the ingest pipeline (useful for demos)."""
    try:
        run_ingest()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, port=port)
