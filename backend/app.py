from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))

from agent import agent_search
from web_ingest import web_search_and_ingest

app = Flask(__name__)
CORS(app)


@app.route("/static/val2017/<path:filename>")
def serve_static_image(filename):
    image_dir = os.path.join(BASE_DIR, "backend", "static", "val2017")
    return send_from_directory(image_dir, filename)


@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json(silent=True) or {}
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({
            "local_results": [],
            "web_results": []
        })

    local_raw = agent_search(query_text, limit=9)

    local_results = [
        f"static/val2017/{os.path.basename(str(p))}"
        for p in local_raw
    ]

    web_results = web_search_and_ingest(query_text, max_images=3)

    return jsonify({
        "local_results": local_results,
        "web_results": web_results
    })


@app.route("/")
def health_check():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)