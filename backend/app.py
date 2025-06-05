from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.abspath('..'))

from query import search_images

app = Flask(__name__)
CORS(app)

# Serve image files
@app.route('/data/<path:filename>')
def serve_image(filename):
    return send_from_directory('../data', filename)

# Handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query_text = data.get("query", "")
    
    # Use your existing image search logic
    results = search_images(query_text, top_k=9)  # Adjust top_k as needed

    # Format paths for frontend access
    formatted = [f"static/val2017/{os.path.basename(path)}" for path in results]
    return jsonify({"results": formatted})

if __name__ == '__main__':
    app.run(debug=True)

