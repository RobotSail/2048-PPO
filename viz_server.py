"""
Visualization server for 2048 training runs.
Serves the web frontend and provides API for accessing training data.

Usage: python viz_server.py [--port PORT] [--viz-dir DIR]
"""

import argparse
import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, abort

app = Flask(__name__, static_folder="viz")

# Default visualization data directory
VIZ_DATA_DIR = Path("viz_data")


@app.route("/")
def index():
    """Serve the main visualization page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/files")
def list_files():
    """List all available visualization data files."""
    if not VIZ_DATA_DIR.exists():
        return jsonify({"files": []})
    
    files = sorted(VIZ_DATA_DIR.glob("*.json"))
    file_list = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                file_list.append({
                    "filename": f.name,
                    "step": data.get("step", 0),
                    "score": data.get("score", 0),
                    "total_steps": data.get("total_steps", 0),
                })
        except (json.JSONDecodeError, IOError):
            continue
    
    return jsonify({"files": file_list})


@app.route("/api/data/<filename>")
def get_data(filename):
    """Return contents of a specific visualization data file."""
    # Security: only allow .json files and prevent path traversal
    if not filename.endswith(".json") or "/" in filename or "\\" in filename:
        abort(400, "Invalid filename")
    
    filepath = VIZ_DATA_DIR / filename
    if not filepath.exists():
        abort(404, "File not found")
    
    try:
        with open(filepath) as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, IOError) as e:
        abort(500, f"Error reading file: {e}")


def main():
    parser = argparse.ArgumentParser(description="2048 Training Visualization Server")
    parser.add_argument(
        "--port", type=int, default=5050, help="Port to run server on (default: 5050)"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="viz_data",
        help="Directory containing visualization data (default: viz_data)",
    )
    args = parser.parse_args()
    
    global VIZ_DATA_DIR
    VIZ_DATA_DIR = Path(args.viz_dir)
    
    print(f"Starting visualization server...")
    print(f"  Data directory: {VIZ_DATA_DIR.absolute()}")
    print(f"  Open http://localhost:{args.port} in your browser")
    
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
