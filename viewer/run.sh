#!/bin/bash
# Launch the BTC sample viewer locally.
#
# 1. Builds viewer/index.json from price_collector/data/*.csv
# 2. Serves the data-collector folder over HTTP
# 3. Open http://localhost:8765/viewer/ in your browser

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "Building viewer/index.json..."
python3 viewer/build_index.py

PORT=${PORT:-8765}
echo
echo "Serving $ROOT_DIR on http://localhost:$PORT"
echo "Open http://localhost:$PORT/viewer/ in your browser"
echo "Ctrl+C to stop"
echo
python3 -m http.server "$PORT"
