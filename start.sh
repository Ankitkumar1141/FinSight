#!/bin/bash
set -e

# ── Start FastAPI backend on port 8008 (background) ───────────────────────────
# Use uvicorn directly — reload=True (used in main.py __main__) does not work
# reliably inside Docker background processes.
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8008 \
    --log-level info &

FASTAPI_PID=$!
echo "FastAPI started (PID $FASTAPI_PID)"

# ── Wait for FastAPI to become ready ──────────────────────────────────────────
echo "Waiting for FastAPI to be ready..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8008/api/v1/health')" 2>/dev/null; then
        echo "FastAPI is ready."
        break
    fi
    sleep 3
done

# ── Start Streamlit frontend on port 7860 (foreground) ────────────────────────
exec streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
