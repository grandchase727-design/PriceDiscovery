#!/bin/bash
# Launch script for the Streamlit dashboard.
# Backend (FastAPI on :8000) must be running separately — see README.

set -e

cd "$(dirname "$0")"

# Verify backend is reachable
if ! curl -s --max-time 3 -o /dev/null http://localhost:8000/api/meta; then
  echo "⚠ Backend not reachable at http://localhost:8000"
  echo "   Start it first:  cd .. && python3 -m uvicorn api:app --port 8000 &"
  exit 1
fi

# Install dependencies if missing
if ! python3 -c "import streamlit, plotly, pandas, requests" 2>/dev/null; then
  echo "📦 Installing dependencies..."
  pip3 install -q -r requirements.txt
fi

# Launch (use python3 -m streamlit to avoid PATH issues)
echo "🚀 Launching Streamlit dashboard on http://localhost:8501"
exec python3 -m streamlit run app.py
