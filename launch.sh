#!/usr/bin/env bash
# Amicus AI — macOS / Linux launcher
# Usage: ./launch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check virtual environment ─────────────────────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
    echo ""
    echo "ERROR: Virtual environment not found."
    echo ""
    echo "Run the one-time setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  python -m spacy download en_core_web_lg"
    echo ""
    exit 1
fi

# ── Activate virtual environment ──────────────────────────────────────────────
# shellcheck disable=SC1091
source venv/bin/activate

# ── Check Ollama ──────────────────────────────────────────────────────────────
if ! curl -s --max-time 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Ollama is not responding on http://localhost:11434"
    echo ""
    echo "Start Ollama in a separate terminal:"
    echo "  ollama serve"
    echo ""
    echo "Or if you installed the Ollama app, launch it from Applications."
    echo ""
    echo "Continuing — the app will show 'Ollama Not Running' in the sidebar."
    echo ""
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "Starting Amicus AI at http://localhost:8501 ..."
streamlit run app.py
