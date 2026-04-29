#!/usr/bin/env bash
# install/terminal_setup.sh
# Amicus AI — first-run installer for macOS and Linux.
# Idempotent: safe to run multiple times; skips steps already completed.
#
# Usage:
#   chmod +x install/terminal_setup.sh
#   ./install/terminal_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv"
CONFIG_PATH="$PROJECT_DIR/db/user_config.json"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[amicus]${RESET} $*"; }
success() { echo -e "${GREEN}[amicus]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[amicus]${RESET} $*"; }
error()   { echo -e "${RED}[amicus] ERROR:${RESET} $*"; }
step()    { echo; echo -e "${BOLD}──────────────────────────────────────────${RESET}"; echo -e "${BOLD}$*${RESET}"; }

echo
echo -e "${BOLD}⚖️  Amicus AI — Setup${RESET}"
echo -e "   Privacy-first legal document analysis"
echo


# ── Step 1: Python 3.11+ ──────────────────────────────────────────────────────
step "Step 1/5 — Checking Python version"

PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "Python 3.11 or newer is required but was not found."
    echo
    echo "  Download Python from: https://www.python.org/downloads/"
    echo "  On macOS you can also run: brew install python@3.11"
    echo
    exit 1
fi

success "Found Python $PY_VER ($PYTHON_CMD)"


# ── Step 2: Virtual environment ───────────────────────────────────────────────
step "Step 2/5 — Setting up virtual environment"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    success "Virtual environment already exists — skipping creation"
else
    info "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# Activate for the rest of this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
info "Activated: $VENV_DIR"


# ── Step 3: Python dependencies ───────────────────────────────────────────────
step "Step 3/5 — Installing Python dependencies"

REQUIREMENTS="$PROJECT_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    error "requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

info "Running pip install (this may take a few minutes on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$REQUIREMENTS"
success "Python dependencies installed"


# ── Step 4: Ollama ────────────────────────────────────────────────────────────
step "Step 4/5 — Checking Ollama"

if ! command -v ollama &>/dev/null; then
    error "Ollama is not installed."
    echo
    echo "  Install Ollama:"
    echo "    macOS:  https://ollama.ai  (download the app)"
    echo "    Linux:  curl -fsSL https://ollama.ai/install.sh | sh"
    echo
    echo "  After installing, run this script again."
    exit 1
fi

success "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"

# Check if Ollama is already running before attempting to start it
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    success "Ollama is already running"
else
    info "Ollama is not running — attempting to start..."
    if [ "$(uname -s)" = "Darwin" ]; then
        open -a Ollama 2>/dev/null || nohup ollama serve >/dev/null 2>&1 &
    else
        nohup ollama serve >/dev/null 2>&1 &
    fi

    # Wait up to 10 seconds for Ollama to become available
    OLLAMA_READY=0
    for i in 1 2 3 4 5; do
        sleep 2
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            OLLAMA_READY=1
            break
        fi
    done

    if [ "$OLLAMA_READY" -eq 1 ]; then
        success "Ollama started successfully"
    else
        warn "Could not start Ollama automatically."
        echo
        echo "  On macOS: open Ollama from your Applications folder."
        echo "  On Linux: run 'ollama serve' in a separate terminal."
        echo
        read -r -p "  Continue anyway? [y/N] " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
            echo "  Exiting. Start Ollama and run this script again."
            exit 0
        fi
    fi
fi


# ── Step 5: Select and pull the right model ───────────────────────────────────
step "Step 5/5 — Selecting and downloading your Analysis Engine"

# Detect RAM in GB
RAM_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" 2>/dev/null || echo "8")
info "Detected RAM: ${RAM_GB} GB"

# Conservative model selection (same logic as get_safe_default_model())
OS_NAME=$(uname -s)
if [ "$RAM_GB" -ge 24 ] && [ "$OS_NAME" = "Darwin" ]; then
    SELECTED_MODEL="llama3.3:8b"
    MODEL_SIZE="5.0"
    PROFILE_NAME="Enhanced"
else
    SELECTED_MODEL="llama3.1:8b"
    MODEL_SIZE="4.7"
    PROFILE_NAME="Standard"
fi

info "Selected profile: $PROFILE_NAME ($SELECTED_MODEL)"

# Check if already installed
if ollama list 2>/dev/null | grep -q "^${SELECTED_MODEL}"; then
    success "Model $SELECTED_MODEL is already installed — skipping download"
else
    info "Downloading your Analysis Engine (${MODEL_SIZE} GB — one time only)..."
    echo "  This may take 10–20 minutes depending on your internet speed."
    echo
    ollama pull "$SELECTED_MODEL"
    success "Model $SELECTED_MODEL downloaded successfully"
fi


# ── Write model choice to config ──────────────────────────────────────────────
mkdir -p "$(dirname "$CONFIG_PATH")"

if [ -f "$CONFIG_PATH" ]; then
    # Merge: preserve existing keys, set/overwrite primary_model
    python3 - <<PYEOF
import json, pathlib
p = pathlib.Path("$CONFIG_PATH")
try:
    cfg = json.loads(p.read_text())
except Exception:
    cfg = {}
cfg["primary_model"] = "$SELECTED_MODEL"
p.write_text(json.dumps(cfg, indent=2))
PYEOF
else
    echo "{\"primary_model\": \"$SELECTED_MODEL\"}" > "$CONFIG_PATH"
fi

success "Model choice saved to db/user_config.json"


# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo -e "${GREEN}${BOLD}✅  Amicus is ready.${RESET}"
echo
echo "  Run:  streamlit run app.py"
echo "  Or:   ./launch.sh"
echo
