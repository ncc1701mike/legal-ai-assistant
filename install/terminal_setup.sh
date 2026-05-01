#!/usr/bin/env bash
# install/terminal_setup.sh
# Amicus AI — first-run installer for macOS and Linux.
# Idempotent: safe to run multiple times; skips steps already completed.
#
# Usage:
#   chmod +x install/terminal_setup.sh
#   ./install/terminal_setup.sh [--dry-run] [--simulate-ram N]
#
# Flags:
#   --dry-run          Simulate the full install without making any changes.
#   --simulate-ram N   Pretend the machine has N GB of RAM for model selection
#                      (useful for testing hardware-tier logic).
#
# Examples:
#   ./install/terminal_setup.sh --dry-run
#   ./install/terminal_setup.sh --dry-run --simulate-ram 8
#   ./install/terminal_setup.sh --dry-run --simulate-ram 32

set -euo pipefail

# ── Flag parsing ──────────────────────────────────────────────────────────────
DRY_RUN=false
SIMULATE_RAM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true;            shift ;;
        --simulate-ram)    SIMULATE_RAM="$2";        shift 2 ;;
        --simulate-ram=*)  SIMULATE_RAM="${1#*=}";   shift ;;
        *)                                           shift ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
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

dryrun_cmd()   { echo -e "${YELLOW}[DRY RUN]${RESET} Would run:  $*"; }
dryrun_write() { echo -e "${YELLOW}[DRY RUN]${RESET} Would write: $*"; }

# In dry-run, non-fatal errors are downgraded to warnings so the full
# simulation can complete and show a meaningful summary.
maybe_exit() {
    local code="${1:-1}"
    if [ "$DRY_RUN" = "true" ]; then
        warn "(dry-run) A real install would stop here — continuing simulation."
    else
        exit "$code"
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}⚖️  Amicus AI — Setup${RESET}"
echo -e "   Privacy-first legal document analysis"
echo

if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}${BOLD}🔍  DRY RUN MODE — No changes will be made to your system${RESET}"
    if [ -n "$SIMULATE_RAM" ]; then
        echo -e "    Simulating RAM: ${SIMULATE_RAM} GB"
    fi
    echo
fi

# Variables collected for the dry-run summary
_SUMMARY_PYTHON=""
_SUMMARY_VENV=""
_SUMMARY_PKGS=""
_SUMMARY_OLLAMA_STATUS=""
_SUMMARY_MODEL=""
_SUMMARY_MODEL_SIZE=""
_SUMMARY_MODEL_ACTION=""
_SUMMARY_OS=""


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
    maybe_exit 1
    # In dry-run only: set a placeholder so the rest of the simulation runs
    PY_VER="(not found)"
    _SUMMARY_PYTHON="Python 3.11+ not found — install required before running"
else
    success "Found Python $PY_VER ($PYTHON_CMD)"
    _SUMMARY_PYTHON="Python $PY_VER ($PYTHON_CMD)"
fi


# ── Step 2: Virtual environment ───────────────────────────────────────────────
step "Step 2/5 — Setting up virtual environment"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    success "Virtual environment already exists — skipping creation"
    _SUMMARY_VENV="venv/ already exists (would skip)"
else
    if [ "$DRY_RUN" = "true" ]; then
        dryrun_cmd "$PYTHON_CMD -m venv $VENV_DIR"
        _SUMMARY_VENV="venv/ (would be created at $VENV_DIR)"
    else
        info "Creating virtual environment in $VENV_DIR ..."
        "$PYTHON_CMD" -m venv "$VENV_DIR"
        success "Virtual environment created"
        _SUMMARY_VENV="venv/ created at $VENV_DIR"
    fi
fi

if [ "$DRY_RUN" = "true" ]; then
    dryrun_cmd "source $VENV_DIR/bin/activate"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    info "Activated: $VENV_DIR"
fi


# ── Step 3: Python dependencies ───────────────────────────────────────────────
step "Step 3/5 — Installing Python dependencies"

REQUIREMENTS="$PROJECT_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    error "requirements.txt not found at $REQUIREMENTS"
    maybe_exit 1
    _SUMMARY_PKGS="requirements.txt not found"
else
    # Count packages (read-only — always runs)
    PKG_COUNT=$(grep -c '.' "$REQUIREMENTS" 2>/dev/null || echo "?")
    _SUMMARY_PKGS="$PKG_COUNT packages from requirements.txt"

    if [ "$DRY_RUN" = "true" ]; then
        dryrun_cmd "pip install --quiet --upgrade pip"
        dryrun_cmd "pip install --quiet -r $REQUIREMENTS"
        info "Would install $_SUMMARY_PKGS"
    else
        info "Running pip install (this may take a few minutes on first run)..."
        pip install --quiet --upgrade pip
        pip install --quiet -r "$REQUIREMENTS"
        success "Python dependencies installed"
    fi
fi


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
    _SUMMARY_OLLAMA_STATUS="NOT INSTALLED — install required"
    maybe_exit 1
else
    success "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"

    # Health check always runs — read-only, shows real Ollama status
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        success "Ollama is already running"
        _SUMMARY_OLLAMA_STATUS="running"
    else
        _SUMMARY_OLLAMA_STATUS="not running"
        if [ "$DRY_RUN" = "true" ]; then
            info "Ollama is not running"
            if [ "$(uname -s)" = "Darwin" ]; then
                dryrun_cmd "open -a Ollama  (or: nohup ollama serve &)"
            else
                dryrun_cmd "nohup ollama serve &"
            fi
        else
            info "Ollama is not running — attempting to start..."
            if [ "$(uname -s)" = "Darwin" ]; then
                open -a Ollama 2>/dev/null || nohup ollama serve >/dev/null 2>&1 &
            else
                nohup ollama serve >/dev/null 2>&1 &
            fi

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
    fi
fi


# ── Step 5: Select and pull the right model ───────────────────────────────────
step "Step 5/5 — Selecting and downloading your Analysis Engine"

# RAM detection — always runs (read-only); can be overridden by --simulate-ram
if [ -n "$SIMULATE_RAM" ]; then
    RAM_GB="$SIMULATE_RAM"
    info "Simulated RAM: ${RAM_GB} GB (--simulate-ram)"
else
    RAM_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" 2>/dev/null || echo "8")
    info "Detected RAM: ${RAM_GB} GB"
fi

OS_NAME=$(uname -s)
_SUMMARY_OS=$([ "$OS_NAME" = "Darwin" ] && echo "macOS" || echo "Linux")

# Conservative model selection (same logic as get_safe_default_model())
if [ "$RAM_GB" -ge 24 ] && [ "$OS_NAME" = "Darwin" ]; then
    SELECTED_MODEL="llama3.3:8b"
    MODEL_SIZE="5.0"
    PROFILE_NAME="Enhanced"
else
    SELECTED_MODEL="llama3.1:8b"
    MODEL_SIZE="4.7"
    PROFILE_NAME="Standard"
fi

_SUMMARY_MODEL="$SELECTED_MODEL ($PROFILE_NAME)"
_SUMMARY_MODEL_SIZE="${MODEL_SIZE} GB"

info "Selected profile: $PROFILE_NAME ($SELECTED_MODEL)"

# Check if already installed (read-only — always runs if Ollama is available)
if command -v ollama &>/dev/null && ollama list 2>/dev/null | grep -q "^${SELECTED_MODEL}"; then
    if [ "$DRY_RUN" = "true" ]; then
        success "Model $SELECTED_MODEL is already installed — would skip download"
        _SUMMARY_MODEL_ACTION="already installed (no download needed)"
    else
        success "Model $SELECTED_MODEL is already installed — skipping download"
    fi
else
    _SUMMARY_MODEL_ACTION="download ${MODEL_SIZE} GB (one time only)"
    if [ "$DRY_RUN" = "true" ]; then
        dryrun_cmd "ollama pull $SELECTED_MODEL  (${MODEL_SIZE} GB download)"
    else
        info "Downloading your Analysis Engine (${MODEL_SIZE} GB — one time only)..."
        echo "  This may take 10–20 minutes depending on your internet speed."
        echo
        ollama pull "$SELECTED_MODEL"
        success "Model $SELECTED_MODEL downloaded successfully"
    fi
fi


# ── Write model choice to config ──────────────────────────────────────────────
if [ "$DRY_RUN" = "true" ]; then
    dryrun_write "$CONFIG_PATH  (primary_model: $SELECTED_MODEL)"
else
    mkdir -p "$(dirname "$CONFIG_PATH")"

    if [ -f "$CONFIG_PATH" ]; then
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
fi


# ── Dry-run summary ───────────────────────────────────────────────────────────
if [ "$DRY_RUN" = "true" ]; then
    echo
    echo -e "${BOLD}── Dry Run Summary ──────────────────────────────────────────────${RESET}"
    echo "  Python:          $_SUMMARY_PYTHON"
    echo "  Detected RAM:    ${RAM_GB} GB$([ -n "$SIMULATE_RAM" ] && echo " (simulated)" || true)"
    echo "  Detected OS:     $_SUMMARY_OS"
    echo "  Selected model:  $_SUMMARY_MODEL"
    echo "  Model action:    $_SUMMARY_MODEL_ACTION"
    echo "  Ollama status:   $_SUMMARY_OLLAMA_STATUS"
    echo "  Would download:  $_SUMMARY_MODEL_SIZE"
    echo "  Would write:     db/user_config.json"
    echo "  Would create:    $_SUMMARY_VENV"
    echo "  Would install:   $_SUMMARY_PKGS"
    echo -e "${BOLD}─────────────────────────────────────────────────────────────────${RESET}"
    echo "  Run without --dry-run to perform actual installation."
    echo
    exit 0
fi


# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo -e "${GREEN}${BOLD}✅  Amicus is ready.${RESET}"
echo
echo "  Run:  streamlit run app.py"
echo "  Or:   ./launch.sh"
echo
