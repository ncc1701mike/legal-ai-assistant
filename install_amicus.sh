#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        AMICUS AI — INSTALLER                           ║
# ║              Privileged Intelligence. Zero Cloud Risk.                  ║
# ║                                                                          ║
# ║  Installs everything Amicus AI needs on a fresh Apple Silicon Mac.      ║
# ║  Safe to re-run — skips steps already completed.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -e  # Exit on any unhandled error

# ── Colors ────────────────────────────────────────────────────────────────────
TEAL='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Config ────────────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/AmicusAI"
REPO_URL="https://github.com/ncc1701mike/legal-ai-assistant.git"
PYTHON_VERSION="3.11"
MODEL="llama3.1:8b"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="$HOME/amicus_install.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
step()    { echo -e "\n${TEAL}${BOLD}▶ $1${RESET}"; }
success() { echo -e "${GREEN}✔ $1${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $1${RESET}"; }
fail()    { echo -e "${RED}✘ $1${RESET}"; exit 1; }
log()     { echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"; }

# ── Header ────────────────────────────────────────────────────────────────────
clear
echo -e "${TEAL}${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║           AMICUS AI — INSTALLER              ║"
echo "  ║    Privileged Intelligence. Zero Cloud Risk. ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  This will set up Amicus AI on your Mac."
echo -e "  ${YELLOW}Estimated time: 10–20 minutes (model download is ~4.7GB)${RESET}"
echo -e "  Log file: ${LOG_FILE}\n"
echo -e "  Press ${BOLD}Enter${RESET} to begin, or Ctrl+C to cancel."
read -r

# Start fresh log
echo "=== Amicus AI Install Log — $(date) ===" > "$LOG_FILE"

# ── STEP 1: Xcode Command Line Tools ─────────────────────────────────────────
step "Step 1/8 — Checking Xcode Command Line Tools"
if xcode-select -p &>/dev/null; then
    success "Xcode tools already installed"
    log "Xcode tools: already present"
else
    warn "Installing Xcode Command Line Tools — a dialog will appear. Click Install."
    xcode-select --install
    echo "  Waiting for installation to complete..."
    until xcode-select -p &>/dev/null; do sleep 5; done
    success "Xcode tools installed"
    log "Xcode tools: freshly installed"
fi

# ── STEP 2: Homebrew ──────────────────────────────────────────────────────────
step "Step 2/8 — Checking Homebrew"
if command -v brew &>/dev/null; then
    success "Homebrew already installed"
    log "Homebrew: already present at $(command -v brew)"
else
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >> "$LOG_FILE" 2>&1
    # Add brew to PATH for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    eval "$(/opt/homebrew/bin/brew shellenv)"
    success "Homebrew installed"
    log "Homebrew: freshly installed"
fi

# Ensure brew is on PATH for Apple Silicon
if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# ── STEP 3: Python 3.11 ───────────────────────────────────────────────────────
step "Step 3/8 — Checking Python ${PYTHON_VERSION}"
if brew list python@${PYTHON_VERSION} &>/dev/null; then
    success "Python ${PYTHON_VERSION} already installed"
    log "Python ${PYTHON_VERSION}: already present"
else
    echo "  Installing Python ${PYTHON_VERSION}..."
    brew install python@${PYTHON_VERSION} >> "$LOG_FILE" 2>&1
    success "Python ${PYTHON_VERSION} installed"
    log "Python ${PYTHON_VERSION}: freshly installed"
fi
PYTHON_BIN="$(brew --prefix python@${PYTHON_VERSION})/bin/python${PYTHON_VERSION}"

# ── STEP 4: Tesseract OCR ─────────────────────────────────────────────────────
step "Step 4/8 — Checking Tesseract OCR"
if command -v tesseract &>/dev/null; then
    success "Tesseract already installed ($(tesseract --version 2>&1 | head -1))"
    log "Tesseract: already present"
else
    echo "  Installing Tesseract OCR..."
    brew install tesseract >> "$LOG_FILE" 2>&1
    success "Tesseract installed"
    log "Tesseract: freshly installed"
fi

# ── STEP 5: Ollama ────────────────────────────────────────────────────────────
step "Step 5/8 — Checking Ollama"
if command -v ollama &>/dev/null; then
    success "Ollama already installed"
    log "Ollama: already present at $(command -v ollama)"
else
    echo "  Installing Ollama..."
    brew install ollama >> "$LOG_FILE" 2>&1
    success "Ollama installed"
    log "Ollama: freshly installed"
fi

# Start Ollama server in background if not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "  Starting Ollama server..."
    ollama serve >> "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    sleep 3
    log "Ollama server started (PID $OLLAMA_PID)"
fi

# ── STEP 6: Pull Llama 3.1 8B ─────────────────────────────────────────────────
step "Step 6/8 — Checking Llama 3.1 8B model (~4.7GB)"
if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    success "llama3.1:8b already downloaded"
    log "Model: already present"
else
    echo -e "  ${YELLOW}Downloading llama3.1:8b — this will take a few minutes on a good connection.${RESET}"
    echo "  You'll see download progress below:"
    echo ""
    ollama pull ${MODEL} 2>&1 | tee -a "$LOG_FILE"
    echo ""
    success "llama3.1:8b downloaded"
    log "Model: freshly downloaded"
fi

# ── STEP 7: Clone repo & set up venv ──────────────────────────────────────────
step "Step 7/8 — Setting up Amicus AI application"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "  Amicus AI folder found — pulling latest updates..."
    cd "$INSTALL_DIR"
    git pull origin main >> "$LOG_FILE" 2>&1
    success "Updated to latest version"
    log "Repo: pulled latest"
else
    echo "  Cloning Amicus AI from GitHub..."
    git clone "$REPO_URL" "$INSTALL_DIR" >> "$LOG_FILE" 2>&1
    success "Amicus AI cloned to $INSTALL_DIR"
    log "Repo: freshly cloned"
fi

cd "$INSTALL_DIR"

# Create virtual environment
if [[ -d "$VENV_DIR" ]]; then
    success "Virtual environment already exists"
    log "venv: already present"
else
    echo "  Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" >> "$LOG_FILE" 2>&1
    success "Virtual environment created"
    log "venv: freshly created"
fi

# Install Python dependencies
echo "  Installing Python packages (this may take a few minutes)..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet >> "$LOG_FILE" 2>&1
"$VENV_DIR/bin/pip" install -r requirements.txt --quiet >> "$LOG_FILE" 2>&1
success "Python packages installed"
log "pip install: complete"

# Create db/chroma directory if missing
mkdir -p "$INSTALL_DIR/db/chroma"
mkdir -p "$INSTALL_DIR/data"
log "Directories: db/chroma and data ensured"

# ── STEP 8: Create launch script ──────────────────────────────────────────────
step "Step 8/8 — Creating launch shortcut"

LAUNCH_SCRIPT="$HOME/Desktop/Launch Amicus AI.command"
cat > "$LAUNCH_SCRIPT" << 'LAUNCHER'
#!/bin/bash
# Amicus AI — Launch Script
INSTALL_DIR="$HOME/AmicusAI"
VENV_DIR="$INSTALL_DIR/venv"

echo "Starting Amicus AI..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/dev/null &
    sleep 2
fi

# Launch Streamlit
cd "$INSTALL_DIR"
"$VENV_DIR/bin/streamlit" run app.py --server.headless false
LAUNCHER

chmod +x "$LAUNCH_SCRIPT"
success "Launch shortcut created on Desktop"
log "Launch script: created at $LAUNCH_SCRIPT"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${TEAL}${BOLD}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║         INSTALLATION COMPLETE  ✔             ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  ${BOLD}Amicus AI is installed at:${RESET} $INSTALL_DIR"
echo -e "  ${BOLD}To launch in future:${RESET}      Double-click 'Launch Amicus AI' on your Desktop"
echo -e "  ${BOLD}Install log:${RESET}              $LOG_FILE"
echo ""
echo -e "  ${YELLOW}Opening Amicus AI now...${RESET}\n"
log "Installation complete."

# ── Auto-launch ───────────────────────────────────────────────────────────────
# Start Ollama if it died during install
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/dev/null &
    sleep 2
fi

cd "$INSTALL_DIR"
"$VENV_DIR/bin/streamlit" run app.py --server.headless false
