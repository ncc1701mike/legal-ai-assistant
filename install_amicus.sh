#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        AMICUS AI — INSTALLER v2                            ║
# ║              Privileged Intelligence. Zero Cloud Risk.                      ║
# ║                                                                              ║
# ║  Supports: macOS · Apple Silicon (M1/M2/M3/M4) · Intel                     ║
# ║  Safe to re-run — resumes from where it left off.                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── NO set -e. We handle errors explicitly per step. ─────────────────────────

# ── Colors ────────────────────────────────────────────────────────────────────
TEAL='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ── Config ────────────────────────────────────────────────────────────────────
INSTALL_DIR="$HOME/AmicusAI"
REPO_URL="https://github.com/ncc1701mike/legal-ai-assistant.git"
PYTHON_VERSION="3.11"
MODEL="llama3.1:8b"
VENV_DIR="$INSTALL_DIR/venv"
LOG_FILE="$HOME/amicus_install.log"
STATE_FILE="$HOME/.amicus_install_state"

# ── Helpers ───────────────────────────────────────────────────────────────────
step()    { echo -e "\n${TEAL}${BOLD}▶  $1${RESET}"; echo "[$(date '+%H:%M:%S')] STEP: $1" >> "$LOG_FILE"; }
success() { echo -e "   ${GREEN}✔  $1${RESET}"; echo "[$(date '+%H:%M:%S')] OK: $1" >> "$LOG_FILE"; }
warn()    { echo -e "   ${YELLOW}⚠  $1${RESET}"; echo "[$(date '+%H:%M:%S')] WARN: $1" >> "$LOG_FILE"; }
fail()    {
    echo -e "\n${RED}${BOLD}✘  INSTALLATION FAILED${RESET}"
    echo -e "   ${RED}$1${RESET}"
    echo -e "   ${DIM}Check the log for details: ${LOG_FILE}${RESET}"
    echo -e "   ${DIM}Re-run this installer to resume from where it left off.${RESET}"
    echo "[$(date '+%H:%M:%S')] FAIL: $1" >> "$LOG_FILE"
    exit 1
}

# State tracking — skip completed steps on re-run
mark_done() { echo "$1" >> "$STATE_FILE"; }
is_done()   { grep -q "^$1$" "$STATE_FILE" 2>/dev/null; }

# Detect architecture
if [[ "$(uname -m)" == "arm64" ]]; then
    ARCH="apple_silicon"
    BREW_PREFIX="/opt/homebrew"
else
    ARCH="intel"
    BREW_PREFIX="/usr/local"
fi

# ── Header ────────────────────────────────────────────────────────────────────
clear
echo ""
echo -e "${TEAL}${BOLD}"
echo "   ╔══════════════════════════════════════════════╗"
echo "   ║           AMICUS AI — INSTALLER v2           ║"
echo "   ║    Privileged Intelligence. Zero Cloud Risk. ║"
echo "   ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "   Architecture : ${BOLD}$(uname -m)${RESET}"
echo -e "   Install path : ${BOLD}$INSTALL_DIR${RESET}"
echo -e "   Log file     : ${DIM}$LOG_FILE${RESET}"
echo ""
echo -e "   ${YELLOW}Estimated time: 15–25 minutes (model download is ~4.7GB)${RESET}"
echo -e "   ${DIM}Safe to re-run if interrupted — resumes from last completed step.${RESET}"
echo ""
echo -e "   Press ${BOLD}Enter${RESET} to begin, or Ctrl+C to cancel."
read -r

# Start log
echo "=== Amicus AI Install Log — $(date) ===" > "$LOG_FILE"
echo "Architecture: $ARCH ($BREW_PREFIX)" >> "$LOG_FILE"

# ── STEP 1: Xcode Command Line Tools ─────────────────────────────────────────
step "Step 1/8 — Xcode Command Line Tools"

if is_done "xcode"; then
    success "Already installed (skipping)"
elif xcode-select -p &>/dev/null; then
    success "Already installed"
    mark_done "xcode"
else
    echo ""
    echo -e "   ${YELLOW}A dialog box will appear asking you to install developer tools.${RESET}"
    echo -e "   ${YELLOW}Click 'Install' in that dialog and wait for it to complete.${RESET}"
    echo -e "   ${YELLOW}Once the dialog says 'Software installed', press Enter here to continue.${RESET}"
    echo ""
    xcode-select --install 2>/dev/null || true

    read -r -p "   Press Enter once the Xcode tools dialog says 'Done' or 'Software installed': "

    if xcode-select -p &>/dev/null; then
        success "Xcode Command Line Tools installed"
        mark_done "xcode"
    else
        fail "Xcode tools installation could not be verified.\nPlease run: xcode-select --install\nWait for it to finish, then re-run this installer."
    fi
fi

# ── STEP 2: Homebrew ──────────────────────────────────────────────────────────
step "Step 2/8 — Homebrew"

if is_done "homebrew"; then
    success "Already installed (skipping)"
    eval "$($BREW_PREFIX/bin/brew shellenv)" 2>/dev/null || true
elif [[ -f "$BREW_PREFIX/bin/brew" ]]; then
    success "Already installed"
    eval "$($BREW_PREFIX/bin/brew shellenv)"
    mark_done "homebrew"
else
    echo ""
    echo -e "   ${YELLOW}Installing Homebrew. You may be prompted for your Mac password.${RESET}"
    echo -e "   ${DIM}This takes 3–5 minutes...${RESET}"
    echo ""

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    BREW_EXIT=$?

    if [[ $BREW_EXIT -ne 0 ]]; then
        fail "Homebrew installation failed (exit $BREW_EXIT).\nCheck $LOG_FILE for details."
    fi

    # Add brew to PATH
    if [[ "$ARCH" == "apple_silicon" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> "$HOME/.zprofile"
        eval "$(/usr/local/bin/brew shellenv)"
    fi

    if command -v brew &>/dev/null; then
        success "Homebrew installed"
        mark_done "homebrew"
    else
        fail "Homebrew installed but 'brew' not found in PATH.\nClose Terminal, open a new one, and re-run this installer."
    fi
fi

# Ensure brew is on PATH
eval "$($BREW_PREFIX/bin/brew shellenv)" 2>/dev/null || true

# ── STEP 3: Python 3.11 ───────────────────────────────────────────────────────
step "Step 3/8 — Python $PYTHON_VERSION"

if is_done "python"; then
    success "Already installed (skipping)"
elif brew list python@${PYTHON_VERSION} &>/dev/null; then
    success "Already installed"
    mark_done "python"
else
    echo -e "   ${DIM}Installing Python $PYTHON_VERSION...${RESET}"
    brew install python@${PYTHON_VERSION} >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Python $PYTHON_VERSION installation failed. Check $LOG_FILE."
    fi
    success "Python $PYTHON_VERSION installed"
    mark_done "python"
fi

# Find the Python binary
PYTHON_BIN="$BREW_PREFIX/opt/python@${PYTHON_VERSION}/bin/python${PYTHON_VERSION}"
if [[ ! -f "$PYTHON_BIN" ]]; then
    PYTHON_BIN=$(command -v python3.11 2>/dev/null || echo "")
fi
if [[ -z "$PYTHON_BIN" ]]; then
    fail "Python 3.11 binary not found. Check $LOG_FILE."
fi

# ── STEP 4: Tesseract OCR ─────────────────────────────────────────────────────
step "Step 4/8 — Tesseract OCR"

if is_done "tesseract"; then
    success "Already installed (skipping)"
elif command -v tesseract &>/dev/null; then
    success "Already installed"
    mark_done "tesseract"
else
    echo -e "   ${DIM}Installing Tesseract OCR...${RESET}"
    brew install tesseract >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Tesseract installation failed. Check $LOG_FILE."
    fi
    success "Tesseract installed"
    mark_done "tesseract"
fi

# ── STEP 5: Ollama ────────────────────────────────────────────────────────────
step "Step 5/8 — Ollama"

if is_done "ollama"; then
    success "Already installed (skipping)"
elif command -v ollama &>/dev/null; then
    success "Already installed"
    mark_done "ollama"
else
    echo -e "   ${DIM}Installing Ollama...${RESET}"
    brew install ollama >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Ollama installation failed. Check $LOG_FILE."
    fi
    if command -v ollama &>/dev/null; then
        success "Ollama installed"
        mark_done "ollama"
    else
        fail "Ollama not found after install. Check $LOG_FILE."
    fi
fi

# Start Ollama server and wait until it's actually responding
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "   ${DIM}Starting Ollama server...${RESET}"
    ollama serve >> "$LOG_FILE" 2>&1 &
    echo -e "   ${DIM}Waiting for Ollama to be ready...${RESET}"
    READY=false
    for i in {1..20}; do
        sleep 2
        if ollama list &>/dev/null 2>&1; then
            READY=true
            break
        fi
    done
    if [[ "$READY" == "false" ]]; then
        fail "Ollama server did not start in time. Check $LOG_FILE and try re-running."
    fi
    success "Ollama server ready"
else
    success "Ollama server already running"
fi

# ── STEP 6: Download Llama 3.1 8B ────────────────────────────────────────────
step "Step 6/8 — Llama 3.1 8B model (~4.7GB)"

if is_done "model"; then
    success "Already downloaded (skipping)"
elif ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    success "Already downloaded"
    mark_done "model"
else
    echo ""
    echo -e "   ${YELLOW}Downloading llama3.1:8b (~4.7GB).${RESET}"
    echo -e "   ${YELLOW}This will take several minutes on most connections.${RESET}"
    echo -e "   ${DIM}If interrupted, re-running will resume the download.${RESET}"
    echo ""

    ollama pull ${MODEL}
    PULL_EXIT=$?

    if [[ $PULL_EXIT -ne 0 ]]; then
        fail "Model download failed (exit $PULL_EXIT).\nRe-run this installer to resume — partial downloads are saved."
    fi

    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        success "llama3.1:8b downloaded and ready"
        mark_done "model"
    else
        fail "Model pull completed but model not found. Try re-running."
    fi
fi

# ── STEP 7: Clone repo and install packages ───────────────────────────────────
step "Step 7/8 — Amicus AI application"

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo -e "   ${DIM}Existing installation found — pulling latest updates...${RESET}"
    cd "$INSTALL_DIR"
    git pull origin main >> "$LOG_FILE" 2>&1
    success "Updated to latest version"
else
    echo -e "   ${DIM}Cloning repository...${RESET}"
    git clone "$REPO_URL" "$INSTALL_DIR" >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Could not clone repository. Check internet connection and try again."
    fi
    success "Repository cloned"
fi

cd "$INSTALL_DIR"

# Virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "   ${DIM}Creating virtual environment...${RESET}"
    "$PYTHON_BIN" -m venv "$VENV_DIR" >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Could not create virtual environment. Check $LOG_FILE."
    fi
    success "Virtual environment created"
else
    success "Virtual environment already exists"
fi

# Python packages
if is_done "packages"; then
    success "Python packages already installed (skipping)"
else
    echo -e "   ${DIM}Installing Python packages — this may take a few minutes...${RESET}"
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet >> "$LOG_FILE" 2>&1
    "$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
    if [[ $? -ne 0 ]]; then
        fail "Package installation failed. Check $LOG_FILE."
    fi
    success "Python packages installed"
    mark_done "packages"
fi

# Required directories
mkdir -p "$INSTALL_DIR/db/chroma"
mkdir -p "$INSTALL_DIR/data"

# ── STEP 8: Desktop shortcut ──────────────────────────────────────────────────
step "Step 8/8 — Desktop shortcut"

LAUNCH_SCRIPT="$HOME/Desktop/Launch Amicus AI.command"

cat > "$LAUNCH_SCRIPT" << 'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/AmicusAI"
VENV_DIR="$INSTALL_DIR/venv"

echo ""
echo "  ⚖  Starting Amicus AI..."
echo ""

# Set up Homebrew PATH
if [[ "$(uname -m)" == "arm64" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
else
    eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null || true
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "  Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Launch Amicus AI
cd "$INSTALL_DIR"
"$VENV_DIR/bin/streamlit" run app.py
LAUNCHER

chmod +x "$LAUNCH_SCRIPT"
success "Desktop shortcut created"

# ── Complete ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${TEAL}${BOLD}"
echo "   ╔══════════════════════════════════════════════╗"
echo "   ║        INSTALLATION COMPLETE  ✔              ║"
echo "   ╚══════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "   ${BOLD}Installed at:${RESET}    $INSTALL_DIR"
echo -e "   ${BOLD}To launch:${RESET}       Double-click ${BOLD}'Launch Amicus AI'${RESET} on your Desktop"
echo -e "   ${BOLD}Install log:${RESET}     $LOG_FILE"
echo ""
echo -e "   ${GREEN}All done. Amicus AI is ready to use.${RESET}"
echo ""

rm -f "$STATE_FILE"
echo "[$(date '+%H:%M:%S')] Installation completed successfully." >> "$LOG_FILE"
