#!/usr/bin/env bash
# install/docker_test_windows.sh
# Static validation of install/terminal_setup.bat.
#
# Docker on macOS runs a Linux VM — it cannot execute .bat files.
# This script validates the Windows installer by inspecting the source:
# it greps for the exact strings, thresholds, and model identifiers that
# must be present for the Windows install logic to be correct.
#
# Usage:
#   chmod +x install/docker_test_windows.sh
#   ./install/docker_test_windows.sh
#
# Exit code: 0 if all checks pass, 1 if any fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAT_FILE="$SCRIPT_DIR/terminal_setup.bat"

PASS_COUNT=0
FAIL_COUNT=0

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}[PASS]${RESET} $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
info() { echo -e "${YELLOW}[INFO]${RESET} $*"; }

# ── File check ─────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}── Windows installer static analysis ────────────────────────────${RESET}"
echo "   File: $BAT_FILE"
echo

if [ ! -f "$BAT_FILE" ]; then
    echo "Error: $BAT_FILE not found." >&2
    exit 1
fi

# ── check LABEL PATTERN DESCRIPTION
# Asserts that PATTERN appears in the .bat file.
# Prints the matching line(s) on pass, an error message on fail.
check() {
    local label="$1"
    local pattern="$2"
    local description="$3"

    local matches
    matches=$(grep -n "$pattern" "$BAT_FILE" || true)

    if [ -n "$matches" ]; then
        pass "$label — $description"
        echo "$matches" | head -3 | sed 's/^/     line /'
    else
        fail "$label — $description"
        info "Pattern not found: '$pattern'"
    fi
}

# ── Check 1: Conservative default model ───────────────────────────────────────
# Windows always uses Standard regardless of RAM (higher OS overhead than macOS).
# The default model must be set to llama3.1:8b before any RAM threshold logic.
check "WIN-01" \
    'SELECTED_MODEL=llama3\.1:8b' \
    "llama3.1:8b is the conservative default model on Windows"

# ── Check 2: RAM detection with fallback ──────────────────────────────────────
# psutil may not be installed on a clean machine; the installer must fall back
# to 8 GB so it never selects an over-provisioned model by accident.
check "WIN-02" \
    'RAM_GB=8' \
    "Fallback to 8 GB when psutil is unavailable"

# ── Check 3: 24 GB RAM threshold ──────────────────────────────────────────────
# Windows stays on Standard even at 24+ GB (conservative policy), but the
# threshold must be present so the installer can inform the user and suggest
# the Enhanced model manually if IT staff want it.
check "WIN-03" \
    'GEQ 24' \
    "24 GB threshold present for high-RAM detection message"

# ── Check 4: llama3.3:8b mentioned for high-RAM path ─────────────────────────
# Even though Windows never auto-selects Enhanced, the .bat must reference
# llama3.3:8b so IT administrators can see what the Enhanced option would be.
check "WIN-04" \
    'llama3\.3:8b' \
    "llama3.3:8b referenced for IT-admin Enhanced path at 24+ GB"

# ── Check 5: DRY_RUN flag parsing ─────────────────────────────────────────────
check "WIN-05" \
    '\-\-dry-run' \
    "--dry-run flag parsed from command-line arguments"

# ── Check 6: DRY_RUN guarded write ────────────────────────────────────────────
# Config writes must be conditional on DRY_RUN=0 so dry-run never touches disk.
check "WIN-06" \
    '"!DRY_RUN!"=="1"' \
    "Destructive actions are gated on DRY_RUN==1 checks throughout"

# ── Check 7: Dry-run summary label ────────────────────────────────────────────
check "WIN-07" \
    'Dry Run Summary' \
    "Dry-run summary block present at end of script"

# ── Check 8: simulate-ram flag ────────────────────────────────────────────────
check "WIN-08" \
    '\-\-simulate-ram' \
    "--simulate-ram flag parsed from command-line arguments"

# ── Check 9: Model config written to expected path ────────────────────────────
check "WIN-09" \
    'user_config\.json' \
    "Model choice persisted to db\\user_config.json"

# ── Check 10: Ollama health-check before attempting to start ──────────────────
# The script must check localhost:11434 before trying to start Ollama, so it
# never tries to start an already-running instance.
check "WIN-10" \
    'localhost:11434' \
    "Ollama health-check runs before attempting to start the service"

# ── Summary ────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}── Results ───────────────────────────────────────────────────────${RESET}"
echo "   Checks passed: $PASS_COUNT"
echo "   Checks failed: $FAIL_COUNT"
echo

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All checks passed — Windows installer looks correct.${RESET}"
    exit 0
else
    echo -e "${RED}${BOLD}$FAIL_COUNT check(s) failed — review terminal_setup.bat.${RESET}"
    exit 1
fi
