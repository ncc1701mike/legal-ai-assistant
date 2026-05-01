#!/usr/bin/env bash
# install/docker_test.sh
# Docker-based clean-room test harness for install/terminal_setup.sh.
#
# Builds a fresh container image, runs three --dry-run scenarios that exercise
# the three hardware-tier paths, and asserts expected model-selection strings.
#
# Usage:
#   chmod +x install/docker_test.sh
#   ./install/docker_test.sh
#
# Exit code: 0 if all scenarios pass, 1 if any fail.
# Expected runtime: under 3 minutes (image build ~60-90 s, each run ~5-10 s).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="amicus-installer-test"
PASS_COUNT=0
FAIL_COUNT=0

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}[PASS]${RESET} $*"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
info() { echo -e "${YELLOW}[INFO]${RESET} $*"; }

# ── Check prerequisites ────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "Error: Docker is not installed or not in PATH." >&2
    exit 1
fi

# ── Build image ────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}── Building Docker image ─────────────────────────────────────────${RESET}"
echo "   Image: $IMAGE"
echo "   Context: $PROJECT_DIR"
echo

docker build \
    -f "$PROJECT_DIR/Dockerfile.test" \
    -t "$IMAGE" \
    "$PROJECT_DIR" \
    --quiet \
    && echo "Image built successfully." \
    || { echo "Docker build failed." >&2; exit 1; }

# ── Scenario runner ────────────────────────────────────────────────────────────
# run_scenario LABEL DOCKER_ARGS EXPECTED_STRING1 EXPECTED_STRING2
#   Runs terminal_setup.sh inside the container with DOCKER_ARGS (passed after
#   `bash install/terminal_setup.sh`).  Checks that EXPECTED_STRING1 and
#   EXPECTED_STRING2 both appear in stdout.  Prints matching lines on failure.
run_scenario() {
    local label="$1"
    local args="$2"
    local expect1="$3"
    local expect2="$4"

    echo
    echo -e "${BOLD}── $label ${RESET}"
    echo "   Args: $args"

    # Run the scenario; capture stdout+stderr; never let a non-zero exit abort
    # the harness (the script exits 0 on dry-run; non-zero would be a bug).
    local output
    # shellcheck disable=SC2086
    output=$(docker run --rm "$IMAGE" bash install/terminal_setup.sh $args 2>&1) || true

    local ok=true

    if echo "$output" | grep -q "$expect1"; then
        info "Found expected string: '$expect1'"
    else
        fail "Expected '$expect1' not found in output"
        ok=false
    fi

    if echo "$output" | grep -q "$expect2"; then
        info "Found expected string: '$expect2'"
    else
        fail "Expected '$expect2' not found in output"
        ok=false
    fi

    if [ "$ok" = "true" ]; then
        pass "$label"
        # Print only the model-selection lines for a clean summary
        echo "   Relevant output:"
        echo "$output" | grep -E "profile:|model:|RAM:|Simulated|DRY RUN.*pull|Selected" \
            | sed 's/^/     /' || true
    else
        echo "   Full output (for debugging):"
        echo "$output" | sed 's/^/     /'
    fi
}

# ── Scenarios ──────────────────────────────────────────────────────────────────
#
# Note on OS simulation: Docker runs Linux, so `uname -s` returns "Linux".
# The Enhanced profile (llama3.3:8b) only activates on Darwin with 24+ GB.
# We pass --simulate-os Darwin for the macOS scenarios so the container
# exercises the same code path an actual Mac would follow.
#
# Scenario 1 — 8 GB MacBook Air (Standard profile, no OS simulation needed)
#   Any OS with 8 GB → Standard (llama3.1:8b).
#
# Scenario 2 — 24 GB MacBook Pro (Enhanced profile threshold, simulated macOS)
#   Darwin + 24 GB is the exact boundary where Enhanced activates.
#
# Scenario 3 — 32 GB Mac (Enhanced profile above threshold, simulated macOS)
#   Confirms that 32 GB on macOS still selects the conservative Enhanced
#   (llama3.3:8b, an 8B model) — not a 70B model.

run_scenario \
    "Scenario 1 — 8 GB MacBook Air (Standard)" \
    "--dry-run --simulate-ram 8" \
    "llama3.1:8b" \
    "Standard"

run_scenario \
    "Scenario 2 — 24 GB MacBook Pro (Enhanced, threshold boundary)" \
    "--dry-run --simulate-ram 24 --simulate-os Darwin" \
    "llama3.3:8b" \
    "Enhanced"

run_scenario \
    "Scenario 3 — 32 GB Mac (Enhanced, above threshold, still conservative 8B)" \
    "--dry-run --simulate-ram 32 --simulate-os Darwin" \
    "llama3.3:8b" \
    "Enhanced"

# ── Summary ────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}── Results ───────────────────────────────────────────────────────${RESET}"
echo "   Passed: $PASS_COUNT"
echo "   Failed: $FAIL_COUNT"
echo

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All scenarios passed.${RESET}"
    exit 0
else
    echo -e "${RED}${BOLD}$FAIL_COUNT scenario(s) failed.${RESET}"
    exit 1
fi
