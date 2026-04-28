# modules/setup_wizard.py
# Health-check utilities for the Amicus sidebar status card and warning banner.
# All functions are fast (2s max timeout), never raise, and carry no side effects.

import json
import platform
import requests
import psutil
from dataclasses import dataclass
from pathlib import Path

_USER_CONFIG_PATH = Path("db") / "user_config.json"
_OLLAMA_URL       = "http://localhost:11434"
_TIMEOUT          = 2  # seconds — never block the UI

# Approximate compressed download sizes (GB)
_MODEL_SIZES_GB: dict = {
    "llama3.1:8b":      4.7,
    "llama3.3:8b":      5.0,
    "mistral-nemo:12b": 7.1,
    "llama3.1:70b":     40.0,
}


# ── Status dataclass ──────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    ollama_running:        bool
    recommended_model:     str
    model_installed:       bool
    ram_gb:                float
    platform:              str   # "mac" | "windows" | "linux"
    estimated_download_gb: float
    ready_to_use:          bool  # True only when ollama running + model installed


# ── Public helpers ────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Returns True if db/user_config.json exists and has a model configured."""
    try:
        if not _USER_CONFIG_PATH.exists():
            return False
        cfg = json.loads(_USER_CONFIG_PATH.read_text())
        return bool(cfg.get("primary_model"))
    except Exception:
        return False


def get_safe_default_model() -> str:
    """Returns the single safest model for this hardware.

    Decision logic (always errs toward the smaller model):
      < 24 GB RAM  → llama3.1:8b  (universally safe, leaves OS headroom)
      24+ GB, Mac  → llama3.3:8b  (tested on Apple Silicon, reliable)
      24+ GB, Win  → llama3.1:8b  (Windows has higher OS overhead)
      24+ GB, Linux→ llama3.1:8b  (server default, conservative)

    Nothing larger than 8B is ever auto-selected without explicit IT confirmation.
    """
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return "llama3.1:8b"

    if ram_gb < 24:
        return "llama3.1:8b"

    os_name = platform.system()
    if os_name == "Darwin":
        return "llama3.3:8b"  # Apple Silicon handles this reliably at 24GB+
    return "llama3.1:8b"       # Windows / Linux — stay conservative


def is_ollama_running() -> bool:
    """Returns True if the Ollama daemon is reachable. Timeout: 2s."""
    try:
        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def is_model_installed(model_id: str) -> bool:
    """Returns True if model_id is present in the local Ollama model list."""
    try:
        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=_TIMEOUT)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        return model_id in names
    except Exception:
        return False


def get_model_file_size_gb(model_id: str) -> float:
    """Returns the approximate download size in GB for a model ID."""
    return _MODEL_SIZES_GB.get(model_id, 4.7)


def run_health_check() -> HealthStatus:
    """Returns current Ollama and model health for this machine. Never raises.

    Called on every app load — total wall time is bounded by _TIMEOUT (2s)
    when Ollama is unreachable, otherwise typically < 100ms.

    Intentionally stateless (no module-level caching) so every call reflects
    the true current state of the Ollama daemon.
    """
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram_gb = 8.0

    try:
        os_name = platform.system()
    except Exception:
        os_name = "Unknown"
    plat = "mac" if os_name == "Darwin" else ("windows" if os_name == "Windows" else "linux")

    recommended = get_safe_default_model()
    ollama_ok   = is_ollama_running()
    installed   = is_model_installed(recommended) if ollama_ok else False
    size_gb     = get_model_file_size_gb(recommended)

    return HealthStatus(
        ollama_running        = ollama_ok,
        recommended_model     = recommended,
        model_installed       = installed,
        ram_gb                = ram_gb,
        platform              = plat,
        estimated_download_gb = size_gb,
        ready_to_use          = ollama_ok and installed,
    )
