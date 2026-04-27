# modules/hardware_detect.py
# Hardware detection and Ollama model recommendation utilities

import platform
import psutil
import requests
from typing import Dict, List, Optional

# ── Model catalogue per RAM tier ──────────────────────────────────────────────
_MODELS_8GB: List[Dict] = [
    {
        "id": "llama3.1:8b",
        "name": "Llama 3.1 8B (default)",
        "description": "Best for 8GB systems — fast and reliable",
    }
]

_MODELS_16GB: List[Dict] = _MODELS_8GB + [
    {
        "id": "llama3.3:8b",
        "name": "Llama 3.3 8B (recommended)",
        "description": "Improved instruction following over 3.1",
    },
    {
        "id": "mistral-nemo:12b",
        "name": "Mistral Nemo 12B",
        "description": "Better long-document reasoning, 32K context",
    },
]

_MODELS_32GB: List[Dict] = _MODELS_16GB + [
    {
        "id": "llama3.1:70b",
        "name": "Llama 3.1 70B",
        "description": "Near GPT-4 quality — requires 32GB+",
    }
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_platform() -> str:
    """Returns the current OS as 'mac', 'windows', or 'linux'."""
    system = platform.system()
    if system == "Darwin":
        return "mac"
    if system == "Windows":
        return "windows"
    return "linux"


def get_system_ram_gb() -> float:
    """Returns total system RAM in GB."""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_available_ollama_models() -> List[str]:
    """Returns installed model names from the local Ollama API."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def get_recommended_models(ram_gb: Optional[float] = None) -> List[Dict]:
    """Returns ordered model recommendations for the given RAM tier.

    Falls back to live RAM detection if ram_gb is None.
    """
    if ram_gb is None:
        ram_gb = get_system_ram_gb()

    if ram_gb >= 32:
        return _MODELS_32GB
    if ram_gb >= 16:
        return _MODELS_16GB
    return _MODELS_8GB


def is_model_installed(model_id: str) -> bool:
    """Returns True if model_id exactly matches a model installed in Ollama."""
    return model_id in get_available_ollama_models()


def get_pull_command(model_id: str) -> str:
    """Returns the shell command to install a model via Ollama."""
    return f"ollama pull {model_id}"
