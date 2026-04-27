# modules/hardware_detect.py
# Hardware detection and Ollama model recommendation utilities

import platform
import psutil
import requests
from typing import Dict, List, Optional

# ── Attorney-facing hardware profiles ─────────────────────────────────────────
# Ordered from least to most capable; get_auto_recommended_profile() picks the
# last one that both fits in RAM and has its model installed.
_PROFILES: List[Dict] = [
    {
        "profile_id":     "standard",
        "display_name":   "Standard",
        "description":    "Reliable everyday performance",
        "recommended_for": "MacBook Air, Windows PC (8GB RAM)",
        "model_id":       "llama3.1:8b",
        "min_ram_gb":     8,
    },
    {
        "profile_id":     "enhanced",
        "display_name":   "Enhanced",
        "description":    "Faster, more thorough analysis",
        "recommended_for": "MacBook Pro (16GB RAM), Windows PC (16GB RAM)",
        "model_id":       "llama3.3:8b",
        "min_ram_gb":     16,
    },
    {
        "profile_id":     "professional",
        "display_name":   "Professional",
        "description":    "Deepest analysis for complex litigation",
        "recommended_for": "MacBook Pro (32GB RAM), High-spec Windows PC",
        "model_id":       "mistral-nemo:12b",
        "min_ram_gb":     16,
    },
    {
        "profile_id":     "enterprise",
        "display_name":   "Enterprise",
        "description":    "Maximum capability for demanding cases",
        "recommended_for": "Mac Studio, High-memory workstation (32GB+)",
        "model_id":       "llama3.1:70b",
        "min_ram_gb":     32,
    },
]

# ── Technical model catalogue per RAM tier ────────────────────────────────────
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


# ── Attorney-facing profile API ───────────────────────────────────────────────

def get_user_friendly_configs(ram_gb: Optional[float] = None) -> List[Dict]:
    """Returns hardware profiles whose min_ram_gb fits this system's RAM.

    Always includes at least Standard (8 GB) so the UI always has options.
    Falls back to live RAM detection when ram_gb is None.
    """
    if ram_gb is None:
        ram_gb = get_system_ram_gb()
    return [p for p in _PROFILES if ram_gb >= p["min_ram_gb"]]


def get_current_profile(model_id: Optional[str] = None) -> Optional[Dict]:
    """Returns the profile dict matching the active model_id, or None."""
    if model_id is None:
        from modules.llm import get_primary_model
        model_id = get_primary_model()
    return next((p for p in _PROFILES if p["model_id"] == model_id), None)


def get_auto_recommended_profile(
    ram_gb: Optional[float] = None,
    installed: Optional[List[str]] = None,
) -> Dict:
    """Returns the best profile this hardware can run with an installed model.

    'Best' = the last (most capable) profile in _PROFILES that both:
    1. fits within available RAM  2. has its model_id installed in Ollama.

    Falls back to the Standard profile dict when nothing better qualifies.
    """
    if ram_gb is None:
        ram_gb = get_system_ram_gb()
    if installed is None:
        installed = get_available_ollama_models()
    viable = [
        p for p in _PROFILES
        if ram_gb >= p["min_ram_gb"] and p["model_id"] in installed
    ]
    return viable[-1] if viable else _PROFILES[0]
