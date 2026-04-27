# modules/update_checker.py
# Checks whether installed Ollama models have newer versions available.
# Designed to fail gracefully — if offline or the registry is unreachable,
# check_failed is set to True and the UI shows a safe "unable to check" message.

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

_USER_CONFIG_PATH  = Path("db") / "user_config.json"
_TIMEOUT           = 3            # seconds — never block the UI
_OLLAMA_SHOW_URL   = "http://localhost:11434/api/show"
_REGISTRY_BASE_URL = "https://registry.ollama.ai/v2"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class UpdateInfo:
    profile_name:     str
    model_id:         str
    current_digest:   Optional[str] = field(default=None)
    latest_digest:    Optional[str] = field(default=None)
    update_available: bool          = False
    check_failed:     bool          = False


# ── Config persistence ────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        if _USER_CONFIG_PATH.exists():
            return json.loads(_USER_CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_config(cfg: dict) -> None:
    _USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def get_last_update_check() -> Optional[datetime]:
    """Returns when the last update check ran, or None if never checked."""
    ts = _load_config().get("last_update_check")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def set_last_update_check(dt: Optional[datetime] = None) -> None:
    """Persists the last update check timestamp to db/user_config.json."""
    cfg = _load_config()
    cfg["last_update_check"] = (dt or datetime.now()).isoformat()
    _save_config(cfg)


# ── Digest helpers ────────────────────────────────────────────────────────────

def _get_installed_digest(model_id: str) -> Optional[str]:
    """Ask the local Ollama daemon for the digest of an installed model."""
    try:
        resp = requests.post(
            _OLLAMA_SHOW_URL,
            json={"name": model_id},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("details", {}).get("digest") or data.get("digest")
    except Exception:
        return None


def _get_registry_digest(model_id: str) -> Optional[str]:
    """Fetch the latest digest from the Ollama registry for a model."""
    try:
        parts     = model_id.split(":", 1)
        tag       = parts[1] if len(parts) > 1 else "latest"
        name_part = parts[0]
        namespace, name = name_part.split("/", 1) if "/" in name_part else ("library", name_part)
        url = f"{_REGISTRY_BASE_URL}/{namespace}/{name}/manifests/{tag}"
        resp = requests.get(
            url,
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.headers.get("Docker-Content-Digest") or resp.json().get("digest")
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def check_for_model_updates(
    profile_model_ids: Optional[List[tuple]] = None,
) -> List[UpdateInfo]:
    """Check for newer versions of installed profile models.

    Args:
        profile_model_ids: list of (profile_display_name, model_id) tuples.
            When None, derives the list from installed hardware profiles.

    Returns:
        List of UpdateInfo. check_failed=True when a digest could not be
        fetched (offline, timeout, registry unavailable). Never raises.
    """
    if profile_model_ids is None:
        from modules.hardware_detect import _PROFILES, get_available_ollama_models
        installed = get_available_ollama_models()
        profile_model_ids = [
            (p["display_name"], p["model_id"])
            for p in _PROFILES
            if p["model_id"] in installed
        ]

    results: List[UpdateInfo] = []
    for profile_name, model_id in profile_model_ids:
        try:
            current = _get_installed_digest(model_id)
            latest  = _get_registry_digest(model_id)
            failed  = current is None or latest is None
        except Exception:
            current = latest = None
            failed  = True

        results.append(UpdateInfo(
            profile_name=profile_name,
            model_id=model_id,
            current_digest=current,
            latest_digest=latest,
            update_available=(not failed and current != latest),
            check_failed=failed,
        ))

    return results
