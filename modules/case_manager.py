# modules/case_manager.py
# Multi-case management — each case gets its own isolated ChromaDB collection.
# Active case persisted in db/active_case.json; metadata in db/cases.json.

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

CHROMA_PATH = "./db/chroma"
CASES_DB_PATH = "./db/cases.json"
ACTIVE_CASE_PATH = "./db/active_case.json"
_LEGACY_COLLECTION = "legal_docs"   # collection name used before multi-case


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _load_cases() -> Dict:
    path = Path(CASES_DB_PATH)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_cases(cases: Dict) -> None:
    Path(CASES_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_DB_PATH, "w") as f:
        json.dump(cases, f, indent=2)


def _auto_register_legacy(cases: Dict) -> bool:
    """Detect the pre-multi-case 'legal_docs' collection and register it as chen_v_nexagen."""
    if "chen_v_nexagen" in cases:
        return False
    try:
        client = _get_chroma_client()
        existing = {c.name for c in client.list_collections()}
        if _LEGACY_COLLECTION in existing:
            cases["chen_v_nexagen"] = {
                "display_name": "Chen v. Nexagen",
                "description": "ADA disability discrimination — Diana Chen vs. Nexagen Solutions",
                "created_at": datetime.now().isoformat(),
            }
            return True
    except Exception:
        pass
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def collection_name_for(case_id: str) -> str:
    """Return the ChromaDB collection name for a case_id.

    Returns 'legal_docs' for chen_v_nexagen when the legacy collection still
    exists and the new 'amicus_chen_v_nexagen' collection hasn't been created,
    so pre-existing data is accessed without a migration step.
    """
    if case_id == "chen_v_nexagen":
        try:
            client = _get_chroma_client()
            existing = {c.name for c in client.list_collections()}
            if _LEGACY_COLLECTION in existing and f"amicus_{case_id}" not in existing:
                return _LEGACY_COLLECTION
        except Exception:
            pass
    return f"amicus_{case_id}"


def list_cases() -> List[Dict]:
    """Return all cases with metadata, auto-detecting the legacy corpus."""
    cases = _load_cases()
    changed = _auto_register_legacy(cases)
    if changed:
        _save_cases(cases)
        if get_active_case() is None:
            set_active_case("chen_v_nexagen")
    return [
        {"case_id": k, **v}
        for k, v in sorted(cases.items(), key=lambda x: x[1].get("created_at", ""))
    ]


def get_active_case() -> Optional[str]:
    """Return the active case_id, or None if none is set."""
    path = Path(ACTIVE_CASE_PATH)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f).get("case_id")
    except Exception:
        return None


def set_active_case(case_id: Optional[str]) -> None:
    """Persist the active case_id to disk."""
    Path(ACTIVE_CASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_CASE_PATH, "w") as f:
        json.dump({"case_id": case_id}, f)


def create_case(case_id: str, display_name: str, description: str = "") -> Dict:
    """Create a new case and its ChromaDB collection.

    case_id must match ^[a-z0-9][a-z0-9_]*$ (lowercase, alphanumeric, underscores).
    The new case becomes the active case if no active case is set.
    Returns the case metadata dict.
    """
    if not re.match(r"^[a-z0-9][a-z0-9_]*$", case_id):
        raise ValueError(
            f"case_id must be lowercase alphanumeric with underscores, got: '{case_id}'"
        )
    cases = _load_cases()
    _auto_register_legacy(cases)
    if case_id in cases:
        raise ValueError(f"Case '{case_id}' already exists")

    client = _get_chroma_client()
    client.get_or_create_collection(
        name=f"amicus_{case_id}",
        metadata={"hnsw:space": "cosine"},
    )

    entry = {
        "display_name": display_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    cases[case_id] = entry
    _save_cases(cases)

    if get_active_case() is None:
        set_active_case(case_id)

    return {"case_id": case_id, **entry}


def delete_case(case_id: str) -> None:
    """Delete a case and permanently remove its ChromaDB collection."""
    cases = _load_cases()
    if case_id not in cases:
        raise ValueError(f"Case '{case_id}' not found")

    client = _get_chroma_client()
    coll_name = collection_name_for(case_id)
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    del cases[case_id]
    _save_cases(cases)

    if get_active_case() == case_id:
        remaining = list(cases.keys())
        set_active_case(remaining[0] if remaining else None)
