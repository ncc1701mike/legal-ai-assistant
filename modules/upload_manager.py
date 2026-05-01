"""
modules/upload_manager.py
Upload session management, per-file processing, and duplicate handling.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from modules.upload_validator import detect_duplicates, sanitize_filename, validate_file
from modules.ingestion import ingest_document

_HISTORY_PATH = Path("db/upload_history.json")


@dataclass
class UploadSession:
    session_id: str
    case_id: str
    files_queued: int = 0
    files_processed: int = 0
    files_failed: int = 0
    started_at: str = ""
    completed_at: str = ""
    error_log: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "case_id": self.case_id,
            "files_queued": self.files_queued,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_log": self.error_log,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UploadSession":
        return cls(
            session_id=d.get("session_id", ""),
            case_id=d.get("case_id", ""),
            files_queued=d.get("files_queued", 0),
            files_processed=d.get("files_processed", 0),
            files_failed=d.get("files_failed", 0),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            error_log=d.get("error_log", []),
        )


def _load_history() -> List[dict]:
    try:
        return json.loads(_HISTORY_PATH.read_text())
    except Exception:
        return []


def _persist_session(session: UploadSession) -> None:
    history = _load_history()
    history.append(session.to_dict())
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(json.dumps(history, indent=2))


def process_upload(
    file: Any,
    case_id: str,
    on_progress: Optional[Callable[[str, str, int], None]] = None,
    override_name: Optional[str] = None,
) -> dict:
    """
    Validate, check duplicates, and ingest a single file.

    on_progress(filename, stage, percent) — called at each stage.
    Stages: "validating", "checking_duplicates", "ingesting", "complete", "failed"

    override_name — use this filename instead of file.name (for rename_new action).

    Returns dict: {success, filename, chunks, error, warnings}
    """
    display_name = override_name or file.name
    _prog = on_progress or (lambda *a: None)

    _prog(display_name, "validating", 10)
    vr = validate_file(file)
    if not vr.valid:
        _prog(display_name, "failed", 100)
        return {
            "success": False,
            "filename": display_name,
            "chunks": 0,
            "error": vr.error_message,
            "warnings": [],
        }

    _prog(display_name, "checking_duplicates", 25)
    dup = detect_duplicates(display_name, case_id)
    if dup.is_duplicate:
        _prog(display_name, "failed", 100)
        return {
            "success": False,
            "filename": display_name,
            "chunks": 0,
            "error": (
                f"Duplicate detected: {dup.suggestion} "
                "Choose Skip, Replace, or Upload as new version."
            ),
            "warnings": vr.warnings,
        }

    _prog(display_name, "ingesting", 50)
    tmp_path: Optional[str] = None
    try:
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.seek(0)
            tmp.write(file.read())
            tmp_path = tmp.name

        result = ingest_document(tmp_path, original_name=display_name, case_id=case_id)  # noqa: E501
    except Exception:
        _prog(display_name, "failed", 100)
        return {
            "success": False,
            "filename": display_name,
            "chunks": 0,
            "error": "An error occurred while processing this file. Contact your IT administrator if this persists.",
            "warnings": vr.warnings,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if result.get("status") == "success":
        _prog(display_name, "complete", 100)
        return {
            "success": True,
            "filename": display_name,
            "chunks": result.get("chunks", 0),
            "error": "",
            "warnings": vr.warnings,
        }

    _prog(display_name, "failed", 100)
    raw = result.get("status", "")
    user_msg = (
        "No readable text was found in this file. "
        "If it is a scanned PDF, it may need to be OCR-processed first."
        if "image" in raw.lower() or "no extractable" in raw.lower()
        else "The file could not be added to the document store. Contact your IT administrator."
    )
    return {
        "success": False,
        "filename": display_name,
        "chunks": 0,
        "error": user_msg,
        "warnings": vr.warnings,
    }


def handle_duplicate(filename: str, case_id: str, action: str) -> dict:
    """
    Prepare a duplicate document for re-upload.

    action:
      "skip"       — keep existing, do not re-upload.
      "replace"    — delete existing chunks, caller re-ingests under same name.
      "rename_new" — return a timestamped new filename; caller re-ingests under new name.

    Returns dict: {action, new_filename, deleted}
    """
    if action == "skip":
        return {"action": "skip", "new_filename": filename, "deleted": False}

    if action == "replace":
        deleted = False
        try:
            from modules.case_manager import collection_name_for
            from modules.ingestion import chroma_client

            coll = chroma_client.get_collection(collection_name_for(case_id))
            existing = coll.get(where={"source": filename}, include=["metadatas"])
            if existing and existing.get("ids"):
                coll.delete(ids=existing["ids"])
                deleted = True
        except Exception:
            pass
        return {"action": "replace", "new_filename": filename, "deleted": deleted}

    if action == "rename_new":
        p = Path(filename)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        new_name = f"{p.stem}_v{ts}{p.suffix}"
        return {"action": "rename_new", "new_filename": new_name, "deleted": False}

    raise ValueError(f"Unknown action {action!r}. Use 'skip', 'replace', or 'rename_new'.")


def get_upload_history(case_id: str) -> List[UploadSession]:
    """Return completed upload sessions for this case, most recent first (max 50)."""
    return [
        UploadSession.from_dict(d)
        for d in reversed(_load_history())
        if d.get("case_id") == case_id
    ][:50]
