"""
modules/upload_validator.py
File validation and duplicate detection for the upload pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

SUPPORTED_FORMATS: List[str] = [".pdf", ".txt", ".docx", ".doc", ".rtf"]
_DEFAULT_MAX_MB: float = 50.0
_CONFIG_PATH = Path("db/user_config.json")

_PATH_TRAVERSAL_RE = re.compile(r"(\.\.[/\\]|^[/\\])")
_UNSAFE_CHARS_RE = re.compile(r"[^\w\s\-.]")


def _max_file_size_mb() -> float:
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        return float(cfg.get("max_file_size_mb", _DEFAULT_MAX_MB))
    except Exception:
        return _DEFAULT_MAX_MB


@dataclass
class ValidationResult:
    valid: bool
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class DuplicateResult:
    is_duplicate: bool
    existing_doc_info: Optional[dict] = None
    suggestion: str = ""


@dataclass
class BatchValidationResult:
    results: List[dict]
    total: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0


def validate_file(file: Any) -> ValidationResult:
    """
    Validate a file object (e.g. Streamlit UploadedFile).
    Expects .name (str), .size (int, optional), and seekable .read().
    """
    warnings: List[str] = []
    name: str = getattr(file, "name", "")

    if len(name) > 255:
        return ValidationResult(False, f"Filename too long ({len(name)} chars; max 255).")

    if _PATH_TRAVERSAL_RE.search(name):
        return ValidationResult(False, "Filename contains path traversal characters.")

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        return ValidationResult(
            False,
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}.",
        )

    size_bytes = getattr(file, "size", None)
    if size_bytes is None:
        try:
            data = file.read()
            file.seek(0)
            size_bytes = len(data)
        except Exception:
            size_bytes = 0

    if size_bytes == 0:
        return ValidationResult(False, "File is empty.")

    max_mb = _max_file_size_mb()
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_mb:
        return ValidationResult(
            False,
            f"File is too large ({size_mb:.1f} MB; limit {max_mb:.0f} MB).",
        )

    try:
        file.seek(0)
        header = file.read(512)
        file.seek(0)
    except Exception as exc:
        return ValidationResult(False, f"Could not read file: {exc}")

    if len(header) == 0:
        return ValidationResult(False, "File appears to be corrupt (no readable content).")

    if suffix == ".pdf" and not header.startswith(b"%PDF"):
        warnings.append(
            "File does not start with a PDF header — it may be corrupt or misnamed."
        )

    stem = Path(name).stem
    if _UNSAFE_CHARS_RE.search(stem):
        warnings.append(
            "Filename contains special characters and will be sanitized before storage."
        )

    return ValidationResult(True, warnings=warnings)


def detect_duplicates(filename: str, case_id: str) -> DuplicateResult:
    """Check ChromaDB for an existing document with the same filename in this case."""
    try:
        from modules.case_manager import collection_name_for
        from modules.ingestion import chroma_client

        coll = chroma_client.get_collection(collection_name_for(case_id))
        results = coll.get(where={"source": filename}, limit=1, include=["metadatas"])
        if results and results.get("ids"):
            meta = (results.get("metadatas") or [{}])[0]
            return DuplicateResult(
                is_duplicate=True,
                existing_doc_info=meta,
                suggestion=f"'{filename}' already exists in this case.",
            )
    except Exception:
        pass
    return DuplicateResult(is_duplicate=False)


def sanitize_filename(filename: str) -> str:
    """Strip path components, replace special characters with underscores, preserve extension."""
    p = Path(filename).name  # drop any directory part
    stem = Path(p).stem
    suffix = Path(p).suffix

    stem = re.sub(r"[^\w\s\-]", "_", stem)
    stem = re.sub(r"[\s_]+", "_", stem).strip("_")
    if not stem:
        stem = "document"

    return stem + suffix


def validate_batch(files: Any) -> BatchValidationResult:
    """Run validate_file on each file and return a BatchValidationResult."""
    items: List[dict] = []
    valid_count = 0
    invalid_count = 0

    for f in files:
        result = validate_file(f)
        items.append({"file": f.name, "result": result})
        if result.valid:
            valid_count += 1
        else:
            invalid_count += 1

    return BatchValidationResult(
        results=items,
        total=len(items),
        valid_count=valid_count,
        invalid_count=invalid_count,
    )
