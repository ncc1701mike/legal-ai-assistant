# modules/feedback.py
# Beta feedback collection — logs user ratings and auto-detected failures
# Syncs daily to Google Drive via sync/sync_feedback.py

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
FEEDBACK_DIR = Path("./feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

# Mode-aware confidence thresholds for auto-flagging
CONFIDENCE_THRESHOLDS = {
    "hybrid":   4.0,   # Basic mode — simple queries should retrieve confidently
    "rerank":   3.5,   # Advanced mode — cross-encoder scoring is more conservative
    "multihop": 3.0,   # Expert mode — multi-hop inference legitimately scores lower
    "vector":   4.0,   # Fallback
}

# Response quality signals for auto-flagging
FAILURE_PHRASES = [
    "i don't know",
    "i cannot find",
    "i was unable to find",
    "no relevant information",
    "not found in the documents",
    "i could not locate",
    "insufficient context",
]

MIN_RESPONSE_LENGTH = 50   # words — below this is considered an empty response
MIN_CHUNKS_REQUIRED = 1    # zero chunks = definite failure


def _ensure_dir():
    """Ensure feedback directory exists."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _get_session_id() -> str:
    """Get or create a session ID stored in a local file."""
    session_file = FEEDBACK_DIR / ".session_id"
    if session_file.exists():
        return session_file.read_text().strip()
    sid = str(uuid.uuid4())[:8]
    _ensure_dir()
    session_file.write_text(sid)
    return sid


def _get_user_id() -> str:
    """Get or create a stable user ID for this installation."""
    uid_file = FEEDBACK_DIR / ".user_id"
    if uid_file.exists():
        return uid_file.read_text().strip()
    uid = str(uuid.uuid4())[:12]
    _ensure_dir()
    uid_file.write_text(uid)
    return uid


def _write_entry(entry: Dict[str, Any]):
    """Append a feedback entry to the JSONL log."""
    _ensure_dir()
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Feedback logged: {entry['type']} — {entry['entry_id']}")


def log_feedback(
    feedback_type: str,                   # "thumbs_up" | "thumbs_down"
    query: str,
    response: str,
    mode: str,
    top_k: int,
    sources: List[Dict],
    chunks_used: int,
    confidence_scores: List[float],
    document_list: List[str],
    comment: Optional[str] = None,
    tab: str = "query",                   # "query" | "summarize" | "redact"
):
    """Log explicit user feedback (thumbs up/down)."""
    entry = {
        "entry_id": str(uuid.uuid4())[:12],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": _get_user_id(),
        "session_id": _get_session_id(),
        "type": feedback_type,
        "tab": tab,
        "mode": mode,
        "top_k": top_k,
        "query": query,
        "response_length_words": len(response.split()),
        "chunks_used": chunks_used,
        "document_count": len(document_list),
        "document_list": document_list,
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else None,
        "min_confidence": round(min(confidence_scores), 2) if confidence_scores else None,
        "max_confidence": round(max(confidence_scores), 2) if confidence_scores else None,
        "comment": comment,
        # Include full response and query only on thumbs_down for privacy
        "query_text": query if feedback_type == "thumbs_down" else None,
        "response_text": response if feedback_type == "thumbs_down" else None,
    }
    _write_entry(entry)


def log_auto_failure(
    failure_reason: str,
    query: str,
    response: str,
    mode: str,
    top_k: int,
    chunks_used: int,
    confidence_scores: List[float],
    document_list: List[str],
    tab: str = "query",
    exception_text: Optional[str] = None,
):
    """Automatically log a detected failure without user action."""
    entry = {
        "entry_id": str(uuid.uuid4())[:12],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": _get_user_id(),
        "session_id": _get_session_id(),
        "type": "auto_fail",
        "tab": tab,
        "mode": mode,
        "top_k": top_k,
        "failure_reason": failure_reason,
        "query_text": query,
        "response_text": response,
        "response_length_words": len(response.split()) if response else 0,
        "chunks_used": chunks_used,
        "document_count": len(document_list),
        "document_list": document_list,
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else None,
        "min_confidence": round(min(confidence_scores), 2) if confidence_scores else None,
        "exception": exception_text,
    }
    _write_entry(entry)
    logger.warning(f"Auto-failure logged: {failure_reason}")


def check_and_log_auto_failures(
    query: str,
    response: str,
    mode: str,
    top_k: int,
    chunks_used: int,
    confidence_scores: List[float],
    document_list: List[str],
    tab: str = "query",
) -> List[str]:
    """
    Check response quality and auto-log any detected failures.
    Returns list of failure reasons found (empty = all good).
    """
    failures = []

    # 1. Empty or near-empty response
    word_count = len(response.split()) if response else 0
    if word_count < MIN_RESPONSE_LENGTH:
        failures.append(f"response_too_short ({word_count} words)")

    # 2. Zero chunks retrieved
    if chunks_used < MIN_CHUNKS_REQUIRED:
        failures.append("no_chunks_retrieved")

    # 3. Failure phrases in response
    response_lower = response.lower() if response else ""
    for phrase in FAILURE_PHRASES:
        if phrase in response_lower:
            failures.append(f"failure_phrase_detected: '{phrase}'")
            break  # one is enough

    # 4. Mode-aware confidence threshold
    if confidence_scores:
        min_conf = min(confidence_scores)
        threshold = CONFIDENCE_THRESHOLDS.get(mode, 3.5)
        if min_conf < threshold:
            failures.append(f"low_confidence_{mode} (min={min_conf:.1f}, threshold={threshold})")

    # Log each failure
    for reason in failures:
        log_auto_failure(
            failure_reason=reason,
            query=query,
            response=response,
            mode=mode,
            top_k=top_k,
            chunks_used=chunks_used,
            confidence_scores=confidence_scores,
            document_list=document_list,
            tab=tab,
        )

    return failures


def log_redaction_feedback(
    feedback_type: str,
    document_name: str,
    categories_selected: List[str],
    redaction_count: int,
    comment: Optional[str] = None,
):
    """Log feedback specifically on redaction quality."""
    entry = {
        "entry_id": str(uuid.uuid4())[:12],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": _get_user_id(),
        "session_id": _get_session_id(),
        "type": feedback_type,
        "tab": "redact",
        "document_name": document_name,
        "categories_selected": categories_selected,
        "redaction_count": redaction_count,
        "comment": comment,
    }
    _write_entry(entry)


def get_feedback_stats() -> Dict[str, Any]:
    """Return basic stats for display in the UI if needed."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "auto_fail": 0}

    counts = {"thumbs_up": 0, "thumbs_down": 0, "auto_fail": 0}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                t = entry.get("type", "")
                if t in counts:
                    counts[t] += 1
            except json.JSONDecodeError:
                continue

    counts["total"] = sum(counts.values())
    return counts
