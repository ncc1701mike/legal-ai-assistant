"""
Tests for modules/upload_manager.py

All external dependencies (ChromaDB, ingestion, file I/O) are mocked.
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_file(name: str = "doc.pdf", content: bytes = b"%PDF-content"):
    buf = io.BytesIO(content)
    buf.name = name
    buf.size = len(content)
    buf.seek(0)
    return buf


def _valid_validation():
    from modules.upload_validator import ValidationResult
    return ValidationResult(valid=True, warnings=[])


def _invalid_validation(msg: str = "bad file"):
    from modules.upload_validator import ValidationResult
    return ValidationResult(valid=False, error_message=msg)


def _no_duplicate():
    from modules.upload_validator import DuplicateResult
    return DuplicateResult(is_duplicate=False)


def _is_duplicate(filename: str = "doc.pdf"):
    from modules.upload_validator import DuplicateResult
    return DuplicateResult(
        is_duplicate=True,
        existing_doc_info={"source": filename},
        suggestion=f"'{filename}' already exists.",
    )


# ── UploadSession dataclass ───────────────────────────────────────────────────

class TestUploadSession:
    def test_to_dict_round_trips(self):
        from modules.upload_manager import UploadSession
        s = UploadSession(
            session_id="abc",
            case_id="case1",
            files_queued=3,
            files_processed=2,
            files_failed=1,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:01:00Z",
            error_log=[{"file": "x.pdf", "error": "oops"}],
        )
        d = s.to_dict()
        restored = UploadSession.from_dict(d)
        assert restored.session_id == "abc"
        assert restored.files_queued == 3
        assert restored.error_log[0]["file"] == "x.pdf"

    def test_from_dict_tolerates_missing_keys(self):
        from modules.upload_manager import UploadSession
        s = UploadSession.from_dict({})
        assert s.session_id == ""
        assert s.files_queued == 0
        assert s.error_log == []

    def test_default_error_log_is_empty_list(self):
        from modules.upload_manager import UploadSession
        s = UploadSession(session_id="x", case_id="c")
        assert s.error_log == []


# ── process_upload ────────────────────────────────────────────────────────────

class TestProcessUpload:
    def _patch_valid(self, valid=True, duplicate=False):
        vr = _valid_validation() if valid else _invalid_validation("invalid file")
        dr = _is_duplicate() if duplicate else _no_duplicate()
        return (
            patch("modules.upload_manager.validate_file", return_value=vr),
            patch("modules.upload_manager.detect_duplicates", return_value=dr),
        )

    def test_success_path(self, tmp_path):
        from modules.upload_manager import process_upload
        f = _make_file()
        mock_ingest = MagicMock(return_value={"status": "success", "chunks": 7})
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", mock_ingest):
            result = process_upload(f, "case1")
        assert result["success"] is True
        assert result["chunks"] == 7
        assert result["error"] == ""

    def test_validation_failure_returns_error(self):
        from modules.upload_manager import process_upload
        f = _make_file("bad.xlsx")
        pv, pd = self._patch_valid(valid=False, duplicate=False)
        with pv, pd:
            result = process_upload(f, "case1")
        assert result["success"] is False
        assert result["error"] != ""

    def test_duplicate_detected_returns_error(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        pv, pd = self._patch_valid(valid=True, duplicate=True)
        with pv, pd:
            result = process_upload(f, "case1")
        assert result["success"] is False
        assert "uplicate" in result["error"] or "duplicate" in result["error"].lower()

    def test_ingest_exception_returns_user_friendly_error(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", side_effect=RuntimeError("boom")):
            result = process_upload(f, "case1")
        assert result["success"] is False
        assert "IT administrator" in result["error"] or result["error"] != ""

    def test_no_stack_trace_in_error(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", side_effect=ValueError("internal")):
            result = process_upload(f, "case1")
        assert "Traceback" not in result["error"]
        assert "ValueError" not in result["error"]

    def test_ingest_skipped_status_returns_friendly_error(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        mock_ingest = MagicMock(return_value={"status": "skipped — no extractable text (possibly image-based PDF)"})
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", mock_ingest):
            result = process_upload(f, "case1")
        assert result["success"] is False
        assert "OCR" in result["error"] or "scanned" in result["error"].lower()

    def test_on_progress_callback_called(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        stages = []
        def cb(filename, stage, pct):
            stages.append(stage)
        mock_ingest = MagicMock(return_value={"status": "success", "chunks": 3})
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", mock_ingest):
            process_upload(f, "case1", on_progress=cb)
        assert "validating" in stages
        assert "ingesting" in stages
        assert "complete" in stages

    def test_on_progress_called_with_failed_on_validation_error(self):
        from modules.upload_manager import process_upload
        f = _make_file()
        stages = []
        pv, pd = self._patch_valid(valid=False)
        with pv, pd:
            process_upload(f, "case1", on_progress=lambda fn, s, p: stages.append(s))
        assert "failed" in stages

    def test_override_name_used_as_filename(self):
        from modules.upload_manager import process_upload
        f = _make_file("original.pdf")
        mock_ingest = MagicMock(return_value={"status": "success", "chunks": 2})
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, patch("modules.upload_manager.ingest_document", mock_ingest):
            result = process_upload(f, "case1", override_name="renamed_v2.pdf")
        assert result["filename"] == "renamed_v2.pdf"
        mock_ingest.assert_called_once()
        args, kwargs = mock_ingest.call_args
        assert kwargs.get("original_name") == "renamed_v2.pdf" or args[1] == "renamed_v2.pdf"

    def test_tmp_file_cleaned_up_on_success(self):
        from modules.upload_manager import process_upload
        import os
        unlinked = []
        real_unlink = os.unlink

        def tracking_unlink(path):
            unlinked.append(path)
            real_unlink(path)

        f = _make_file()
        mock_ingest = MagicMock(return_value={"status": "success", "chunks": 1})
        pv, pd = self._patch_valid(valid=True, duplicate=False)
        with pv, pd, \
             patch("modules.upload_manager.ingest_document", mock_ingest), \
             patch("modules.upload_manager.os.unlink", side_effect=tracking_unlink):
            process_upload(f, "case1")

        assert len(unlinked) == 1


# ── handle_duplicate ──────────────────────────────────────────────────────────

class TestHandleDuplicate:
    def test_skip_action(self):
        from modules.upload_manager import handle_duplicate
        result = handle_duplicate("doc.pdf", "case1", "skip")
        assert result["action"] == "skip"
        assert result["new_filename"] == "doc.pdf"
        assert result["deleted"] is False

    def test_rename_new_action(self):
        from modules.upload_manager import handle_duplicate
        result = handle_duplicate("brief.pdf", "case1", "rename_new")
        assert result["action"] == "rename_new"
        assert result["new_filename"] != "brief.pdf"
        assert result["new_filename"].startswith("brief_v")
        assert result["new_filename"].endswith(".pdf")

    def test_rename_new_preserves_extension(self):
        from modules.upload_manager import handle_duplicate
        result = handle_duplicate("contract.docx", "case1", "rename_new")
        assert result["new_filename"].endswith(".docx")

    def test_replace_action_deletes_existing(self):
        from modules.upload_manager import handle_duplicate
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": ["chunk_1", "chunk_2"], "metadatas": []}
        with (
            patch("modules.case_manager.collection_name_for", return_value="amicus_c"),
            patch("modules.ingestion.chroma_client") as mc,
        ):
            mc.get_collection.return_value = mock_coll
            result = handle_duplicate("old.pdf", "case1", "replace")
        assert result["action"] == "replace"
        assert result["new_filename"] == "old.pdf"
        mock_coll.delete.assert_called_once()

    def test_replace_action_returns_deleted_true_on_success(self):
        from modules.upload_manager import handle_duplicate
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": ["id1"], "metadatas": []}
        with (
            patch("modules.case_manager.collection_name_for", return_value="amicus_c"),
            patch("modules.ingestion.chroma_client") as mc,
        ):
            mc.get_collection.return_value = mock_coll
            result = handle_duplicate("f.pdf", "c", "replace")
        assert result["deleted"] is True

    def test_replace_chroma_error_still_returns_result(self):
        from modules.upload_manager import handle_duplicate
        with patch("modules.case_manager.collection_name_for", side_effect=Exception("no db")):
            result = handle_duplicate("doc.pdf", "case1", "replace")
        assert result["action"] == "replace"
        assert result["deleted"] is False

    def test_unknown_action_raises(self):
        from modules.upload_manager import handle_duplicate
        with pytest.raises(ValueError, match="Unknown action"):
            handle_duplicate("doc.pdf", "case1", "teleport")


# ── get_upload_history ────────────────────────────────────────────────────────

class TestGetUploadHistory:
    def _write_history(self, tmp_path, sessions: list):
        p = tmp_path / "upload_history.json"
        p.write_text(json.dumps(sessions))
        return p

    def test_returns_sessions_for_case(self, tmp_path):
        from modules.upload_manager import get_upload_history
        data = [
            {"session_id": "s1", "case_id": "case_a", "files_queued": 2,
             "files_processed": 2, "files_failed": 0, "started_at": "", "completed_at": "", "error_log": []},
            {"session_id": "s2", "case_id": "case_b", "files_queued": 1,
             "files_processed": 1, "files_failed": 0, "started_at": "", "completed_at": "", "error_log": []},
        ]
        hist_path = self._write_history(tmp_path, data)
        with patch("modules.upload_manager._HISTORY_PATH", hist_path):
            results = get_upload_history("case_a")
        assert len(results) == 1
        assert results[0].session_id == "s1"

    def test_returns_most_recent_first(self, tmp_path):
        from modules.upload_manager import get_upload_history
        data = [
            {"session_id": "old", "case_id": "c", "files_queued": 1,
             "files_processed": 1, "files_failed": 0, "started_at": "2024-01-01", "completed_at": "", "error_log": []},
            {"session_id": "new", "case_id": "c", "files_queued": 2,
             "files_processed": 2, "files_failed": 0, "started_at": "2024-02-01", "completed_at": "", "error_log": []},
        ]
        hist_path = self._write_history(tmp_path, data)
        with patch("modules.upload_manager._HISTORY_PATH", hist_path):
            results = get_upload_history("c")
        assert results[0].session_id == "new"

    def test_returns_empty_list_when_no_history(self, tmp_path):
        from modules.upload_manager import get_upload_history
        missing = tmp_path / "upload_history.json"
        with patch("modules.upload_manager._HISTORY_PATH", missing):
            results = get_upload_history("any_case")
        assert results == []

    def test_filters_other_cases(self, tmp_path):
        from modules.upload_manager import get_upload_history
        data = [
            {"session_id": f"s{i}", "case_id": f"case_{i}", "files_queued": 1,
             "files_processed": 1, "files_failed": 0, "started_at": "", "completed_at": "", "error_log": []}
            for i in range(5)
        ]
        hist_path = self._write_history(tmp_path, data)
        with patch("modules.upload_manager._HISTORY_PATH", hist_path):
            results = get_upload_history("case_2")
        assert len(results) == 1
        assert results[0].session_id == "s2"

    def test_caps_at_50_results(self, tmp_path):
        from modules.upload_manager import get_upload_history
        data = [
            {"session_id": f"s{i}", "case_id": "big_case", "files_queued": 1,
             "files_processed": 1, "files_failed": 0, "started_at": "", "completed_at": "", "error_log": []}
            for i in range(60)
        ]
        hist_path = self._write_history(tmp_path, data)
        with patch("modules.upload_manager._HISTORY_PATH", hist_path):
            results = get_upload_history("big_case")
        assert len(results) == 50

    def test_returns_upload_session_objects(self, tmp_path):
        from modules.upload_manager import get_upload_history, UploadSession
        data = [{"session_id": "x", "case_id": "c", "files_queued": 1,
                 "files_processed": 1, "files_failed": 0, "started_at": "", "completed_at": "", "error_log": []}]
        hist_path = self._write_history(tmp_path, data)
        with patch("modules.upload_manager._HISTORY_PATH", hist_path):
            results = get_upload_history("c")
        assert isinstance(results[0], UploadSession)
