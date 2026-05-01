"""
Tests for modules/upload_validator.py

All ChromaDB and config calls are mocked — runs fully offline.
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_file(name: str, content: bytes = b"hello world", size: int = None):
    """Return a minimal file-like object resembling a Streamlit UploadedFile."""
    buf = io.BytesIO(content)
    buf.name = name
    buf.size = size if size is not None else len(content)
    # expose .read() and .seek() via the underlying BytesIO
    buf.seek(0)
    return buf


def _pdf_file(name: str = "doc.pdf", size_bytes: int = 1024):
    return _make_file(name, b"%PDF-1.4 " + b"x" * (size_bytes - 9), size=size_bytes)


# ── validate_file ─────────────────────────────────────────────────────────────

class TestValidateFile:
    def test_valid_pdf(self):
        from modules.upload_validator import validate_file
        assert validate_file(_pdf_file()).valid is True

    def test_valid_txt(self):
        from modules.upload_validator import validate_file
        f = _make_file("brief.txt", b"case notes")
        assert validate_file(f).valid is True

    def test_valid_docx(self):
        from modules.upload_validator import validate_file
        f = _make_file("contract.docx", b"PK\x03\x04" + b"\x00" * 10)
        assert validate_file(f).valid is True

    def test_unsupported_extension(self):
        from modules.upload_validator import validate_file
        f = _make_file("spreadsheet.xlsx", b"data")
        result = validate_file(f)
        assert result.valid is False
        assert "Unsupported" in result.error_message

    def test_unsupported_csv(self):
        from modules.upload_validator import validate_file
        f = _make_file("data.csv", b"a,b,c")
        result = validate_file(f)
        assert result.valid is False

    def test_empty_file(self):
        from modules.upload_validator import validate_file
        f = _make_file("empty.pdf", b"", size=0)
        result = validate_file(f)
        assert result.valid is False
        assert "empty" in result.error_message.lower()

    def test_file_too_large(self):
        from modules.upload_validator import validate_file
        with patch("modules.upload_validator._max_file_size_mb", return_value=1.0):
            f = _make_file("big.pdf", b"%PDF" + b"x" * (2 * 1024 * 1024))
            result = validate_file(f)
        assert result.valid is False
        assert "too large" in result.error_message.lower()

    def test_file_exactly_at_limit_is_valid(self):
        from modules.upload_validator import validate_file
        limit_mb = 50.0
        size_bytes = int(limit_mb * 1024 * 1024)
        content = b"%PDF" + b"x" * (size_bytes - 4)
        f = _make_file("exact.pdf", content)
        with patch("modules.upload_validator._max_file_size_mb", return_value=limit_mb):
            result = validate_file(f)
        assert result.valid is True

    def test_filename_too_long(self):
        from modules.upload_validator import validate_file
        long_name = "a" * 256 + ".pdf"
        f = _make_file(long_name, b"%PDF-data")
        result = validate_file(f)
        assert result.valid is False
        assert "too long" in result.error_message.lower()

    def test_filename_exactly_255_chars_is_valid(self):
        from modules.upload_validator import validate_file
        # 251 'a' chars + ".pdf" = 255
        name = "a" * 251 + ".pdf"
        assert len(name) == 255
        f = _make_file(name, b"%PDF-data")
        result = validate_file(f)
        assert result.valid is True

    def test_path_traversal_dotdot_slash(self):
        from modules.upload_validator import validate_file
        f = _make_file("../etc/passwd.pdf", b"%PDF-data")
        result = validate_file(f)
        assert result.valid is False
        assert "path traversal" in result.error_message.lower()

    def test_path_traversal_backslash(self):
        from modules.upload_validator import validate_file
        f = _make_file("..\\windows\\system32.pdf", b"%PDF-data")
        result = validate_file(f)
        assert result.valid is False

    def test_absolute_path_rejected(self):
        from modules.upload_validator import validate_file
        f = _make_file("/etc/shadow.pdf", b"%PDF-data")
        result = validate_file(f)
        assert result.valid is False

    def test_pdf_without_pdf_header_produces_warning(self):
        from modules.upload_validator import validate_file
        f = _make_file("fake.pdf", b"not a pdf at all")
        result = validate_file(f)
        assert result.valid is True
        assert any("header" in w.lower() for w in result.warnings)

    def test_special_chars_in_stem_produces_warning(self):
        from modules.upload_validator import validate_file
        f = _make_file("doc@#$.pdf", b"%PDF-ok")
        result = validate_file(f)
        assert result.valid is True
        assert any("special" in w.lower() for w in result.warnings)

    def test_normal_filename_no_warnings(self):
        from modules.upload_validator import validate_file
        f = _pdf_file("normal_document.pdf")
        result = validate_file(f)
        assert result.valid is True
        assert result.warnings == []

    def test_size_none_falls_back_to_read(self):
        from modules.upload_validator import validate_file
        f = _make_file("doc.txt", b"content here")
        f.size = None
        result = validate_file(f)
        assert result.valid is True

    def test_corrupt_unreadable_file(self):
        from modules.upload_validator import validate_file
        f = _make_file("bad.pdf", b"some data")
        f.seek = MagicMock(side_effect=OSError("seek failed"))
        result = validate_file(f)
        assert result.valid is False


# ── sanitize_filename ─────────────────────────────────────────────────────────

class TestSanitizeFilename:
    def test_strips_directory_prefix(self):
        from modules.upload_validator import sanitize_filename
        assert sanitize_filename("/home/user/doc.pdf") == "doc.pdf"

    def test_strips_windows_path(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("C:\\Users\\Attorney\\brief.docx")
        assert "\\" not in result
        assert result.endswith(".docx")

    def test_replaces_special_chars_with_underscore(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("doc@#!name.pdf")
        assert "@" not in result
        assert "#" not in result
        assert result.endswith(".pdf")

    def test_preserves_extension(self):
        from modules.upload_validator import sanitize_filename
        assert sanitize_filename("my_file.docx").endswith(".docx")

    def test_collapses_multiple_underscores(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("doc___name.pdf")
        assert "__" not in result

    def test_strips_leading_trailing_underscores_from_stem(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("_doc_.pdf")
        assert not Path(result).stem.startswith("_")
        assert not Path(result).stem.endswith("_")

    def test_empty_stem_becomes_document(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("@@@.pdf")
        assert Path(result).stem == "document"

    def test_normal_filename_unchanged(self):
        from modules.upload_validator import sanitize_filename
        assert sanitize_filename("brief_2024.pdf") == "brief_2024.pdf"

    def test_hyphens_and_underscores_preserved(self):
        from modules.upload_validator import sanitize_filename
        result = sanitize_filename("smith-v-acme_complaint.pdf")
        assert result == "smith-v-acme_complaint.pdf"


# ── detect_duplicates ─────────────────────────────────────────────────────────

class TestDetectDuplicates:
    def _mock_collection(self, found: bool):
        mock_coll = MagicMock()
        if found:
            mock_coll.get.return_value = {
                "ids": ["chunk_001"],
                "metadatas": [{"source": "complaint.pdf", "page": 1}],
            }
        else:
            mock_coll.get.return_value = {"ids": [], "metadatas": []}
        return mock_coll

    def test_duplicate_found(self):
        from modules.upload_validator import detect_duplicates
        mock_coll = self._mock_collection(found=True)
        with patch("modules.upload_validator.detect_duplicates") as _:
            pass  # can't easily patch imports inside function; test via full mock chain

        with (
            patch("modules.case_manager.collection_name_for", return_value="amicus_case1"),
            patch("modules.ingestion.chroma_client") as mock_client,
        ):
            mock_client.get_collection.return_value = self._mock_collection(found=True)
            result = detect_duplicates("complaint.pdf", "case1")

        assert result.is_duplicate is True
        assert result.suggestion != ""

    def test_no_duplicate(self):
        from modules.upload_validator import detect_duplicates
        with (
            patch("modules.case_manager.collection_name_for", return_value="amicus_case1"),
            patch("modules.ingestion.chroma_client") as mock_client,
        ):
            mock_client.get_collection.return_value = self._mock_collection(found=False)
            result = detect_duplicates("new_doc.pdf", "case1")

        assert result.is_duplicate is False

    def test_connection_error_returns_no_duplicate(self):
        from modules.upload_validator import detect_duplicates
        with (
            patch("modules.case_manager.collection_name_for", side_effect=Exception("no db")),
        ):
            result = detect_duplicates("doc.pdf", "case1")
        assert result.is_duplicate is False

    def test_empty_ids_means_no_duplicate(self):
        from modules.upload_validator import detect_duplicates
        mock_coll = MagicMock()
        mock_coll.get.return_value = {"ids": [], "metadatas": []}
        with (
            patch("modules.case_manager.collection_name_for", return_value="amicus_c"),
            patch("modules.ingestion.chroma_client") as mc,
        ):
            mc.get_collection.return_value = mock_coll
            result = detect_duplicates("doc.pdf", "case1")
        assert result.is_duplicate is False


# ── validate_batch ────────────────────────────────────────────────────────────

class TestValidateBatch:
    def test_all_valid(self):
        from modules.upload_validator import validate_batch
        files = [_pdf_file("a.pdf"), _pdf_file("b.pdf")]
        batch = validate_batch(files)
        assert batch.total == 2
        assert batch.valid_count == 2
        assert batch.invalid_count == 0

    def test_mixed_valid_invalid(self):
        from modules.upload_validator import validate_batch
        files = [
            _pdf_file("good.pdf"),
            _make_file("bad.xlsx", b"data"),
            _make_file("empty.pdf", b"", size=0),
        ]
        batch = validate_batch(files)
        assert batch.total == 3
        assert batch.valid_count == 1
        assert batch.invalid_count == 2

    def test_empty_list(self):
        from modules.upload_validator import validate_batch
        batch = validate_batch([])
        assert batch.total == 0
        assert batch.valid_count == 0

    def test_results_list_has_filename(self):
        from modules.upload_validator import validate_batch
        files = [_pdf_file("contract.pdf")]
        batch = validate_batch(files)
        assert batch.results[0]["file"] == "contract.pdf"

    def test_results_contain_validation_result_objects(self):
        from modules.upload_validator import validate_batch, ValidationResult
        files = [_pdf_file("x.pdf")]
        batch = validate_batch(files)
        assert isinstance(batch.results[0]["result"], ValidationResult)

    def test_max_file_size_respected_per_file(self):
        from modules.upload_validator import validate_batch
        big = _make_file("huge.pdf", b"%PDF" + b"x" * (60 * 1024 * 1024))
        with patch("modules.upload_validator._max_file_size_mb", return_value=50.0):
            batch = validate_batch([big])
        assert batch.invalid_count == 1
