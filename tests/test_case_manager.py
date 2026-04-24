"""
Tests for modules/case_manager.py

Uses monkeypatch to redirect DB paths to a temp directory and mocks
chromadb.PersistentClient so no real ChromaDB collections are touched.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cm(tmp_path, monkeypatch):
    """Return case_manager module with paths redirected to tmp_path and chromadb mocked."""
    import modules.case_manager as _cm

    monkeypatch.setattr(_cm, "CASES_DB_PATH", str(tmp_path / "cases.json"))
    monkeypatch.setattr(_cm, "ACTIVE_CASE_PATH", str(tmp_path / "active_case.json"))
    monkeypatch.setattr(_cm, "CHROMA_PATH", str(tmp_path / "chroma"))

    mock_client = MagicMock()
    mock_client.list_collections.return_value = []
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    monkeypatch.setattr(_cm, "chromadb", mock_chromadb)

    return _cm


@pytest.fixture
def cm_with_legacy(tmp_path, monkeypatch):
    """case_manager with a mock chromadb that reports 'legal_docs' collection."""
    import modules.case_manager as _cm

    monkeypatch.setattr(_cm, "CASES_DB_PATH", str(tmp_path / "cases.json"))
    monkeypatch.setattr(_cm, "ACTIVE_CASE_PATH", str(tmp_path / "active_case.json"))
    monkeypatch.setattr(_cm, "CHROMA_PATH", str(tmp_path / "chroma"))

    mock_collection = MagicMock()
    mock_collection.name = "legal_docs"

    mock_client = MagicMock()
    mock_client.list_collections.return_value = [mock_collection]
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    monkeypatch.setattr(_cm, "chromadb", mock_chromadb)

    return _cm


# ── get_active_case / set_active_case ─────────────────────────────────────────

class TestActiveCase:
    def test_returns_none_when_not_set(self, cm):
        assert cm.get_active_case() is None

    def test_set_and_get(self, cm):
        cm.set_active_case("my_case")
        assert cm.get_active_case() == "my_case"

    def test_set_none_clears(self, cm):
        cm.set_active_case("my_case")
        cm.set_active_case(None)
        assert cm.get_active_case() is None

    def test_persists_to_disk(self, cm, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "ACTIVE_CASE_PATH", str(tmp_path / "active_case.json"))
        cm.set_active_case("persist_test")
        raw = json.loads(Path(tmp_path / "active_case.json").read_text())
        assert raw["case_id"] == "persist_test"


# ── create_case ───────────────────────────────────────────────────────────────

class TestCreateCase:
    def test_creates_case(self, cm):
        result = cm.create_case("test_case", "Test Case", "A description")
        assert result["case_id"] == "test_case"
        assert result["display_name"] == "Test Case"
        assert result["description"] == "A description"
        assert "created_at" in result

    def test_first_case_becomes_active(self, cm):
        cm.create_case("first", "First Case")
        assert cm.get_active_case() == "first"

    def test_second_case_does_not_change_active(self, cm):
        cm.create_case("first", "First Case")
        cm.create_case("second", "Second Case")
        assert cm.get_active_case() == "first"

    def test_invalid_id_spaces_rejected(self, cm):
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            cm.create_case("My Case", "My Case")

    def test_invalid_id_uppercase_rejected(self, cm):
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            cm.create_case("MyCase", "My Case")

    def test_invalid_id_leading_underscore_rejected(self, cm):
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            cm.create_case("_bad", "Bad")

    def test_valid_id_with_numbers(self, cm):
        result = cm.create_case("case2025", "Case 2025")
        assert result["case_id"] == "case2025"

    def test_valid_id_with_underscores(self, cm):
        result = cm.create_case("smith_v_acme", "Smith v. Acme")
        assert result["case_id"] == "smith_v_acme"

    def test_duplicate_raises(self, cm):
        cm.create_case("dup", "Dup")
        with pytest.raises(ValueError, match="already exists"):
            cm.create_case("dup", "Dup Again")

    def test_chromadb_collection_created(self, cm):
        cm.create_case("newcase", "New Case")
        cm.chromadb.PersistentClient.return_value.get_or_create_collection.assert_called_with(
            name="amicus_newcase",
            metadata={"hnsw:space": "cosine"},
        )


# ── list_cases ────────────────────────────────────────────────────────────────

class TestListCases:
    def test_empty_when_no_cases(self, cm):
        assert cm.list_cases() == []

    def test_returns_created_cases(self, cm):
        cm.create_case("alpha", "Alpha")
        cm.create_case("beta", "Beta")
        ids = [c["case_id"] for c in cm.list_cases()]
        assert "alpha" in ids
        assert "beta" in ids

    def test_sorted_by_created_at(self, cm):
        cm.create_case("first", "First")
        cm.create_case("second", "Second")
        ids = [c["case_id"] for c in cm.list_cases()]
        assert ids.index("first") < ids.index("second")

    def test_auto_detects_legacy_collection(self, cm_with_legacy):
        cases = cm_with_legacy.list_cases()
        assert any(c["case_id"] == "chen_v_nexagen" for c in cases)

    def test_legacy_auto_sets_active_case(self, cm_with_legacy):
        cm_with_legacy.list_cases()
        assert cm_with_legacy.get_active_case() == "chen_v_nexagen"

    def test_legacy_only_registered_once(self, cm_with_legacy):
        cm_with_legacy.list_cases()
        cm_with_legacy.list_cases()
        cases = cm_with_legacy.list_cases()
        assert sum(1 for c in cases if c["case_id"] == "chen_v_nexagen") == 1


# ── delete_case ───────────────────────────────────────────────────────────────

class TestDeleteCase:
    def test_deletes_case(self, cm):
        cm.create_case("to_delete", "To Delete")
        cm.delete_case("to_delete")
        ids = [c["case_id"] for c in cm.list_cases()]
        assert "to_delete" not in ids

    def test_nonexistent_raises(self, cm):
        with pytest.raises(ValueError, match="not found"):
            cm.delete_case("no_such_case")

    def test_active_case_updated_after_delete(self, cm):
        cm.create_case("keep", "Keep")
        cm.create_case("remove", "Remove")
        cm.set_active_case("remove")
        cm.delete_case("remove")
        assert cm.get_active_case() == "keep"

    def test_active_becomes_none_when_last_case_deleted(self, cm):
        cm.create_case("only", "Only")
        cm.set_active_case("only")
        cm.delete_case("only")
        assert cm.get_active_case() is None

    def test_chromadb_collection_deleted(self, cm):
        cm.create_case("gone", "Gone")
        cm.delete_case("gone")
        cm.chromadb.PersistentClient.return_value.delete_collection.assert_called()


# ── collection_name_for ───────────────────────────────────────────────────────

class TestCollectionNameFor:
    def test_standard_case_gets_amicus_prefix(self, cm):
        assert cm.collection_name_for("smith_v_acme") == "amicus_smith_v_acme"

    def test_chen_uses_legacy_when_legal_docs_present(self, cm_with_legacy):
        name = cm_with_legacy.collection_name_for("chen_v_nexagen")
        assert name == "legal_docs"

    def test_chen_uses_amicus_when_no_legacy(self, cm):
        # mock_client.list_collections returns [] so no legacy collection
        name = cm.collection_name_for("chen_v_nexagen")
        assert name == "amicus_chen_v_nexagen"
