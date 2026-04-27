"""
Tests for modules/update_checker.py

All network calls and filesystem I/O are mocked so the suite runs offline
without requiring Ollama or internet connectivity.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open


# ── get_last_update_check / set_last_update_check ────────────────────────────

class TestGetLastUpdateCheck:
    def test_returns_none_when_no_config(self, tmp_path):
        from modules.update_checker import get_last_update_check, _USER_CONFIG_PATH
        with patch("modules.update_checker._USER_CONFIG_PATH", tmp_path / "user_config.json"):
            result = get_last_update_check()
        assert result is None

    def test_returns_none_when_key_missing(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        cfg_path.write_text(json.dumps({"primary_model": "llama3.1:8b"}))
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import get_last_update_check
            assert get_last_update_check() is None

    def test_returns_datetime_when_present(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        ts = "2024-01-15T10:30:00"
        cfg_path.write_text(json.dumps({"last_update_check": ts}))
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import get_last_update_check
            result = get_last_update_check()
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1

    def test_returns_none_on_corrupt_timestamp(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        cfg_path.write_text(json.dumps({"last_update_check": "not-a-date"}))
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import get_last_update_check
            assert get_last_update_check() is None


class TestSetLastUpdateCheck:
    def test_writes_isoformat_timestamp(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import set_last_update_check, get_last_update_check
            dt = datetime(2024, 6, 1, 12, 0, 0)
            set_last_update_check(dt)
            result = get_last_update_check()
        assert result is not None
        assert result.year == 2024

    def test_defaults_to_now_when_no_arg(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        before = datetime.now()
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import set_last_update_check, get_last_update_check
            set_last_update_check()
            result = get_last_update_check()
        after = datetime.now()
        assert before <= result <= after

    def test_preserves_existing_config_keys(self, tmp_path):
        cfg_path = tmp_path / "user_config.json"
        cfg_path.write_text(json.dumps({"primary_model": "llama3.1:8b"}))
        with patch("modules.update_checker._USER_CONFIG_PATH", cfg_path):
            from modules.update_checker import set_last_update_check
            set_last_update_check(datetime(2024, 1, 1))
            saved = json.loads(cfg_path.read_text())
        assert saved["primary_model"] == "llama3.1:8b"
        assert "last_update_check" in saved


# ── _get_installed_digest ─────────────────────────────────────────────────────

class TestGetInstalledDigest:
    def test_returns_digest_from_details(self):
        from modules.update_checker import _get_installed_digest
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"details": {"digest": "sha256:abc123"}}
        with patch("modules.update_checker.requests.post", return_value=mock_resp):
            result = _get_installed_digest("llama3.1:8b")
        assert result == "sha256:abc123"

    def test_falls_back_to_top_level_digest(self):
        from modules.update_checker import _get_installed_digest
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"digest": "sha256:def456", "details": {}}
        with patch("modules.update_checker.requests.post", return_value=mock_resp):
            result = _get_installed_digest("llama3.1:8b")
        assert result == "sha256:def456"

    def test_returns_none_on_connection_error(self):
        from modules.update_checker import _get_installed_digest
        with patch("modules.update_checker.requests.post", side_effect=ConnectionError()):
            assert _get_installed_digest("llama3.1:8b") is None

    def test_returns_none_on_timeout(self):
        import requests as _req
        from modules.update_checker import _get_installed_digest
        with patch("modules.update_checker.requests.post", side_effect=_req.Timeout()):
            assert _get_installed_digest("llama3.1:8b") is None

    def test_returns_none_on_http_error(self):
        import requests as _req
        from modules.update_checker import _get_installed_digest
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _req.HTTPError("404")
        with patch("modules.update_checker.requests.post", return_value=mock_resp):
            assert _get_installed_digest("llama3.1:8b") is None


# ── _get_registry_digest ──────────────────────────────────────────────────────

class TestGetRegistryDigest:
    def test_returns_docker_content_digest_header(self):
        from modules.update_checker import _get_registry_digest
        mock_resp = MagicMock()
        mock_resp.headers = {"Docker-Content-Digest": "sha256:registry999"}
        mock_resp.json.return_value = {}
        with patch("modules.update_checker.requests.get", return_value=mock_resp):
            result = _get_registry_digest("llama3.1:8b")
        assert result == "sha256:registry999"

    def test_falls_back_to_json_digest(self):
        from modules.update_checker import _get_registry_digest
        mock_resp = MagicMock()
        mock_resp.headers = {}
        mock_resp.json.return_value = {"digest": "sha256:jsondigest"}
        with patch("modules.update_checker.requests.get", return_value=mock_resp):
            result = _get_registry_digest("llama3.1:8b")
        assert result == "sha256:jsondigest"

    def test_namespaced_model_parsed_correctly(self):
        from modules.update_checker import _get_registry_digest
        mock_resp = MagicMock()
        mock_resp.headers = {"Docker-Content-Digest": "sha256:xyz"}
        mock_resp.json.return_value = {}
        with patch("modules.update_checker.requests.get", return_value=mock_resp) as mock_get:
            _get_registry_digest("myorg/mymodel:latest")
            call_url = mock_get.call_args[0][0]
        assert "myorg/mymodel" in call_url
        assert "latest" in call_url

    def test_bare_model_uses_library_namespace(self):
        from modules.update_checker import _get_registry_digest
        mock_resp = MagicMock()
        mock_resp.headers = {"Docker-Content-Digest": "sha256:xyz"}
        mock_resp.json.return_value = {}
        with patch("modules.update_checker.requests.get", return_value=mock_resp) as mock_get:
            _get_registry_digest("llama3.1:8b")
            call_url = mock_get.call_args[0][0]
        assert "library/llama3.1" in call_url

    def test_returns_none_on_connection_error(self):
        from modules.update_checker import _get_registry_digest
        with patch("modules.update_checker.requests.get", side_effect=ConnectionError()):
            assert _get_registry_digest("llama3.1:8b") is None

    def test_returns_none_on_timeout(self):
        import requests as _req
        from modules.update_checker import _get_registry_digest
        with patch("modules.update_checker.requests.get", side_effect=_req.Timeout()):
            assert _get_registry_digest("llama3.1:8b") is None


# ── check_for_model_updates ───────────────────────────────────────────────────

class TestCheckForModelUpdates:
    def _patch_digests(self, installed_digest, registry_digest):
        return (
            patch("modules.update_checker._get_installed_digest", return_value=installed_digest),
            patch("modules.update_checker._get_registry_digest", return_value=registry_digest),
        )

    def test_up_to_date_model(self):
        from modules.update_checker import check_for_model_updates
        digest = "sha256:same"
        p1, p2 = self._patch_digests(digest, digest)
        with p1, p2:
            results = check_for_model_updates([("Standard", "llama3.1:8b")])
        assert len(results) == 1
        assert results[0].update_available is False
        assert results[0].check_failed is False

    def test_update_available_when_digests_differ(self):
        from modules.update_checker import check_for_model_updates
        p1, p2 = self._patch_digests("sha256:old", "sha256:new")
        with p1, p2:
            results = check_for_model_updates([("Standard", "llama3.1:8b")])
        assert results[0].update_available is True
        assert results[0].check_failed is False

    def test_check_failed_when_installed_digest_none(self):
        from modules.update_checker import check_for_model_updates
        p1, p2 = self._patch_digests(None, "sha256:new")
        with p1, p2:
            results = check_for_model_updates([("Standard", "llama3.1:8b")])
        assert results[0].check_failed is True
        assert results[0].update_available is False

    def test_check_failed_when_registry_digest_none(self):
        from modules.update_checker import check_for_model_updates
        p1, p2 = self._patch_digests("sha256:old", None)
        with p1, p2:
            results = check_for_model_updates([("Standard", "llama3.1:8b")])
        assert results[0].check_failed is True

    def test_multiple_models_checked_independently(self):
        from modules.update_checker import check_for_model_updates
        with patch("modules.update_checker._get_installed_digest") as mock_ins, \
             patch("modules.update_checker._get_registry_digest") as mock_reg:
            mock_ins.side_effect = ["sha256:same", "sha256:old"]
            mock_reg.side_effect = ["sha256:same", "sha256:new"]
            results = check_for_model_updates([
                ("Standard", "llama3.1:8b"),
                ("Enhanced", "llama3.3:8b"),
            ])
        assert len(results) == 2
        assert results[0].update_available is False
        assert results[1].update_available is True

    def test_result_fields_populated(self):
        from modules.update_checker import check_for_model_updates
        p1, p2 = self._patch_digests("sha256:old", "sha256:new")
        with p1, p2:
            results = check_for_model_updates([("Standard", "llama3.1:8b")])
        r = results[0]
        assert r.profile_name == "Standard"
        assert r.model_id == "llama3.1:8b"
        assert r.current_digest == "sha256:old"
        assert r.latest_digest == "sha256:new"

    def test_never_raises_on_exception(self):
        from modules.update_checker import check_for_model_updates
        with patch("modules.update_checker._get_installed_digest", side_effect=RuntimeError("boom")):
            with patch("modules.update_checker._get_registry_digest", return_value="sha256:x"):
                results = check_for_model_updates([("Standard", "llama3.1:8b")])
        assert results[0].check_failed is True
