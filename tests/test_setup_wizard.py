"""
Tests for modules/setup_wizard.py

All network and psutil calls are mocked — runs fully offline.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── is_configured ─────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_false_when_config_missing(self, tmp_path):
        from modules.setup_wizard import is_configured
        with patch("modules.setup_wizard._USER_CONFIG_PATH", tmp_path / "user_config.json"):
            assert is_configured() is False

    def test_false_when_config_has_no_model(self, tmp_path):
        p = tmp_path / "user_config.json"
        p.write_text(json.dumps({"last_update_check": "2024-01-01"}))
        with patch("modules.setup_wizard._USER_CONFIG_PATH", p):
            from modules.setup_wizard import is_configured
            assert is_configured() is False

    def test_false_when_model_key_is_empty_string(self, tmp_path):
        p = tmp_path / "user_config.json"
        p.write_text(json.dumps({"primary_model": ""}))
        with patch("modules.setup_wizard._USER_CONFIG_PATH", p):
            from modules.setup_wizard import is_configured
            assert is_configured() is False

    def test_true_when_model_is_configured(self, tmp_path):
        p = tmp_path / "user_config.json"
        p.write_text(json.dumps({"primary_model": "llama3.1:8b"}))
        with patch("modules.setup_wizard._USER_CONFIG_PATH", p):
            from modules.setup_wizard import is_configured
            assert is_configured() is True

    def test_false_on_corrupt_json(self, tmp_path):
        p = tmp_path / "user_config.json"
        p.write_text("not valid json{{")
        with patch("modules.setup_wizard._USER_CONFIG_PATH", p):
            from modules.setup_wizard import is_configured
            assert is_configured() is False


# ── get_safe_default_model ────────────────────────────────────────────────────

class TestGetSafeDefaultModel:
    def _mock_ram(self, gb: float):
        mock_mem = MagicMock()
        mock_mem.total = int(gb * 1024 ** 3)
        return patch("modules.setup_wizard.psutil.virtual_memory", return_value=mock_mem)

    def test_8gb_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(8.0):
            assert get_safe_default_model() == "llama3.1:8b"

    def test_12gb_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(12.0):
            assert get_safe_default_model() == "llama3.1:8b"

    def test_16gb_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(16.0):
            assert get_safe_default_model() == "llama3.1:8b"

    def test_24gb_mac_returns_llama33_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(24.0):
            with patch("modules.setup_wizard.platform.system", return_value="Darwin"):
                assert get_safe_default_model() == "llama3.3:8b"

    def test_24gb_windows_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(24.0):
            with patch("modules.setup_wizard.platform.system", return_value="Windows"):
                assert get_safe_default_model() == "llama3.1:8b"

    def test_32gb_windows_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(32.0):
            with patch("modules.setup_wizard.platform.system", return_value="Windows"):
                assert get_safe_default_model() == "llama3.1:8b"

    def test_64gb_linux_returns_llama31_8b(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(64.0):
            with patch("modules.setup_wizard.platform.system", return_value="Linux"):
                assert get_safe_default_model() == "llama3.1:8b"

    def test_never_returns_70b_automatically(self):
        from modules.setup_wizard import get_safe_default_model
        with self._mock_ram(128.0):
            with patch("modules.setup_wizard.platform.system", return_value="Darwin"):
                result = get_safe_default_model()
        assert "70b" not in result

    def test_psutil_failure_falls_back_safely(self):
        from modules.setup_wizard import get_safe_default_model
        with patch("modules.setup_wizard.psutil.virtual_memory", side_effect=RuntimeError("no psutil")):
            result = get_safe_default_model()
        assert result == "llama3.1:8b"


# ── is_ollama_running ─────────────────────────────────────────────────────────

class TestIsOllamaRunning:
    def test_true_when_api_returns_200(self):
        from modules.setup_wizard import is_ollama_running
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("modules.setup_wizard.requests.get", return_value=mock_resp):
            assert is_ollama_running() is True

    def test_false_when_connection_refused(self):
        from modules.setup_wizard import is_ollama_running
        with patch("modules.setup_wizard.requests.get", side_effect=ConnectionError()):
            assert is_ollama_running() is False

    def test_false_when_timeout(self):
        import requests as _req
        from modules.setup_wizard import is_ollama_running
        with patch("modules.setup_wizard.requests.get", side_effect=_req.Timeout()):
            assert is_ollama_running() is False

    def test_false_when_non_200_status(self):
        from modules.setup_wizard import is_ollama_running
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("modules.setup_wizard.requests.get", return_value=mock_resp):
            assert is_ollama_running() is False


# ── is_model_installed ────────────────────────────────────────────────────────

class TestIsModelInstalled:
    def _mock_tags(self, names: list):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": n} for n in names]}
        return patch("modules.setup_wizard.requests.get", return_value=mock_resp)

    def test_true_when_model_present(self):
        from modules.setup_wizard import is_model_installed
        with self._mock_tags(["llama3.1:8b", "phi4:14b"]):
            assert is_model_installed("llama3.1:8b") is True

    def test_false_when_model_absent(self):
        from modules.setup_wizard import is_model_installed
        with self._mock_tags(["phi4:14b"]):
            assert is_model_installed("llama3.1:8b") is False

    def test_false_on_connection_error(self):
        from modules.setup_wizard import is_model_installed
        with patch("modules.setup_wizard.requests.get", side_effect=ConnectionError()):
            assert is_model_installed("llama3.1:8b") is False

    def test_false_on_empty_list(self):
        from modules.setup_wizard import is_model_installed
        with self._mock_tags([]):
            assert is_model_installed("llama3.1:8b") is False


# ── get_model_file_size_gb ────────────────────────────────────────────────────

class TestGetModelFileSizeGb:
    def test_llama31_8b_size(self):
        from modules.setup_wizard import get_model_file_size_gb
        assert get_model_file_size_gb("llama3.1:8b") == pytest.approx(4.7)

    def test_llama33_8b_size(self):
        from modules.setup_wizard import get_model_file_size_gb
        assert get_model_file_size_gb("llama3.3:8b") == pytest.approx(5.0)

    def test_mistral_nemo_size(self):
        from modules.setup_wizard import get_model_file_size_gb
        assert get_model_file_size_gb("mistral-nemo:12b") == pytest.approx(7.1)

    def test_llama31_70b_size(self):
        from modules.setup_wizard import get_model_file_size_gb
        assert get_model_file_size_gb("llama3.1:70b") == pytest.approx(40.0)

    def test_unknown_model_returns_default(self):
        from modules.setup_wizard import get_model_file_size_gb
        size = get_model_file_size_gb("unknown-model:99b")
        assert size > 0


# ── run_health_check ──────────────────────────────────────────────────────────

class TestRunHealthCheck:
    def _mock_env(self, ram_gb=16.0, os_name="Darwin", ollama_ok=True, model_present=True):
        mock_mem = MagicMock()
        mock_mem.total = int(ram_gb * 1024 ** 3)

        if ollama_ok and model_present:
            tags_resp = MagicMock()
            tags_resp.status_code = 200
            tags_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}, {"name": "llama3.3:8b"}]}
            get_mock = patch("modules.setup_wizard.requests.get", return_value=tags_resp)
        elif ollama_ok and not model_present:
            tags_resp = MagicMock()
            tags_resp.status_code = 200
            tags_resp.json.return_value = {"models": []}
            get_mock = patch("modules.setup_wizard.requests.get", return_value=tags_resp)
        else:
            get_mock = patch("modules.setup_wizard.requests.get", side_effect=ConnectionError())

        return (
            patch("modules.setup_wizard.psutil.virtual_memory", return_value=mock_mem),
            patch("modules.setup_wizard.platform.system", return_value=os_name),
            get_mock,
        )

    def test_ready_to_use_when_all_ok(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(ollama_ok=True, model_present=True)
        with p1, p2, p3:
            status = run_health_check()
        assert status.ready_to_use is True
        assert status.ollama_running is True
        assert status.model_installed is True

    def test_not_ready_when_ollama_down(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(ollama_ok=False)
        with p1, p2, p3:
            status = run_health_check()
        assert status.ready_to_use is False
        assert status.ollama_running is False
        assert status.model_installed is False

    def test_not_ready_when_model_missing(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(ollama_ok=True, model_present=False)
        with p1, p2, p3:
            status = run_health_check()
        assert status.ready_to_use is False
        assert status.ollama_running is True
        assert status.model_installed is False

    def test_platform_detected_as_mac(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(os_name="Darwin")
        with p1, p2, p3:
            status = run_health_check()
        assert status.platform == "mac"

    def test_platform_detected_as_windows(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(os_name="Windows", ollama_ok=False)
        with p1, p2, p3:
            status = run_health_check()
        assert status.platform == "windows"

    def test_ram_gb_populated(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env(ram_gb=16.0)
        with p1, p2, p3:
            status = run_health_check()
        assert abs(status.ram_gb - 16.0) < 0.1

    def test_recommended_model_populated(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env()
        with p1, p2, p3:
            status = run_health_check()
        assert status.recommended_model in ("llama3.1:8b", "llama3.3:8b")

    def test_estimated_download_gb_positive(self):
        from modules.setup_wizard import run_health_check
        p1, p2, p3 = self._mock_env()
        with p1, p2, p3:
            status = run_health_check()
        assert status.estimated_download_gb > 0

    def test_never_raises_on_all_failures(self):
        from modules.setup_wizard import run_health_check
        with patch("modules.setup_wizard.psutil.virtual_memory", side_effect=RuntimeError()):
            with patch("modules.setup_wizard.requests.get", side_effect=ConnectionError()):
                with patch("modules.setup_wizard.platform.system", side_effect=OSError()):
                    status = run_health_check()
        assert isinstance(status.ready_to_use, bool)

    def test_health_status_dataclass_fields(self):
        from modules.setup_wizard import run_health_check, HealthStatus
        p1, p2, p3 = self._mock_env()
        with p1, p2, p3:
            status = run_health_check()
        assert isinstance(status, HealthStatus)
        assert hasattr(status, "ollama_running")
        assert hasattr(status, "recommended_model")
        assert hasattr(status, "model_installed")
        assert hasattr(status, "ram_gb")
        assert hasattr(status, "platform")
        assert hasattr(status, "estimated_download_gb")
        assert hasattr(status, "ready_to_use")
