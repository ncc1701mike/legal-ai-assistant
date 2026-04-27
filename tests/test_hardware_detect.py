"""
Tests for modules/hardware_detect.py

All network/psutil calls are mocked so the suite runs offline and
does not require Ollama to be running.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── RAM detection ─────────────────────────────────────────────────────────────

class TestGetSystemRamGb:
    def test_returns_positive_float(self):
        from modules.hardware_detect import get_system_ram_gb
        ram = get_system_ram_gb()
        assert isinstance(ram, float)
        assert ram > 0

    def test_converts_bytes_to_gb(self):
        from modules.hardware_detect import get_system_ram_gb
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024 ** 3)  # 16 GiB in bytes
        with patch("modules.hardware_detect.psutil.virtual_memory", return_value=mock_mem):
            result = get_system_ram_gb()
        assert abs(result - 16.0) < 0.01

    def test_8gb_system(self):
        from modules.hardware_detect import get_system_ram_gb
        mock_mem = MagicMock()
        mock_mem.total = 8 * (1024 ** 3)
        with patch("modules.hardware_detect.psutil.virtual_memory", return_value=mock_mem):
            assert abs(get_system_ram_gb() - 8.0) < 0.01


# ── Ollama model list ─────────────────────────────────────────────────────────

class TestGetAvailableOllamaModels:
    def test_returns_model_names_on_success(self):
        from modules.hardware_detect import get_available_ollama_models
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.1:8b"}, {"name": "phi4:14b"}]
        }
        with patch("modules.hardware_detect.requests.get", return_value=mock_resp):
            result = get_available_ollama_models()
        assert result == ["llama3.1:8b", "phi4:14b"]

    def test_returns_empty_list_on_connection_error(self):
        from modules.hardware_detect import get_available_ollama_models
        with patch(
            "modules.hardware_detect.requests.get",
            side_effect=ConnectionError("Ollama not running"),
        ):
            assert get_available_ollama_models() == []

    def test_returns_empty_list_on_timeout(self):
        from modules.hardware_detect import get_available_ollama_models
        import requests as _req
        with patch(
            "modules.hardware_detect.requests.get",
            side_effect=_req.Timeout(),
        ):
            assert get_available_ollama_models() == []

    def test_handles_empty_models_list(self):
        from modules.hardware_detect import get_available_ollama_models
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        with patch("modules.hardware_detect.requests.get", return_value=mock_resp):
            assert get_available_ollama_models() == []


# ── Model recommendations by RAM tier ─────────────────────────────────────────

class TestGetRecommendedModels:
    def test_8gb_tier_returns_one_model(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=8.0)
        assert len(models) == 1
        assert models[0]["id"] == "llama3.1:8b"

    def test_7gb_falls_into_8gb_tier(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=7.5)
        assert len(models) == 1

    def test_16gb_tier_returns_three_models(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=16.0)
        ids = [m["id"] for m in models]
        assert "llama3.1:8b" in ids
        assert "llama3.3:8b" in ids
        assert "mistral-nemo:12b" in ids
        assert len(models) == 3

    def test_24gb_falls_into_16gb_tier(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=24.0)
        assert len(models) == 3

    def test_32gb_tier_includes_70b(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=32.0)
        ids = [m["id"] for m in models]
        assert "llama3.1:70b" in ids
        assert len(models) == 4

    def test_64gb_includes_all_models(self):
        from modules.hardware_detect import get_recommended_models
        models = get_recommended_models(ram_gb=64.0)
        assert len(models) == 4

    def test_each_model_has_required_keys(self):
        from modules.hardware_detect import get_recommended_models
        for model in get_recommended_models(ram_gb=64.0):
            assert "id" in model
            assert "name" in model
            assert "description" in model

    def test_higher_tier_is_superset_of_lower(self):
        from modules.hardware_detect import get_recommended_models
        ids_8  = {m["id"] for m in get_recommended_models(ram_gb=8)}
        ids_16 = {m["id"] for m in get_recommended_models(ram_gb=16)}
        ids_32 = {m["id"] for m in get_recommended_models(ram_gb=32)}
        assert ids_8.issubset(ids_16)
        assert ids_16.issubset(ids_32)

    def test_no_ram_arg_calls_live_detection(self):
        from modules.hardware_detect import get_recommended_models
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024 ** 3)
        with patch("modules.hardware_detect.psutil.virtual_memory", return_value=mock_mem):
            models = get_recommended_models()
        assert len(models) == 3


# ── Installed model checking ──────────────────────────────────────────────────

class TestIsModelInstalled:
    def _patch_installed(self, model_list):
        return patch(
            "modules.hardware_detect.get_available_ollama_models",
            return_value=model_list,
        )

    def test_exact_match_returns_true(self):
        from modules.hardware_detect import is_model_installed
        with self._patch_installed(["llama3.1:8b", "phi4:14b"]):
            assert is_model_installed("llama3.1:8b") is True

    def test_missing_model_returns_false(self):
        from modules.hardware_detect import is_model_installed
        with self._patch_installed(["llama3.1:8b"]):
            assert is_model_installed("llama3.3:8b") is False

    def test_different_tag_returns_false(self):
        from modules.hardware_detect import is_model_installed
        with self._patch_installed(["llama3.1:latest"]):
            assert is_model_installed("llama3.1:8b") is False

    def test_empty_installed_list_returns_false(self):
        from modules.hardware_detect import is_model_installed
        with self._patch_installed([]):
            assert is_model_installed("llama3.1:8b") is False

    def test_multiple_models_correct_one_found(self):
        from modules.hardware_detect import is_model_installed
        with self._patch_installed(["phi4:14b", "mistral-nemo:12b", "llama3.3:8b"]):
            assert is_model_installed("mistral-nemo:12b") is True
            assert is_model_installed("llama3.1:8b") is False


# ── Pull command ──────────────────────────────────────────────────────────────

class TestGetPullCommand:
    def test_standard_model(self):
        from modules.hardware_detect import get_pull_command
        assert get_pull_command("llama3.1:8b") == "ollama pull llama3.1:8b"

    def test_large_model(self):
        from modules.hardware_detect import get_pull_command
        assert get_pull_command("llama3.1:70b") == "ollama pull llama3.1:70b"

    def test_mistral_model(self):
        from modules.hardware_detect import get_pull_command
        assert get_pull_command("mistral-nemo:12b") == "ollama pull mistral-nemo:12b"
