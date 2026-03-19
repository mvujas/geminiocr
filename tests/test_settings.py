"""Tests for Settings in geminiocr.config."""

import pytest

from geminiocr.config import Settings, _DEFAULT_SYSTEM_INSTRUCTION


def _clear_env(monkeypatch):
    """Remove all GEMINI_ env vars so tests start clean."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_CACHE_TTL", raising=False)


class TestEnvVarLoading:
    def test_loads_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-pro")
        monkeypatch.setenv("GEMINI_CACHE_TTL", "7200s")

        s = Settings()

        assert s.api_key == "env-key"
        assert s.model == "gemini-pro"
        assert s.cache_ttl == "7200s"

    def test_explicit_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        monkeypatch.setenv("GEMINI_MODEL", "env-model")

        s = Settings(api_key="my-key", model="my-model")

        assert s.api_key == "my-key"
        assert s.model == "my-model"


class TestValidation:
    def test_missing_api_key_raises(self, monkeypatch):
        _clear_env(monkeypatch)

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            Settings()


class TestDefaults:
    def test_default_system_instruction(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        s = Settings()

        assert s.system_instruction == _DEFAULT_SYSTEM_INSTRUCTION

    def test_custom_system_instruction_preserved(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        s = Settings(system_instruction="Custom prompt")

        assert s.system_instruction == "Custom prompt"

    def test_default_values(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        s = Settings()

        assert s.model == "gemini-2.5-flash"
        assert s.cache_ttl == "3600s"
        assert s.max_retries == 3
        assert s.retry_delay == 2.0
        assert s.concurrency == 5
        assert s.response_schema is None
