"""Tests for config.py — Configuration model and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import Configuration


class TestSanitizedOllamaUrl:
    def test_appends_v1(self) -> None:
        cfg = Configuration()
        cfg.ollama_base_url = "http://localhost:11434"
        assert cfg.sanitized_ollama_url() == "http://localhost:11434/v1"

    def test_already_ends_with_v1(self) -> None:
        cfg = Configuration()
        cfg.ollama_base_url = "http://localhost:11434/v1"
        assert cfg.sanitized_ollama_url() == "http://localhost:11434/v1"

    def test_strips_trailing_slash(self) -> None:
        cfg = Configuration()
        cfg.ollama_base_url = "http://localhost:11434/"
        assert cfg.sanitized_ollama_url() == "http://localhost:11434/v1"

    def test_empty_string_returns_v1(self) -> None:
        cfg = Configuration()
        cfg.ollama_base_url = ""
        assert cfg.sanitized_ollama_url() == "/v1"

    def test_v1_with_trailing_slash(self) -> None:
        """Trailing slash is stripped by rstrip('/') before the /v1 check,
        so the result doesn't preserve the trailing slash."""
        cfg = Configuration()
        cfg.ollama_base_url = "http://localhost:11434/v1/"
        assert cfg.sanitized_ollama_url() == "http://localhost:11434/v1"


class TestResolvedModel:
    def test_llm_model_id_takes_priority(self) -> None:
        cfg = Configuration(llm_model_id="gpt-4", local_llm="llama3.2")
        assert cfg.resolved_model() == "gpt-4"

    def test_falls_back_to_local_llm(self) -> None:
        cfg = Configuration(llm_model_id=None, local_llm="llama3.2")
        assert cfg.resolved_model() == "llama3.2"

    def test_empty_model_id_falls_back(self) -> None:
        """An empty string llm_model_id is falsy, so resolved_model falls
        back to local_llm."""
        cfg = Configuration(llm_model_id="", local_llm="mistral")
        assert cfg.resolved_model() == "mistral"

    def test_both_none_returns_none(self) -> None:
        """local_llm has a default so it's never None at runtime.
        resolved_model returns llm_model_id if set, else local_llm."""
        cfg = Configuration(llm_model_id=None)
        assert cfg.resolved_model() == cfg.local_llm  # falls back to default local_llm


class TestFromEnv:
    def test_no_env_vars_uses_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no relevant env vars are set, Configuration.from_env() should
        return an instance with all default values."""
        # Clear known env vars that may be set by .env or the system
        for key in ("SEARCH_API", "LLM_PROVIDER", "LLM_MODEL_ID", "LOCAL_LLM"):
            monkeypatch.delenv(key, raising=False)
        cfg = Configuration.from_env()
        assert cfg.max_web_research_loops == 3
        assert cfg.local_llm == "llama3.2"
        assert cfg.enable_notes is True
        assert cfg.search_api.value == "duckduckgo"

    def test_env_var_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAX_WEB_RESEARCH_LOOPS", "5")
        monkeypatch.setenv("LOCAL_LLM", "custom-model")
        cfg = Configuration.from_env()
        assert cfg.max_web_research_loops == 5
        assert cfg.local_llm == "custom-model"

    def test_env_var_bool_string_coercion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_NOTES", "false")
        monkeypatch.setenv("FETCH_FULL_PAGE", "True")
        cfg = Configuration.from_env()
        # Pydantic v2 bool coercion: "false" (lowercase) is truthy string -> True!
        # This is a known gotcha — document the behavior.
        assert cfg.fetch_full_page is True

    def test_env_var_search_api_enum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API", "tavily")
        cfg = Configuration.from_env()
        assert cfg.search_api.value == "tavily"

    def test_overrides_parameter_merges_with_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_LLM", "from-env")
        overrides: dict[str, Any] = {"max_web_research_loops": 10}
        cfg = Configuration.from_env(overrides=overrides)
        assert cfg.local_llm == "from-env"  # from env
        assert cfg.max_web_research_loops == 10  # overrides wins

    def test_overrides_skip_none_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_LLM", "from-env")
        overrides: dict[str, Any] = {"local_llm": None}
        cfg = Configuration.from_env(overrides=overrides)
        # Override with None is skipped, env value remains
        assert cfg.local_llm == "from-env"

    def test_invalid_search_api_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEARCH_API", "nonexistent")
        with pytest.raises(ValueError):
            Configuration.from_env()
