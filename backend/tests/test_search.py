"""Tests for search.py — dispatch_search and prepare_research_context."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.services.search import dispatch_search, prepare_research_context


# ---------------------------------------------------------------------------
# dispatch_search
# ---------------------------------------------------------------------------


class TestDispatchSearch:
    def test_normal_dict_response(
        self, mock_search_tool: MagicMock, test_config: Any
    ) -> None:
        mock_search_tool.run.return_value = {
            "results": [{"title": "A", "url": "https://a.com", "content": "body"}],
            "backend": "tavily",
            "answer": "direct answer",
            "notices": [],
        }
        payload, notices, answer, backend = dispatch_search("test query", test_config, 0)
        assert len(payload["results"]) == 1
        assert answer == "direct answer"
        assert backend == "tavily"
        assert notices == []

    def test_string_response_becomes_notice(
        self, mock_search_tool: MagicMock, test_config: Any
    ) -> None:
        mock_search_tool.run.return_value = "rate limited"
        payload, notices, answer, backend = dispatch_search("query", test_config, 0)
        assert payload["results"] == []
        assert payload["answer"] is None
        assert "rate limited" in notices
        assert len(notices) == 1

    def test_no_notices_key(self, mock_search_tool: MagicMock, test_config: Any) -> None:
        mock_search_tool.run.return_value = {
            "results": [],
            "backend": "duckduckgo",
        }
        _, notices, _, _ = dispatch_search("query", test_config, 0)
        assert notices == []

    def test_backend_fallback_when_not_in_payload(
        self, mock_search_tool: MagicMock, test_config: Any
    ) -> None:
        mock_search_tool.run.return_value = {"results": []}
        _, _, _, backend = dispatch_search("query", test_config, 0)
        # Falls back to config value's string representation
        assert backend is not None

    def test_exception_propagates(
        self, mock_search_tool: MagicMock, test_config: Any
    ) -> None:
        mock_search_tool.run.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="API down"):
            dispatch_search("query", test_config, 0)

    def test_passes_correct_parameters(
        self, mock_search_tool: MagicMock, test_config: Any
    ) -> None:
        dispatch_search("my query", test_config, 2)
        call_kwargs = mock_search_tool.run.call_args[0][0]
        assert call_kwargs["input"] == "my query"
        assert call_kwargs["backend"] == "duckduckgo"
        assert call_kwargs["max_results"] == 5
        assert call_kwargs["loop_count"] == 2


# ---------------------------------------------------------------------------
# prepare_research_context
# ---------------------------------------------------------------------------


class TestPrepareResearchContext:
    def test_normal(self, sample_search_result: dict[str, Any], test_config: Any) -> None:
        summary, context = prepare_research_context(sample_search_result, None, test_config)
        assert "Source Alpha" in summary
        assert "example.com/alpha" in summary
        assert "Alpha content body" in context

    def test_with_answer_text(self, sample_search_result: dict[str, Any], test_config: Any) -> None:
        _, context = prepare_research_context(sample_search_result, "AI answer", test_config)
        assert context.startswith("AI直接答案：")
        assert "AI answer" in context

    def test_without_answer_text(
        self, sample_search_result: dict[str, Any], test_config: Any
    ) -> None:
        _, context = prepare_research_context(sample_search_result, "", test_config)
        assert not context.startswith("AI直接答案：")

    def test_none_search_result(self, test_config: Any) -> None:
        summary, context = prepare_research_context(None, None, test_config)
        assert summary == ""
        assert context == ""

    def test_empty_results(self, test_config: Any) -> None:
        summary, context = prepare_research_context({"results": []}, None, test_config)
        assert summary == ""
        assert context == ""

    def test_fetch_full_page_flag(
        self, sample_search_result: dict[str, Any], test_config: Any
    ) -> None:
        """When fetch_full_page is True, raw_content should appear in context."""
        test_config.fetch_full_page = True
        _, context = prepare_research_context(sample_search_result, None, test_config)
        assert "详细信息内容限制为" in context
