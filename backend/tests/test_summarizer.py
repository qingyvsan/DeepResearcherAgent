"""Tests for SummarizationService — synchronous and streaming modes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import SummaryState, TodoItem
from src.services.summarizer import SummarizationService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def summarizer(mock_agent_factory: Any, test_config: Any) -> SummarizationService:
    return SummarizationService(mock_agent_factory, test_config)


@pytest.fixture
def state() -> SummaryState:
    return SummaryState(research_topic="量子计算")


@pytest.fixture
def task() -> TodoItem:
    return TodoItem(id=1, title="背景调研", intent="收集背景", query="量子计算 2025")


# ---------------------------------------------------------------------------
# summarize_task
# ---------------------------------------------------------------------------


class TestSummarizeTask:
    def test_normal_response(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "## 任务总结\n量子计算在2025年取得突破。"
        result = summarizer.summarize_task(state, task, "some context")
        assert "量子计算" in result
        assert mock_agent.clear_history.called

    def test_empty_response_falls_back(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = ""
        result = summarizer.summarize_task(state, task, "ctx")
        assert result == "暂无可用信息"

    def test_whitespace_only_falls_back(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "   \n  "
        result = summarizer.summarize_task(state, task, "ctx")
        assert result == "暂无可用信息"

    def test_strips_tool_calls(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "内容[TOOL_CALL:note:{}]结尾"
        result = summarizer.summarize_task(state, task, "ctx")
        assert "内容结尾" == result
        assert "[TOOL_CALL" not in result

    def test_strips_think_tokens(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "<think>思考</think>可见内容"
        # strip_thinking_tokens is True by default in test_config
        result = summarizer.summarize_task(state, task, "ctx")
        assert result == "可见内容"

    def test_builds_prompt_with_all_fields(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "summary"
        summarizer.summarize_task(state, task, "research context here")
        call_args = mock_agent.run.call_args[0][0]
        assert "量子计算" in call_args       # research_topic
        assert "背景调研" in call_args       # task title
        assert "收集背景" in call_args       # task intent
        assert "量子计算 2025" in call_args  # task query
        assert "research context here" in call_args

    def test_agent_exception_propagates(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.run.side_effect = RuntimeError("LLM failed")
        with pytest.raises(RuntimeError):
            summarizer.summarize_task(state, task, "ctx")


# ---------------------------------------------------------------------------
# stream_task_summary
# ---------------------------------------------------------------------------


class TestStreamTaskSummary:
    def test_streams_chunks(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        mock_agent.stream_run.return_value = iter(["Hello", " ", "World"])
        stream, get_summary = summarizer.stream_task_summary(state, task, "ctx")
        result = "".join(stream)
        assert result == "Hello World"
        assert get_summary() == "Hello World"

    def test_strips_think_tokens_in_stream(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        """When strip_thinking_tokens is True (default in test_config),
        <think> blocks should be removed from the streamed output."""
        mock_agent.stream_run.return_value = iter([
            "可见", "<think>隐藏", "内容</think>", "结尾"
        ])
        stream, get_summary = summarizer.stream_task_summary(state, task, "ctx")
        streamed = "".join(stream)
        assert streamed == "可见结尾"
        assert get_summary() == "可见结尾"

    def test_without_think_token_stripping(self, test_config: Any, mock_agent_factory: Any, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        test_config.strip_thinking_tokens = False
        svc = SummarizationService(mock_agent_factory, test_config)
        mock_agent.stream_run.return_value = iter(["a", "<think>", "b", "</think>", "c"])
        stream, get_summary = svc.stream_task_summary(state, task, "ctx")
        streamed = "".join(stream)
        assert streamed == "a<think>b</think>c"
        assert get_summary() == "a<think>b</think>c"

    def test_unclosed_think_not_emitted(self, summarizer: SummarizationService, state: SummaryState, task: TodoItem, mock_agent: MagicMock) -> None:
        """Content inside unclosed <think> should not appear in visible output."""
        mock_agent.stream_run.return_value = iter([
            "前置", "<think>未关闭内容", "还在隐藏"
        ])
        stream, get_summary = summarizer.stream_task_summary(state, task, "ctx")
        streamed = "".join(stream)
        assert streamed == "前置"
        # get_summary after stream ends should strip unclosed think
        assert get_summary() == "前置"

    def test_get_summary_after_stream(
        self, summarizer: SummarizationService, state: SummaryState,
        task: TodoItem, mock_agent: MagicMock
    ) -> None:
        mock_agent.stream_run.return_value = iter(["最终", "总结"])
        stream, get_summary = summarizer.stream_task_summary(state, task, "ctx")
        # consume the stream
        for _ in stream:
            pass
        assert get_summary() == "最终总结"

    def test_get_summary_strips_tool_calls(
        self, summarizer: SummarizationService, state: SummaryState,
        task: TodoItem, mock_agent: MagicMock
    ) -> None:
        mock_agent.stream_run.return_value = iter([
            "内容", "[TOOL_CALL:note:{}]", "结尾"
        ])
        stream, get_summary = summarizer.stream_task_summary(state, task, "ctx")
        for _ in stream:
            pass
        assert get_summary() == "内容结尾"

    def test_agent_exception_still_calls_clear_history(
        self, summarizer: SummarizationService, state: SummaryState,
        task: TodoItem, mock_agent: MagicMock
    ) -> None:
        mock_agent.stream_run.side_effect = RuntimeError("stream died")
        stream, _ = summarizer.stream_task_summary(state, task, "ctx")
        with pytest.raises(RuntimeError):
            for _ in stream:
                pass
        assert mock_agent.clear_history.called
