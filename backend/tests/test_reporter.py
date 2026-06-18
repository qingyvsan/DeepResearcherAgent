"""Tests for ReportingService — final report generation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import SummaryState, TodoItem
from src.services.reporter import ReportingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter(mock_agent: MagicMock, test_config: Any) -> ReportingService:
    return ReportingService(mock_agent, test_config)


@pytest.fixture
def state() -> SummaryState:
    st = SummaryState(research_topic="深度学习")
    st.todo_items = [
        TodoItem(id=1, title="背景", intent="收集背景", query="DL history",
                 status="completed", summary="## 背景信息", sources_summary="* S1 : url1"),
        TodoItem(id=2, title="应用", intent="分析应用", query="DL applications",
                 status="completed", summary="## 应用场景", sources_summary="* S2 : url2"),
    ]
    return st


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_normal_report(self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock) -> None:
        mock_agent.run.return_value = "## 报告内容\n深度学习取得重大进展。"
        report = reporter.generate_report(state)
        assert "深度学习" in report
        assert mock_agent.clear_history.called

    def test_contains_all_task_data_in_prompt(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = "report"
        reporter.generate_report(state)
        prompt = mock_agent.run.call_args[0][0]
        assert "深度学习" in prompt      # research topic
        assert "背景" in prompt          # task title
        assert "应用" in prompt          # task title
        assert "DL history" in prompt    # task query
        assert "## 背景信息" in prompt   # task summary
        assert "S1" in prompt            # task sources

    def test_empty_todo_items(self, reporter: ReportingService, mock_agent: MagicMock) -> None:
        empty_state = SummaryState(research_topic="测试")
        mock_agent.run.return_value = "报告"
        report = reporter.generate_report(empty_state)
        assert report == "报告"

    def test_no_note_id_shows_fallback_message(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = "report"
        reporter.generate_report(state)
        prompt = mock_agent.run.call_args[0][0]
        assert "暂无可用任务笔记" in prompt

    def test_strips_tool_calls_from_report(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = "内容[TOOL_CALL:note:{}]结尾"
        report = reporter.generate_report(state)
        assert "内容结尾" == report
        assert "[TOOL_CALL" not in report

    def test_strips_think_tokens(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = "<think>思考</think>可见"
        report = reporter.generate_report(state)
        assert report == "可见"

    def test_empty_response_falls_back(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = ""
        report = reporter.generate_report(state)
        assert report == "报告生成失败，请检查输入。"

    def test_tool_call_only_response_falls_back(
        self, reporter: ReportingService, state: SummaryState, mock_agent: MagicMock
    ) -> None:
        mock_agent.run.return_value = "[TOOL_CALL:note:{}]"
        report = reporter.generate_report(state)
        assert report == "报告生成失败，请检查输入。"

    def test_with_note_references(self, state: SummaryState, reporter: ReportingService, mock_agent: MagicMock) -> None:
        """When tasks have note_ids, the prompt should include references."""
        state.todo_items[0].note_id = "note_001"
        mock_agent.run.return_value = "report"
        reporter.generate_report(state)
        prompt = mock_agent.run.call_args[0][0]
        assert "任务 1" in prompt
        assert "note_001" in prompt
