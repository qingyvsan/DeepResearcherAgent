"""Tests for DeepResearchAgent — orchestration, fallbacks, and streaming."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agent import DeepResearchAgent
from src.config import Configuration
from src.models import SummaryState, SummaryStateOutput, TodoItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent(test_config: Configuration, mocker: pytest.MockFixture) -> DeepResearchAgent:
    """Build a ``DeepResearchAgent`` with all external dependencies mocked.

    * ``HelloAgentsLLM`` patched to prevent real API calls.
    * ``dispatch_search`` / ``prepare_research_context`` patched to return
      canned results.
    * Internal services (planner, summarizer, reporter) replaced with
      ``MagicMock`` so tests control orchestration.
    """
    mocker.patch("src.agent.HelloAgentsLLM")
    _mock_search(test_config, mocker)

    agent = DeepResearchAgent(config=test_config)
    _mock_services(agent, mocker)
    return agent


@pytest.fixture
def agent_with_notes(
    test_config: Configuration,
    mocker: pytest.MockFixture,
) -> DeepResearchAgent:
    """Same as ``agent`` but with ``enable_notes=True`` and a mocked
    ``NoteTool.run`` to avoid filesystem side-effects."""
    mocker.patch("src.agent.HelloAgentsLLM")
    _mock_search(test_config, mocker)

    config = Configuration(
        enable_notes=True,
        search_api="duckduckgo",
        local_llm="test-model",
        notes_workspace="./test_notes",
    )
    agent = DeepResearchAgent(config=config)
    mocker.patch.object(
        agent.note_tool, "run", return_value="ID: note_new\nContent saved."
    )
    _mock_services(agent, mocker)
    return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search(config: Configuration, mocker: pytest.MockFixture) -> None:
    """Prevent real web search calls from ``_execute_task``."""
    mock_result: dict[str, Any] = {
        "results": [
            {
                "title": "Mock Source",
                "url": "https://example.com/mock",
                "content": "Mock body",
            },
        ],
        "backend": "duckduckgo",
    }
    mocker.patch(
        "src.agent.dispatch_search",
        return_value=(mock_result, [], None, "duckduckgo"),
    )
    mocker.patch(
        "src.agent.prepare_research_context",
        return_value=("* Mock Source : url", "Mock context body"),
    )


def _mock_services(agent: DeepResearchAgent, mocker: pytest.MockFixture) -> None:
    """Replace internal services with mocks for orchestration testing."""
    agent.planner = mocker.MagicMock(name="planner")
    agent.summarizer = mocker.MagicMock(name="summarizer")
    agent.reporting = mocker.MagicMock(name="reporter")
    agent._tool_tracker = mocker.MagicMock(name="tool_tracker")
    agent._tool_tracker.drain.return_value = []
    agent._tool_tracker.as_dicts.return_value = []


@pytest.fixture
def task() -> TodoItem:
    return TodoItem(id=1, title="背景调研", intent="收集背景", query="AI 趋势")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_notes_disabled_no_note_tool(self, agent: DeepResearchAgent) -> None:
        assert agent.note_tool is None
        assert agent.tools_registry is None

    def test_notes_enabled_creates_note_tool(
        self, agent_with_notes: DeepResearchAgent
    ) -> None:
        assert agent_with_notes.note_tool is not None
        assert agent_with_notes.tools_registry is not None


# ---------------------------------------------------------------------------
# run (synchronous)
# ---------------------------------------------------------------------------


class TestRun:
    def test_normal_flow(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        agent.planner.plan_todo_list.return_value = [task]
        agent.summarizer.summarize_task.return_value = "## 总结内容"
        agent.reporting.generate_report.return_value = "# 最终报告"

        result: SummaryStateOutput = agent.run("测试主题")

        assert result.report_markdown == "# 最终报告"
        assert len(result.todo_items) == 1
        assert result.todo_items[0].id == 1
        assert result.todo_items[0].status == "completed"
        assert agent.planner.plan_todo_list.called
        assert agent.reporting.generate_report.called

    def test_empty_todo_creates_fallback(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        agent.planner.plan_todo_list.return_value = []
        agent.planner.create_fallback_task.return_value = task
        agent.summarizer.summarize_task.return_value = "fallback summary"
        agent.reporting.generate_report.return_value = "# Report"

        result = agent.run("测试主题")

        assert len(result.todo_items) == 1
        assert agent.planner.create_fallback_task.called

    def test_summary_only_triggers_reconstruction(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        # The planner returned a "plan complete" placeholder
        summary_task = TodoItem(
            id=1, title="规划完成", intent="", query="无需检索"
        )
        agent.planner.plan_todo_list.return_value = [summary_task]

        # Tool tracker has real task events for reconstruction
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "note",
                "parsed_parameters": {
                    "action": "create",
                    "task_id": 1,
                    "title": "任务 1: 背景",
                    "intent": "收集背景",
                    "content": "检索方向：AI 最新进展",
                },
                "result": "ID: note_001",
            },
        ]

        agent.summarizer.summarize_task.return_value = "summary"
        agent.reporting.generate_report.return_value = "# Report"

        result = agent.run("AI 研究")

        assert len(result.todo_items) == 1
        assert result.todo_items[0].title == "任务 1: 背景"
        assert result.todo_items[0].query == "AI 最新进展"

    def test_reconstruction_no_events_falls_back(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        summary_task = TodoItem(
            id=1, title="planning complete", intent="", query="no need"
        )
        agent.planner.plan_todo_list.return_value = [summary_task]
        # No note events → reconstruction returns None → original task kept
        agent.summarizer.summarize_task.return_value = "s"
        agent.reporting.generate_report.return_value = "# R"

        result = agent.run("t")

        # The original summary-only task should remain
        assert len(result.todo_items) == 1
        assert result.todo_items[0].title == "planning complete"


# ---------------------------------------------------------------------------
# _execute_task
# ---------------------------------------------------------------------------


class TestExecuteTask:
    def test_normal_completes_task(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        state = SummaryState(research_topic="测试")
        agent.summarizer.summarize_task.return_value = "## 总结"

        for _ in agent._execute_task(state, task, emit_stream=False):
            pass

        assert task.status == "completed"
        assert task.summary == "## 总结"
        assert task.sources_summary == "* Mock Source : url"
        # State should be updated
        assert state.research_loop_count > 0

    def test_no_results_skips_task(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "src.agent.dispatch_search",
            return_value=({"results": [], "backend": "mock"}, [], None, "mock"),
        )
        state = SummaryState(research_topic="测试")

        for _ in agent._execute_task(state, task, emit_stream=False):
            pass

        assert task.status == "skipped"
        assert task.summary is None

    def test_emit_stream_yields_events(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        state = SummaryState(research_topic="测试")
        agent.summarizer.stream_task_summary.return_value = (
            iter(["chunk1", "chunk2"]),
            lambda: "## 总结",
        )

        events = list(agent._execute_task(state, task, emit_stream=True, step=1))

        types = {e["type"] for e in events}
        assert "sources" in types
        assert "task_summary_chunk" in types
        assert "task_status" in types
        assert task.status == "completed"

    def test_emit_stream_no_results_yields_skip_status(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
        mocker: pytest.MockFixture,
    ) -> None:
        mocker.patch(
            "src.agent.dispatch_search",
            return_value=({"results": [], "backend": "mock"}, [], None, "mock"),
        )
        state = SummaryState(research_topic="t")

        events = list(agent._execute_task(state, task, emit_stream=True, step=1))

        status_events = [e for e in events if e.get("type") == "task_status"]
        assert any(e.get("status") == "skipped" for e in status_events)


# ---------------------------------------------------------------------------
# _is_summary_only_tasks (static)
# ---------------------------------------------------------------------------


class TestIsSummaryOnlyTasks:
    def test_plan_complete_title(self) -> None:
        t = TodoItem(id=1, title="规划完成", intent="", query="q")
        assert DeepResearchAgent._is_summary_only_tasks([t])

    def test_plan_complete_english(self) -> None:
        t = TodoItem(id=1, title="plan complete", intent="", query="q")
        assert DeepResearchAgent._is_summary_only_tasks([t])

    def test_no_need_query(self) -> None:
        t = TodoItem(id=1, title="Task", intent="", query="无需检索")
        assert DeepResearchAgent._is_summary_only_tasks([t])

    def test_normal_task_returns_false(self) -> None:
        t = TodoItem(id=1, title="背景调研", intent="收集背景", query="AI 背景")
        assert not DeepResearchAgent._is_summary_only_tasks([t])

    def test_multiple_tasks_returns_false(self) -> None:
        tasks = [
            TodoItem(id=1, title="规划完成", intent="", query="q"),
            TodoItem(id=2, title="Task 2", intent="", query="q2"),
        ]
        assert not DeepResearchAgent._is_summary_only_tasks(tasks)


# ---------------------------------------------------------------------------
# _reconstruct_tasks_from_events
# ---------------------------------------------------------------------------


class TestReconstructTasksFromEvents:
    def test_reconstructs_from_note_events(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "note",
                "parsed_parameters": {
                    "action": "create",
                    "task_id": 1,
                    "title": "任务 1: 背景",
                    "intent": "收集背景知识",
                    "content": "检索方向：AI 发展史",
                },
                "result": "ID: note_001",
            },
        ]
        state = SummaryState(research_topic="AI 研究")

        result = agent._reconstruct_tasks_from_events(state)

        assert result is not None
        assert len(result) == 1
        assert result[0].title == "任务 1: 背景"
        assert result[0].query == "AI 发展史"

    def test_no_note_events_returns_none(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        agent._tool_tracker.as_dicts.return_value = []
        state = SummaryState(research_topic="t")

        assert agent._reconstruct_tasks_from_events(state) is None

    def test_ignores_non_note_tools(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "search",
                "parsed_parameters": {"input": "query"},
                "result": '{"results": []}',
            },
        ]
        state = SummaryState(research_topic="t")

        assert agent._reconstruct_tasks_from_events(state) is None


# ---------------------------------------------------------------------------
# _persist_final_report
# ---------------------------------------------------------------------------


class TestPersistFinalReport:
    def test_notes_disabled_returns_none(self, agent: DeepResearchAgent) -> None:
        state = SummaryState(research_topic="t")
        assert agent._persist_final_report(state, "# Report") is None

    def test_notes_enabled_creates_note(
        self,
        agent_with_notes: DeepResearchAgent,
    ) -> None:
        state = SummaryState(research_topic="AI")

        result = agent_with_notes._persist_final_report(state, "# Final Report")

        assert result is not None
        assert result["type"] == "report_note"
        assert result["note_id"] == "note_new"
        assert state.report_note_id == "note_new"

    def test_empty_report_skips(self, agent_with_notes: DeepResearchAgent) -> None:
        state = SummaryState(research_topic="t")
        assert agent_with_notes._persist_final_report(state, "") is None
        assert agent_with_notes._persist_final_report(state, "   ") is None


# ---------------------------------------------------------------------------
# _find_existing_report_note_id
# ---------------------------------------------------------------------------


class TestFindExistingReportNoteId:
    def test_finds_existing_conclusion_note(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "note",
                "parsed_parameters": {
                    "action": "create",
                    "note_type": "conclusion",
                    "note_id": "note_conclusion",
                    "title": "研究报告：AI",
                },
                "result": "ID: note_conclusion",
            },
        ]
        state = SummaryState(research_topic="AI")

        note_id = agent._find_existing_report_note_id(state)
        assert note_id == "note_conclusion"

    def test_finds_by_title_when_no_note_type(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "note",
                "parsed_parameters": {
                    "action": "create",
                    "title": "研究报告：AI 发展",
                    "note_id": "note_report",
                },
                "result": "ID: note_report",
            },
        ]
        state = SummaryState(research_topic="AI")

        note_id = agent._find_existing_report_note_id(state)
        assert note_id == "note_report"

    def test_uses_state_if_already_set(self, agent: DeepResearchAgent) -> None:
        state = SummaryState(research_topic="AI", report_note_id="note_cached")
        assert agent._find_existing_report_note_id(state) == "note_cached"
        # as_dicts should not be called
        agent._tool_tracker.as_dicts.assert_not_called()

    def test_no_match_returns_none(self, agent: DeepResearchAgent) -> None:
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "search",
                "parsed_parameters": {"input": "q"},
                "result": "{}",
            },
        ]
        state = SummaryState(research_topic="t")
        assert agent._find_existing_report_note_id(state) is None


# ---------------------------------------------------------------------------
# _guess_query_from_content (static)
# ---------------------------------------------------------------------------


class TestGuessQueryFromContent:
    def test_keyword_extraction(self) -> None:
        result = DeepResearchAgent._guess_query_from_content(
            "检索方向：深度学习 2025", "背景", "AI"
        )
        assert "深度学习 2025" in result

    def test_keyword_priority_order(self) -> None:
        """Keywords checked in defined order, not content position."""
        # "检索方向" is checked before "查询" in the keyword tuple,
        # so even though "查询" appears first in text, "检索方向" wins.
        result = DeepResearchAgent._guess_query_from_content(
            "随机文本 查询：AI 安全 检索方向：其他方面", "背景", "T"
        )
        assert "其他方面" in result

    def test_no_content_uses_topic_title(self) -> None:
        result = DeepResearchAgent._guess_query_from_content("", "背景", "AI 研究")
        assert "AI 研究" in result
        assert "背景" in result

    def test_no_keyword_falls_back_to_cleaned_content(self) -> None:
        result = DeepResearchAgent._guess_query_from_content(
            "待办任务：一些有意义的描述信息", "标题", "T"
        )
        assert "有意义的描述信息" in result

    def test_empty_content_and_no_topic(self) -> None:
        result = DeepResearchAgent._guess_query_from_content("", "标题", "")
        assert result == "标题"


# ---------------------------------------------------------------------------
# _serialize_task
# ---------------------------------------------------------------------------


class TestSerializeTask:
    def test_returns_all_fields(self, agent: DeepResearchAgent, task: TodoItem) -> None:
        task.status = "completed"
        task.summary = "## 总结"
        task.sources_summary = "* S : url"
        task.note_id = "note_001"
        task.note_path = "/notes/note_001.md"
        task.stream_token = "task_1"

        serialized = agent._serialize_task(task)

        assert serialized["id"] == 1
        assert serialized["title"] == "背景调研"
        assert serialized["status"] == "completed"
        assert serialized["summary"] == "## 总结"
        assert serialized["stream_token"] == "task_1"

    def test_default_values(self, agent: DeepResearchAgent) -> None:
        task = TodoItem(id=1, title="T", intent="I", query="Q")
        serialized = agent._serialize_task(task)
        assert serialized["status"] == "pending"
        assert serialized["summary"] is None
        assert serialized["stream_token"] is None


# ---------------------------------------------------------------------------
# run_stream
# ---------------------------------------------------------------------------


class TestRunStream:
    def test_basic_event_flow(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        agent.planner.plan_todo_list.return_value = [task]
        agent.reporting.generate_report.return_value = "# Final"

        def mock_execute(
            state: SummaryState,
            t: TodoItem,
            *,
            emit_stream: bool,
            step: int | None = None,
        ) -> Any:
            if emit_stream:
                yield {"type": "sources", "task_id": t.id}
                yield {"type": "task_summary_chunk", "task_id": t.id, "content": "c"}

        agent._execute_task = mock_execute

        events = list(agent.run_stream("测试"))
        event_types = [e["type"] for e in events]

        assert event_types[0] == "status"
        assert "todo_list" in event_types
        assert "sources" in event_types
        assert "task_summary_chunk" in event_types
        assert "final_report" in event_types
        assert event_types[-1] == "done"

    def test_empty_todo_fallback_in_stream(
        self,
        agent: DeepResearchAgent,
        task: TodoItem,
    ) -> None:
        agent.planner.plan_todo_list.return_value = []
        agent.planner.create_fallback_task.return_value = task
        agent.reporting.generate_report.return_value = "# R"
        agent._execute_task = lambda *a, **kw: iter([])  # no-op

        events = list(agent.run_stream("测试"))
        event_types = [e["type"] for e in events]

        assert "todo_list" in event_types
        assert "final_report" in event_types
        assert event_types[-1] == "done"

    def test_stream_reconstructs_tasks(
        self,
        agent: DeepResearchAgent,
    ) -> None:
        summary_task = TodoItem(
            id=1, title="planning complete", intent="", query="no need"
        )
        agent.planner.plan_todo_list.return_value = [summary_task]
        agent._tool_tracker.as_dicts.return_value = [
            {
                "tool": "note",
                "parsed_parameters": {
                    "action": "create",
                    "task_id": 1,
                    "title": "任务 1: 背景",
                    "intent": "收集背景",
                    "content": "检索方向：AI 趋势",
                },
                "result": "ID: note_001",
            },
        ]
        agent.reporting.generate_report.return_value = "# R"

        def mock_execute(
            state: SummaryState,
            t: TodoItem,
            *,
            emit_stream: bool,
            step: int | None = None,
        ) -> Any:
            if emit_stream:
                yield {"type": "sources", "task_id": t.id}

        agent._execute_task = mock_execute

        events = list(agent.run_stream("AI"))
        event_types = {e["type"] for e in events}

        assert "todo_list" in event_types
        assert "final_report" in event_types
        assert "done" in event_types
