"""Tests for the PlanningService — task planning and JSON parsing."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.models import SummaryState, TodoItem
from src.services.planner import PlanningService, _extract_task_number

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def planner(mock_agent: MagicMock, test_config: Any) -> PlanningService:
    """Build a PlanningService with a mocked agent and minimal config."""
    return PlanningService(mock_agent, test_config)


@pytest.fixture
def state() -> SummaryState:
    return SummaryState(research_topic="人工智能发展")


# ---------------------------------------------------------------------------
# _extract_task_number (module-level helper)
# ---------------------------------------------------------------------------


class TestExtractTaskNumber:
    def test_normal(self) -> None:
        assert _extract_task_number("任务 3: 背景") == 3

    def test_no_space(self) -> None:
        assert _extract_task_number("任务1: 背景") == 1

    def test_no_match_returns_999(self) -> None:
        assert _extract_task_number("自由文本") == 999


# ---------------------------------------------------------------------------
# plan_todo_list
# ---------------------------------------------------------------------------


class TestPlanTodoList:
    def test_parses_json_dict_with_tasks(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        """The prompt tells agents to return ``{"tasks": [...]}`` — this is the
        primary code path."""
        planner._agent.run.return_value = (
            '{"tasks": [{"title": "背景调研", "intent": "收集背景", "query": "AI 背景"}]}'
        )
        items = planner.plan_todo_list(state)
        assert len(items) == 1
        assert items[0].title == "背景调研"

    def test_multiple_tasks(self, planner: PlanningService, state: SummaryState) -> None:
        planner._agent.run.return_value = (
            '{"tasks": ['
            '{"title": "任务1", "intent": "意图1", "query": "查询1"},'
            '{"title": "任务2", "intent": "意图2", "query": "查询2"}'
            "]}"
        )
        items = planner.plan_todo_list(state)
        assert len(items) == 2
        assert items[0].title == "任务1"
        assert items[1].query == "查询2"
        assert planner._agent.clear_history.called

    def test_empty_response_returns_no_tasks(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        planner._agent.run.return_value = ""
        items = planner.plan_todo_list(state)
        assert items == []

    def test_fallback_defaults_for_missing_fields(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        planner._agent.run.return_value = (
            '{"tasks": [{"title": "仅有标题"}]}'
        )
        items = planner.plan_todo_list(state)
        assert len(items) == 1
        assert items[0].intent == "聚焦主题的关键问题"
        assert items[0].query == "人工智能发展"

    def test_query_fallback_to_topic(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        planner._agent.run.return_value = '{"tasks": [{"title": "T", "intent": "I"}]}'
        items = planner.plan_todo_list(state)
        assert items[0].query == "人工智能发展"

    def test_think_tokens_are_stripped(self, planner: PlanningService, state: SummaryState) -> None:
        planner._agent.run.return_value = (
            "<think>planning...</think>"
            '{"tasks": [{"title": "任务", "intent": "意图", "query": "查询"}]}'
        )
        items = planner.plan_todo_list(state)
        assert len(items) == 1

    def test_prompt_contains_topic_and_date(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        planner._agent.run.return_value = '{"tasks": []}'
        planner.plan_todo_list(state)
        call_args = planner._agent.run.call_args[0][0]
        assert "人工智能发展" in call_args
        assert "202" in call_args  # year in date string

    def test_too_many_tasks_ids_are_sequential(
        self, planner: PlanningService, state: SummaryState
    ) -> None:
        planner._agent.run.return_value = (
            '{"tasks": ['
            '{"title": "A", "intent": "I1", "query": "Q1"},'
            '{"title": "B", "intent": "I2", "query": "Q2"},'
            '{"title": "C", "intent": "I3", "query": "Q3"}'
            "]}"
        )
        items = planner.plan_todo_list(state)
        assert [t.id for t in items] == [1, 2, 3]


# ---------------------------------------------------------------------------
# create_fallback_task (static)
# ---------------------------------------------------------------------------


class TestCreateFallbackTask:
    def test_with_topic(self) -> None:
        st = SummaryState(research_topic="量子计算")
        task = PlanningService.create_fallback_task(st)
        assert task.title == "基础背景梳理"
        assert "量子计算" in task.query

    def test_with_empty_topic(self) -> None:
        st = SummaryState(research_topic="")
        task = PlanningService.create_fallback_task(st)
        assert task.query == "基础背景梳理"

    def test_with_none_topic(self) -> None:
        st = SummaryState(research_topic=None)  # type: ignore[arg-type]
        task = PlanningService.create_fallback_task(st)
        assert task.query == "基础背景梳理"


# ---------------------------------------------------------------------------
# _extract_tasks (low-level parsing, bypasses agent)
# ---------------------------------------------------------------------------


class TestExtractTasks:
    def test_json_dict_with_tasks(self, planner: PlanningService) -> None:
        text = '{"tasks": [{"title": "A", "intent": "B", "query": "C"}]}'
        result = planner._extract_tasks(text)
        assert len(result) == 1
        assert result[0]["title"] == "A"

    def test_json_dict_without_tasks_key(self, planner: PlanningService) -> None:
        text = '{"not_tasks": [{"title": "A"}]}'
        result = planner._extract_tasks(text)
        assert result == []

    def test_json_array_with_two_items_falls_through(
        self, planner: PlanningService
    ) -> None:
        """``[{...},{...}]`` — the ``{}`` parser picks up the first object
        (``json.loads`` of two concatenated objects fails), then the ``[]``
        parser succeeds.  This is a fragile code path.
        """
        text = '[{"title": "A"}, {"title": "B"}]'
        result = planner._extract_tasks(text)
        assert len(result) == 2

    def test_single_item_array_treated_as_dict(
        self, planner: PlanningService
    ) -> None:
        """``[{...}]`` — ``{}`` parser finds the inner dict and returns it
        directly, so ``.get("tasks")`` fails and no tasks are extracted.
        This is a known limitation of the parsing order.
        """
        text = '[{"title": "Solo"}]'
        result = planner._extract_tasks(text)
        assert result == []

    def test_json_list_of_non_dicts(self, planner: PlanningService) -> None:
        text = '["a", "b"]'
        result = planner._extract_tasks(text)
        assert result == []

    def test_tool_call_without_inner_brackets(self, planner: PlanningService) -> None:
        """TOOL_CALL with no ``]`` inside the JSON body is parsed correctly
        via the ``_extract_tasks_from_note_calls`` fallback."""
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,'
            '"title":"任务 1", "intent":"意图", "tags":"task_1"}]'
        )
        result = planner._extract_tasks(text)
        assert len(result) >= 1
        assert result[0]["title"] == "任务 1"

    def test_no_parsable_content(self, planner: PlanningService) -> None:
        result = planner._extract_tasks("纯文本无结构化内容")
        assert result == []


# ---------------------------------------------------------------------------
# _extract_json_payload (unit)
# ---------------------------------------------------------------------------


class TestExtractJsonPayload:
    def test_valid_object(self, planner: PlanningService) -> None:
        text = '前文{"key": "value"}后文'
        result = planner._extract_json_payload(text)
        assert result == {"key": "value"}

    def test_valid_array(self, planner: PlanningService) -> None:
        text = '前文[1, 2, 3]后文'
        result = planner._extract_json_payload(text)
        assert result == [1, 2, 3]

    def test_object_after_text(self, planner: PlanningService) -> None:
        text = '先输出一些描述{"tasks": []}'
        result = planner._extract_json_payload(text)
        assert result == {"tasks": []}

    def test_invalid_json(self, planner: PlanningService) -> None:
        result = planner._extract_json_payload("{invalid")
        assert result is None

    def test_no_braces(self, planner: PlanningService) -> None:
        result = planner._extract_json_payload("no brackets at all")
        assert result is None

    def test_unmatched_braces(self, planner: PlanningService) -> None:
        result = planner._extract_json_payload('{"key": value')
        assert result is None

    def test_object_takes_priority_over_array(self, planner: PlanningService) -> None:
        text = '{"a": 1}[1, 2]'
        result = planner._extract_json_payload(text)
        assert result == {"a": 1}

    def test_single_object_inside_array_returns_inner_dict(
        self, planner: PlanningService
    ) -> None:
        """The ``{}`` search finds the inner dict before the ``[]`` search
        is attempted, so ``[{...}]`` returns the inner dict."""
        text = '[{"x": 1}]'
        result = planner._extract_json_payload(text)
        assert result == {"x": 1}  # not [{"x": 1}]


# ---------------------------------------------------------------------------
# _extract_tool_payload (unit)
# ---------------------------------------------------------------------------


class TestExtractToolPayload:
    def test_json_body(self, planner: PlanningService) -> None:
        text = '[TOOL_CALL:note:{"action":"create","task_id":1}]'
        result = planner._extract_tool_payload(text)
        assert result == {"action": "create", "task_id": 1}

    def test_key_value_body(self, planner: PlanningService) -> None:
        text = "[TOOL_CALL:note:action=create,task_id=1]"
        result = planner._extract_tool_payload(text)
        assert result == {"action": "create", "task_id": "1"}

    def test_no_match(self, planner: PlanningService) -> None:
        assert planner._extract_tool_payload("no tool call") is None

    def test_invalid_json_body_falls_back_to_kv(
        self, planner: PlanningService
    ) -> None:
        text = "[TOOL_CALL:note:action=create,title=hello]"
        result = planner._extract_tool_payload(text)
        assert result == {"action": "create", "title": "hello"}


# ---------------------------------------------------------------------------
# _extract_tasks_from_note_calls (unit)
# ---------------------------------------------------------------------------
#
# NOTE on TOOL_CALL_PATTERN limitation:
#   The regex ``[^\]]+`` stops at the first ``]`` inside the body,
#   so ``[TOOL_CALL:note:{"tags":["a","b"]}]`` is truncated at the ``]``
#   inside ``["a","b"]``, making the JSON unparseable.
#   All test inputs below avoid ``]`` inside the note body (use
#   ``"tags":"task_1"`` string form instead of ``["task_1"]`` array form).
# ---------------------------------------------------------------------------


class TestExtractTasksFromNoteCalls:
    def test_standard_note_call(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,'
            '"title":"任务 1: 背景","tags":"task_1",'
            '"content":"检索方向：AI 趋势"}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "任务 1: 背景"

    def test_deduplicates_by_task_id(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"A",'
            '"tags":"task_1","content":""}]'
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"B",'
            '"tags":"task_1","content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "A"

    def test_tags_as_string(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"T",'
            '"tags":"task_1","content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert len(tasks) == 1

    def test_json_tags_array_with_brackets_not_parsed(
        self, planner: PlanningService
    ) -> None:
        """``["task_1"]`` contains ``]`` which truncates the TOOL_CALL_PATTERN
        regex — this is a known limitation."""
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"T",'
            '"tags":["task_1"],"content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert tasks == []

    def test_no_task_tag(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","title":"T",'
            '"tags":"other","content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert tasks == []

    def test_extracts_query_from_content(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"T",'
            '"tags":"task_1","content":"检索方向：深度学习 2025"}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI")
        assert tasks[0]["query"] == "深度学习 2025"

    def test_query_falls_back_to_topic_title(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"背景",'
            '"tags":"task_1","content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "AI 研究")
        assert "AI 研究" in tasks[0]["query"]

    def test_sorts_by_task_number(self, planner: PlanningService) -> None:
        text = (
            '[TOOL_CALL:note:{"action":"create","task_id":2,"title":"任务 2",'
            '"tags":"task_2","content":""}]'
            '[TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务 1",'
            '"tags":"task_1","content":""}]'
        )
        tasks = planner._extract_tasks_from_note_calls(text, "T")
        assert len(tasks) == 2
        assert tasks[0]["title"] == "任务 1"  # sorted by id
        assert tasks[1]["title"] == "任务 2"
