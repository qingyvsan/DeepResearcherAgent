"""Tests for ToolCallTracker — event recording, draining, and inference."""

from __future__ import annotations

from threading import Thread
from typing import Any

import pytest

from src.models import SummaryState, TodoItem
from src.services.tool_events import ToolCallTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> ToolCallTracker:
    return ToolCallTracker(notes_workspace="./test_notes")


@pytest.fixture
def note_create_event() -> dict[str, Any]:
    return {
        "agent_name": "研究规划专家",
        "tool_name": "note",
        "raw_parameters": '{"action":"create","task_id":1,'
                         '"title":"任务 1","tags":["task_1"]}',
        "parsed_parameters": {
            "action": "create",
            "task_id": 1,
            "title": "任务 1",
            "tags": ["deep_research", "task_1"],
        },
        "result": "ID: note_abc123\nContent saved.",
    }


@pytest.fixture
def search_event() -> dict[str, Any]:
    return {
        "agent_name": "研究执行专家",
        "tool_name": "search",
        "raw_parameters": '{"input":"AI query"}',
        "parsed_parameters": {"input": "AI query"},
        "result": '{"results": []}',
    }


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_records_event(self, tracker: ToolCallTracker, note_create_event: dict[str, Any]) -> None:
        tracker.record(note_create_event)
        assert len(tracker._events) == 1

    def test_records_note_id_from_parameters(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        tracker.record(note_create_event)
        event = tracker._events[0]
        assert event.note_id == "note_abc123"

    def test_records_note_id_from_result_fallback(
        self, tracker: ToolCallTracker
    ) -> None:
        event = {
            "agent_name": "agent",
            "tool_name": "note",
            "raw_parameters": '{"action":"create"}',
            "parsed_parameters": {"action": "create"},
            "result": "ID: note_fallback\nSaved.",
        }
        tracker.record(event)
        assert tracker._events[0].note_id == "note_fallback"

    def test_empty_parameters_becomes_empty_dict(
        self, tracker: ToolCallTracker
    ) -> None:
        event = {
            "agent_name": "agent",
            "tool_name": "tool",
            "raw_parameters": "",
            "parsed_parameters": "not_a_dict",  # will be reset to {}
            "result": "",
        }
        tracker.record(event)
        assert tracker._events[0].parsed_parameters == {}

    def test_infers_task_id(self, tracker: ToolCallTracker, note_create_event: dict[str, Any]) -> None:
        tracker.record(note_create_event)
        assert tracker._events[0].task_id == 1

    def test_calls_event_sink(self, tracker: ToolCallTracker) -> None:
        sink_calls: list[dict[str, Any]] = []
        tracker.set_event_sink(sink_calls.append)
        tracker.record({
            "agent_name": "a", "tool_name": "t",
            "raw_parameters": "", "parsed_parameters": {}, "result": "",
        })
        assert len(sink_calls) == 1
        assert sink_calls[0]["type"] == "tool_call"

    def test_missing_keys_use_defaults(self, tracker: ToolCallTracker) -> None:
        tracker.record({})
        assert len(tracker._events) == 1
        assert tracker._events[0].agent == "unknown"
        assert tracker._events[0].tool == "unknown"


# ---------------------------------------------------------------------------
# drain
# ---------------------------------------------------------------------------


class TestDrain:
    def test_no_new_events_returns_empty(
        self, tracker: ToolCallTracker
    ) -> None:
        assert tracker.drain(SummaryState(research_topic="t")) == []

    def test_drains_new_events(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        tracker.record(note_create_event)
        events = tracker.drain(SummaryState(research_topic="t"))
        assert len(events) == 1
        assert events[0]["type"] == "tool_call"

    def test_drain_is_idempotent(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        tracker.record(note_create_event)
        tracker.drain(SummaryState(research_topic="t"))
        # Second drain should return nothing
        assert tracker.drain(SummaryState(research_topic="t")) == []

    def test_drain_attaches_note_to_task(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        task = TodoItem(id=1, title="T", intent="I", query="Q")
        state = SummaryState(research_topic="t")
        state.todo_items = [task]
        tracker.record(note_create_event)
        tracker.drain(state)
        assert task.note_id == "note_abc123"


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_all_events(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        tracker.record(note_create_event)
        tracker.reset()
        assert tracker._events == []
        assert tracker._cursor == 0


# ---------------------------------------------------------------------------
# as_dicts
# ---------------------------------------------------------------------------


class TestAsDicts:
    def test_returns_snapshot(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        tracker.record(note_create_event)
        snap = tracker.as_dicts()
        assert len(snap) == 1
        assert snap[0]["tool"] == "note"
        assert snap[0]["task_id"] == 1

    def test_is_thread_safe(
        self, tracker: ToolCallTracker, note_create_event: dict[str, Any]
    ) -> None:
        """Concurrent calls to as_dicts should not raise."""
        errors: list[Exception] = []

        def read() -> None:
            try:
                for _ in range(50):
                    tracker.as_dicts()
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=read) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# _infer_task_id
# ---------------------------------------------------------------------------


class TestInferTaskId:
    def test_from_task_id_param(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({"task_id": 3}) == 3

    def test_from_tags(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({"tags": ["deep", "task_2"]}) == 2

    def test_from_title(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({"title": "任务 4: 调研"}) == 4

    def test_no_match_returns_none(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({"action": "create"}) is None

    def test_empty_params_returns_none(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({}) is None

    def test_task_id_non_int_falls_through(self, tracker: ToolCallTracker) -> None:
        assert tracker._infer_task_id({"task_id": "abc"}) is None


# ---------------------------------------------------------------------------
# _attach_note_to_task
# ---------------------------------------------------------------------------


class TestAttachNoteToTask:
    def test_updates_matching_task(self, tracker: ToolCallTracker) -> None:
        tasks = [TodoItem(id=1, title="T", intent="I", query="Q")]
        tracker._attach_note_to_task(tasks, 1, "note_new")
        assert tasks[0].note_id == "note_new"
        assert tasks[0].note_path is not None

    def test_skips_when_no_match(self, tracker: ToolCallTracker) -> None:
        tasks = [TodoItem(id=1, title="T", intent="I", query="Q")]
        tracker._attach_note_to_task(tasks, 99, "note_x")
        assert tasks[0].note_id is None

    def test_same_note_id_does_not_overwrite_path(
        self, tracker: ToolCallTracker
    ) -> None:
        tasks = [
            TodoItem(id=1, title="T", intent="I", query="Q",
                     note_id="note_x", note_path="/custom/path")
        ]
        tracker._attach_note_to_task(tasks, 1, "note_x")
        # note_id unchanged, note_path should remain as originally set
        assert tasks[0].note_path == "/custom/path"

    def test_updates_only_first_match(self, tracker: ToolCallTracker) -> None:
        tasks = [
            TodoItem(id=1, title="A", intent="I", query="Q"),
            TodoItem(id=1, title="B", intent="I", query="Q"),
        ]
        tracker._attach_note_to_task(tasks, 1, "note_first")
        assert tasks[0].note_id == "note_first"
        assert tasks[1].note_id is None


# ---------------------------------------------------------------------------
# _extract_note_id
# ---------------------------------------------------------------------------


class TestExtractNoteId:
    def test_standard_format(self, tracker: ToolCallTracker) -> None:
        assert tracker._extract_note_id("ID: note_001\nContent") == "note_001"

    def test_no_match(self, tracker: ToolCallTracker) -> None:
        assert tracker._extract_note_id("no id here") is None

    def test_empty(self, tracker: ToolCallTracker) -> None:
        assert tracker._extract_note_id("") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_record_and_drain(
        self, tracker: ToolCallTracker
    ) -> None:
        """Simultaneous record/drain calls should not corrupt state."""

        def writer() -> None:
            for i in range(100):
                tracker.record({
                    "agent_name": "a",
                    "tool_name": "t",
                    "raw_parameters": str(i),
                    "parsed_parameters": {"i": i},
                    "result": str(i),
                })

        def reader() -> None:
            state = SummaryState(research_topic="t")
            for _ in range(50):
                tracker.drain(state)

        threads = [Thread(target=writer) for _ in range(3)]
        threads += [Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = len(tracker._events)
        assert total == 300  # 3 writers × 100 events each
        # Draining should not lose events
        final_events = tracker.drain(SummaryState(research_topic="t"))
        assert len(final_events) <= 300
        assert len(tracker._events) == 300
