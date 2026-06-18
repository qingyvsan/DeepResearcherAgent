"""Tests for models.py — dataclass definitions."""

from __future__ import annotations

from src.models import SummaryState, SummaryStateInput, SummaryStateOutput, TodoItem


class TestTodoItem:
    def test_default_status(self) -> None:
        item = TodoItem(id=1, title="t", intent="i", query="q")
        assert item.status == "pending"

    def test_optional_fields_default_to_none(self) -> None:
        item = TodoItem(id=1, title="t", intent="i", query="q")
        assert item.summary is None
        assert item.sources_summary is None
        assert item.note_id is None
        assert item.note_path is None
        assert item.stream_token is None

    def test_notices_defaults_to_empty_list(self) -> None:
        item = TodoItem(id=1, title="t", intent="i", query="q")
        assert item.notices == []

    def test_notices_is_distinct_per_instance(self) -> None:
        a = TodoItem(id=1, title="t", intent="i", query="q")
        b = TodoItem(id=2, title="t", intent="i", query="q")
        a.notices.append("msg")
        assert len(a.notices) == 1
        assert len(b.notices) == 0

    def test_custom_status(self) -> None:
        item = TodoItem(id=1, title="t", intent="i", query="q", status="completed")
        assert item.status == "completed"

    def test_full_construction(self) -> None:
        item = TodoItem(
            id=42,
            title="Full",
            intent="Test intent",
            query="test query",
            status="in_progress",
            summary="Short summary",
            sources_summary="* src : url",
            notices=["rate-limited"],
            note_id="note_001",
            note_path="/path/to/note.md",
            stream_token="tok_abc",
        )
        assert item.id == 42
        assert item.note_id == "note_001"
        assert item.stream_token == "tok_abc"


class TestSummaryState:
    def test_default_loop_count_zero(self) -> None:
        state = SummaryState(research_topic="test")
        assert state.research_loop_count == 0

    def test_default_collections_are_empty(self) -> None:
        state = SummaryState(research_topic="test")
        assert state.web_research_results == []
        assert state.sources_gathered == []
        assert state.todo_items == []

    def test_str_fields_can_be_none_at_runtime(self) -> None:
        """research_topic is typed as str but defaults to None via field(default=None)."""
        state = SummaryState(research_topic=None)  # type: ignore[arg-type]
        assert state.research_topic is None

    def test_structured_report_defaults_to_none(self) -> None:
        state = SummaryState(research_topic="t")
        assert state.structured_report is None

    def test_running_summary_defaults_to_none(self) -> None:
        state = SummaryState(research_topic="t")
        assert state.running_summary is None


class TestSummaryStateInput:
    def test_topic_defaults_to_none(self) -> None:
        inp = SummaryStateInput()
        assert inp.research_topic is None

    def test_construction(self) -> None:
        inp = SummaryStateInput(research_topic="AI")
        assert inp.research_topic == "AI"


class TestSummaryStateOutput:
    def test_running_summary_default_to_none(self) -> None:
        out = SummaryStateOutput()
        assert out.running_summary is None

    def test_todo_items_defaults_to_empty_list(self) -> None:
        out = SummaryStateOutput()
        assert out.todo_items == []

    def test_todo_items_is_distinct_per_instance(self) -> None:
        a = SummaryStateOutput()
        b = SummaryStateOutput()
        a.todo_items.append(TodoItem(id=1, title="t", intent="i", query="q"))
        assert len(a.todo_items) == 1
        assert len(b.todo_items) == 0

    def test_report_markdown_default_none(self) -> None:
        out = SummaryStateOutput()
        assert out.report_markdown is None
