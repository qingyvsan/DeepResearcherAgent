"""Tests for notes.py — build_note_guidance."""

from __future__ import annotations

import json

import pytest

from src.models import TodoItem
from src.services.notes import build_note_guidance


class TestBuildNoteGuidance:
    def test_without_note_id_contains_create_instruction(self) -> None:
        task = TodoItem(id=1, title="背景调研", intent="收集背景", query="AI 背景")
        guidance = build_note_guidance(task)
        assert "尚未建立笔记" in guidance
        assert "TOOL_CALL:note:" in guidance
        assert '"action": "create"' in guidance
        assert "task_1" in guidance

    def test_without_note_id_json_payload_is_valid(self) -> None:
        task = TodoItem(id=3, title="Test", intent="Test", query="test")
        guidance = build_note_guidance(task)
        # Extract the JSON payload from the create instruction
        start = guidance.find('{')
        end = guidance.rfind('}') + 1
        payload = json.loads(guidance[start:end])
        assert payload["action"] == "create"
        assert payload["task_id"] == 3
        assert "task_3" in payload["tags"]

    def test_with_note_id_contains_read_and_update(self) -> None:
        task = TodoItem(
            id=2,
            title="竞品分析",
            intent="分析竞品",
            query="competitor analysis",
            note_id="note_abc123",
        )
        guidance = build_note_guidance(task)
        assert "当前任务笔记 ID：note_abc123" in guidance
        assert '"action": "read"' in guidance
        assert '"action": "update"' in guidance
        assert "task_2" in guidance

    def test_with_note_id_json_payload_is_valid(self) -> None:
        task = TodoItem(
            id=5,
            title="My Task",
            intent="Intent",
            query="query",
            note_id="note_xyz",
        )
        guidance = build_note_guidance(task)
        # Should contain both a read and update payload
        assert guidance.count("TOOL_CALL:note:") >= 2

    def test_title_with_special_characters(self) -> None:
        task = TodoItem(
            id=1,
            title='测试"引号"任务',
            intent="测试",
            query="test",
        )
        guidance = build_note_guidance(task)
        # The title should appear JSON-escaped in the payload
        assert '测试' in guidance

    def test_task_id_zero(self) -> None:
        task = TodoItem(id=0, title="Zero", intent="Test", query="t")
        guidance = build_note_guidance(task)
        assert "task_0" in guidance

    def test_task_id_negative(self) -> None:
        task = TodoItem(id=-1, title="Negative", intent="Test", query="t")
        guidance = build_note_guidance(task)
        assert "task_-1" in guidance

    def test_with_note_id_update_json_payload(self) -> None:
        task = TodoItem(
            id=7,
            title="更新测试",
            intent="验证更新",
            query="update test",
            note_id="note_upd",
        )
        guidance = build_note_guidance(task)
        # Find the update payload
        for line in guidance.split('\n'):
            if 'TOOL_CALL:note:' in line and 'update' in line:
                start = line.find('{')
                end = line.rfind('}') + 1
                payload = json.loads(line[start:end])
                assert payload["action"] == "update"
                assert payload["note_id"] == "note_upd"
                break
        else:
            pytest.fail("No update instruction found in guidance")  # type: ignore[name-defined]  # noqa: F821
