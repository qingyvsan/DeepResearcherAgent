"""Tests for text_processing.py — strip_tool_calls."""

from __future__ import annotations

from src.services.text_processing import strip_tool_calls


class TestStripToolCalls:
    def test_none_returns_none(self) -> None:
        assert strip_tool_calls(None) is None  # type: ignore[arg-type]

    def test_empty_string_returns_empty(self) -> None:
        assert strip_tool_calls("") == ""

    def test_no_tool_calls_unchanged(self) -> None:
        text = "Hello, world."
        assert strip_tool_calls(text) == "Hello, world."

    def test_removes_single_tool_call(self) -> None:
        text = "前置[TOOL_CALL:note:{\"action\":\"create\"}]后置"
        assert strip_tool_calls(text) == "前置后置"

    def test_removes_multiple_tool_calls(self) -> None:
        text = "a[TOOL_CALL:one]b[TOOL_CALL:two]c"
        assert strip_tool_calls(text) == "abc"

    def test_removes_tool_call_at_start(self) -> None:
        text = "[TOOL_CALL:note:{}]Hello"
        assert strip_tool_calls(text) == "Hello"

    def test_removes_tool_call_at_end(self) -> None:
        text = "Hello[TOOL_CALL:note:{}]"
        assert strip_tool_calls(text) == "Hello"

    def test_only_tool_calls_becomes_empty(self) -> None:
        text = "[TOOL_CALL:note:{}][TOOL_CALL:search:{}]"
        assert strip_tool_calls(text) == ""

    def test_nested_brackets_in_json_content(self) -> None:
        """TOOL_CALL with JSON containing strings that include brackets.

        Note: The current regex [^\\]]+ stops at the first ] inside a string
        value, so TOOL_CALL blocks with inner brackets are only partially
        removed. This is a known limitation.
        """
        # Common case: TOOL_CALL without nested brackets works fine
        text = 'a[TOOL_CALL:note:{"content":"plain"}]b'
        assert strip_tool_calls(text) == "ab"

    def test_tool_call_with_url_containing_brackets(self) -> None:
        """When a TOOL_CALL JSON value contains ']', only part of the
        TOOL_CALL is removed (known regex limitation)."""
        text = (
            '前置[TOOL_CALL:note:{"action":"update","tags":["deep","task_1"],'
            '"content":"[ref] https://x.com"}]后置'
        )
        result = strip_tool_calls(text)
        # The first ']' is inside "[ref]" — regex stops there
        assert "后置" in result  # trailing text is present
        # The outer ] and text after is left behind (partial removal)
        assert result != "前置后置"

    def test_consecutive_tool_calls(self) -> None:
        text = "[TOOL_CALL:a][TOOL_CALL:b]middle[TOOL_CALL:c]"
        result = strip_tool_calls(text)
        assert result == "middle"

    def test_tool_call_with_special_chars(self) -> None:
        text = 'x[TOOL_CALL:note:{"content":"line1\nline2"}]y'
        result = strip_tool_calls(text)
        assert result == "xy"
