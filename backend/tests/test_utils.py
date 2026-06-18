"""Tests for utils.py — standalone pure logic functions."""

from __future__ import annotations

from enum import Enum
from typing import Any

import pytest

from src.utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
    strip_thinking_tokens,
)


# ---------------------------------------------------------------------------
# get_config_value
# ---------------------------------------------------------------------------

class _DummyEnum(Enum):
    FOO = "foo_value"
    BAR = "bar_value"


class TestGetConfigValue:
    def test_str_returns_itself(self) -> None:
        assert get_config_value("hello") == "hello"

    def test_enum_returns_value(self) -> None:
        assert get_config_value(_DummyEnum.FOO) == "foo_value"
        assert get_config_value(_DummyEnum.BAR) == "bar_value"

    def test_int_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            get_config_value(42)

    def test_none_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            get_config_value(None)


# ---------------------------------------------------------------------------
# strip_thinking_tokens
# ---------------------------------------------------------------------------


class TestStripThinkingTokens:
    def test_no_think_tags_returns_unchanged(self) -> None:
        text = "Hello, world."
        assert strip_thinking_tokens(text) == "Hello, world."

    def test_removes_simple_think_block(self) -> None:
        text = "Before<think>internal</think>After"
        assert strip_thinking_tokens(text) == "BeforeAfter"

    def test_removes_block_at_start(self) -> None:
        text = "<think>hidden</think>Visible"
        assert strip_thinking_tokens(text) == "Visible"

    def test_removes_block_at_end(self) -> None:
        text = "Visible<think>hidden</think>"
        assert strip_thinking_tokens(text) == "Visible"

    def test_unclosed_think_tag_kept(self) -> None:
        """An opening <think> without </think> should be left in place."""
        text = "keep<think>no close"
        assert strip_thinking_tokens(text) == "keep<think>no close"

    def test_no_opening_tag_but_closing_kept(self) -> None:
        text = "keep</think>around"
        assert strip_thinking_tokens(text) == "keep</think>around"

    def test_nested_think_tags_removes_only_first_pair(self) -> None:
        """Nested <think> pairs — only the outermost matching pair is removed."""
        text = "<think>a<think>b</think>c</think>"
        result = strip_thinking_tokens(text)
        assert "<think>" not in result, (
            "Expected all <think> tags to be removed, got: %r" % result
        )

    def test_multiple_think_blocks(self) -> None:
        text = "A<think>first</think>B<think>second</think>C"
        assert strip_thinking_tokens(text) == "ABC"

    def test_empty_string(self) -> None:
        assert strip_thinking_tokens("") == ""

    def test_only_think_tags(self) -> None:
        text = "<think>content</think>"
        assert strip_thinking_tokens(text) == ""

    def test_think_with_empty_content(self) -> None:
        text = "a<think></think>b"
        assert strip_thinking_tokens(text) == "ab"


# ---------------------------------------------------------------------------
# deduplicate_and_format_sources
# ---------------------------------------------------------------------------


class TestDeduplicateAndFormatSources:
    EMPTY = ""

    def test_empty_results_returns_empty_string(self) -> None:
        result = deduplicate_and_format_sources({"results": []}, 100)
        assert result == self.EMPTY

    def test_empty_list_returns_empty_string(self) -> None:
        result = deduplicate_and_format_sources([], 100)
        assert result == self.EMPTY

    def test_single_source_in_dict(self, sample_search_result: dict[str, Any]) -> None:
        single = {"results": [sample_search_result["results"][0]]}
        result = deduplicate_and_format_sources(single, 100)
        assert "信息来源: Source Alpha" in result
        assert "URL: https://example.com/alpha" in result
        assert "信息内容: Alpha content body" in result

    def test_single_source_in_list(self, sample_search_result: dict[str, Any]) -> None:
        items = [sample_search_result["results"][0]]
        result = deduplicate_and_format_sources(items, 100)
        assert "信息来源: Source Alpha" in result

    def test_deduplicates_by_url(self, sample_search_result: dict[str, Any]) -> None:
        dup = sample_search_result["results"][0]
        items = {"results": [dup, dup]}
        result = deduplicate_and_format_sources(items, 100)
        assert result.count("信息来源:") == 1

    def test_skips_items_without_url(self) -> None:
        items = {"results": [{"title": "No URL", "content": "x"}]}
        result = deduplicate_and_format_sources(items, 100)
        assert result == self.EMPTY

    def test_fallback_title_from_url(self) -> None:
        items = {"results": [{"url": "https://example.com/no-title", "content": "x"}]}
        result = deduplicate_and_format_sources(items, 100)
        assert "信息来源: https://example.com/no-title" in result

    def test_fetch_full_page_includes_raw_content(
        self, sample_search_result: dict[str, Any]
    ) -> None:
        result = deduplicate_and_format_sources(
            sample_search_result, 1000, fetch_full_page=True
        )
        assert "详细信息内容限制为" in result
        # Alpha has raw_content
        assert "Alpha full page content" in result

    def test_fetch_full_page_skips_none_raw(
        self, sample_search_result: dict[str, Any]
    ) -> None:
        """Sources with raw_content=None should not produce raw_content section."""
        result = deduplicate_and_format_sources(
            sample_search_result, 1000, fetch_full_page=True
        )
        # Beta has raw_content=None — no raw content block expected for it
        # But it should still appear in the formatted output
        assert "信息来源: Source Beta" in result

    def test_truncation_at_token_limit(self) -> None:
        long_raw = "a" * 5000
        items = {
            "results": [
                {
                    "url": "https://example.com/long",
                    "title": "Long",
                    "content": "body",
                    "raw_content": long_raw,
                }
            ]
        }
        # max_tokens_per_source=10 → char_limit = 40
        result = deduplicate_and_format_sources(items, 10, fetch_full_page=True)
        assert "... [truncated]" in result
        # "a" * 40 should appear before truncation marker
        assert "a" * 40 in result

    def test_exactly_at_limit_no_truncation(self) -> None:
        exact = "a" * 40  # 10 tokens * 4 chars
        items = {
            "results": [
                {
                    "url": "https://example.com/exact",
                    "title": "Exact",
                    "content": "body",
                    "raw_content": exact,
                }
            ]
        }
        result = deduplicate_and_format_sources(items, 10, fetch_full_page=True)
        assert "... [truncated]" not in result


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------


class TestFormatSources:
    def test_none_returns_empty(self) -> None:
        assert format_sources(None) == ""

    def test_empty_dict_returns_empty(self) -> None:
        assert format_sources({}) == ""

    def test_no_results_key_returns_empty(self) -> None:
        assert format_sources({"other": 1}) == ""

    def test_skip_item_without_url(self) -> None:
        data = {"results": [{"title": "X"}, {"title": "Y", "url": "https://y.com"}]}
        result = format_sources(data)
        assert "* X :" not in result
        assert "* Y : https://y.com" in result

    def test_skip_item_with_empty_url(self) -> None:
        data = {"results": [{"title": "X", "url": ""}]}
        result = format_sources(data)
        assert result == ""

    def test_uses_title_fallback_to_url(self) -> None:
        data = {"results": [{"url": "https://fallback.com"}]}
        result = format_sources(data)
        assert "* https://fallback.com : https://fallback.com" in result

    def test_full_format(self, sample_search_result: dict[str, Any]) -> None:
        result = format_sources(sample_search_result)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "* Source Alpha : https://example.com/alpha"
        assert lines[1] == "* Source Beta : https://example.com/beta"
