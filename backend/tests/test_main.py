"""Tests for FastAPI endpoints — health check, research, and streaming."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models import SummaryStateOutput, TodoItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(resp: Any) -> list[dict[str, Any]]:
    """Parse SSE ``text/event-stream`` response into event dicts."""
    events: list[dict[str, Any]] = []
    for block in resp.text.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[6:]))
    return events


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /research
# ---------------------------------------------------------------------------


class TestResearch:
    def test_success(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        task = TodoItem(id=1, title="背景调研", intent="收集背景", query="AI 背景")
        mock_deep_research_instance.run.return_value = SummaryStateOutput(
            running_summary="# Report",
            report_markdown="# Report",
            todo_items=[task],
        )
        resp = client.post("/research", json={"topic": "深度学习"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_markdown"] == "# Report"
        assert len(data["todo_items"]) == 1
        assert data["todo_items"][0]["title"] == "背景调研"

    def test_value_error_returns_400(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        mock_deep_research_instance.run.side_effect = ValueError("bad config")
        resp = client.post("/research", json={"topic": "AI"})
        assert resp.status_code == 400
        assert "bad config" in resp.json()["detail"]

    def test_exception_returns_500(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        mock_deep_research_instance.run.side_effect = RuntimeError("LLM failed")
        resp = client.post("/research", json={"topic": "AI"})
        assert resp.status_code == 500

    def test_todo_items_serialized_correctly(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        task = TodoItem(
            id=1,
            title="背景调研",
            intent="收集背景",
            query="AI history",
            status="completed",
            summary="## 总结",
            sources_summary="* S : url",
            note_id="note_001",
            note_path="/notes/note_001.md",
        )
        mock_deep_research_instance.run.return_value = SummaryStateOutput(
            running_summary="# R",
            report_markdown="# R",
            todo_items=[task],
        )
        resp = client.post("/research", json={"topic": "t"})
        items = resp.json()["todo_items"]
        assert items[0]["id"] == 1
        assert items[0]["title"] == "背景调研"
        assert items[0]["status"] == "completed"
        assert items[0]["note_id"] == "note_001"
        assert items[0]["note_path"] == "/notes/note_001.md"

    def test_empty_topic(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        resp = client.post("/research", json={"topic": ""})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /research/stream
# ---------------------------------------------------------------------------


class TestResearchStream:
    def test_event_sequence(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        mock_deep_research_instance.run_stream.return_value = iter([
            {"type": "status", "message": "init"},
            {"type": "todo_list", "tasks": []},
            {"type": "final_report", "report": "# R"},
            {"type": "done"},
        ])
        resp = client.post("/research/stream", json={"topic": "AI"})
        assert resp.status_code == 200
        events = _parse_sse(resp)
        assert events[0]["type"] == "status"
        assert events[-1]["type"] == "done"

    def test_sse_format(
        self,
        client: TestClient,
    ) -> None:
        resp = client.post("/research/stream", json={"topic": "AI"})
        text = resp.text
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith("data: "):
                payload = line[6:]
                assert json.loads(payload)  # must be valid JSON

    def test_agent_value_error_returns_400(
        self,
        client: TestClient,
        mocker: pytest.MockFixture,
    ) -> None:
        """If ``DeepResearchAgent`` construction raises ``ValueError``,
        the endpoint should return 400 before any streaming begins."""
        mocker.patch("src.main.DeepResearchAgent", side_effect=ValueError("invalid config"))
        resp = client.post("/research/stream", json={"topic": "AI"})
        assert resp.status_code == 400

    def test_stream_error_event(
        self,
        mock_deep_research_instance: MagicMock,
        client: TestClient,
    ) -> None:
        """An exception mid-stream should yield an ``error`` event."""

        def error_stream() -> Any:
            yield {"type": "status", "message": "start"}
            raise RuntimeError("stream failed")

        mock_deep_research_instance.run_stream.return_value = error_stream()
        resp = client.post("/research/stream", json={"topic": "AI"})
        assert resp.status_code == 200  # SSE is still a valid HTTP response
        events = _parse_sse(resp)
        types = {e["type"] for e in events}
        assert "status" in types
        assert "error" in types


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_options_returns_cors_headers(self, client: TestClient) -> None:
        resp = client.options(
            "/research",
            headers={"Origin": "http://example.com"},
        )
        # With allow_credentials=True, the middleware echoes the specific origin
        cors_origin = resp.headers.get("access-control-allow-origin")
        assert cors_origin is not None
        assert cors_origin in ("*", "http://example.com")

    def test_post_has_cors_headers(self, client: TestClient) -> None:
        resp = client.post(
            "/research",
            json={"topic": "t"},
            headers={"Origin": "http://example.com"},
        )
        cors_origin = resp.headers.get("access-control-allow-origin")
        assert cors_origin is not None
        assert cors_origin in ("*", "http://example.com")


# ---------------------------------------------------------------------------
# _mask_secret
# ---------------------------------------------------------------------------


class TestMaskSecret:
    def test_none_returns_unset(self) -> None:
        from src.main import _mask_secret

        assert _mask_secret(None) == "unset"

    def test_empty_string_returns_unset(self) -> None:
        from src.main import _mask_secret

        assert _mask_secret("") == "unset"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("ab", "**"),
            ("abc", "***"),
            ("abcdef", "******"),
            ("a" * 8, "*" * 8),
        ],
    )
    def test_short_value_all_stars(self, value: str, expected: str) -> None:
        from src.main import _mask_secret

        assert _mask_secret(value) == expected

    def test_normal_value(self) -> None:
        from src.main import _mask_secret

        result = _mask_secret("abcdefghijklmnop")
        assert result == "abcd...mnop"
        assert len(result) < len("abcdefghijklmnop")
