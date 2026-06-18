"""Shared fixtures for deep researcher tests."""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from src.config import Configuration
from src.models import TodoItem, SummaryStateOutput


@pytest.fixture
def test_config() -> Configuration:
    """Return a minimal Configuration with notes disabled and known defaults."""
    return Configuration(
        enable_notes=False,
        search_api="duckduckgo",
        local_llm="test-model",
        notes_workspace="./test_notes",
    )


@pytest.fixture
def sample_todo_item() -> TodoItem:
    """Return a minimal TodoItem fixture."""
    return TodoItem(
        id=1,
        title="测试任务",
        intent="测试意图",
        query="测试查询",
    )


@pytest.fixture
def sample_search_result() -> dict[str, Any]:
    """Return a realistic search result dict."""
    return {
        "results": [
            {
                "title": "Source Alpha",
                "url": "https://example.com/alpha",
                "content": "Alpha content body",
                "raw_content": "Alpha full page content here with more detail.",
            },
            {
                "title": "Source Beta",
                "url": "https://example.com/beta",
                "content": "Beta content body",
                "raw_content": None,
            },
        ],
        "backend": "duckduckgo",
        "answer": None,
    }


# ---------------------------------------------------------------------------
# Mock fixtures for Phase 2 (Service Layer)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agent(mocker: pytest.MockFixture) -> MagicMock:
    """Mock a ToolAwareSimpleAgent with configurable run/stream_run returns.

    Use ``mock_agent.run.return_value = "..."`` in tests to control output.
    """
    agent = mocker.MagicMock(name="ToolAwareSimpleAgent")
    agent.run.return_value = "mocked response"
    agent.stream_run.return_value = iter(["chunk1", "chunk2"])
    agent.clear_history.return_value = None
    return agent


@pytest.fixture
def mock_search_tool(mocker: pytest.MockFixture) -> MagicMock:
    """Patch the module-level ``_GLOBAL_SEARCH_TOOL`` singleton."""
    return mocker.patch("src.services.search._GLOBAL_SEARCH_TOOL", autospec=True)


@pytest.fixture
def mock_agent_factory(
    mock_agent: MagicMock,
) -> Callable[[], MagicMock]:
    """Return a factory callable that yields the same mock agent instance.

    This matches the ``Callable[[], ToolAwareSimpleAgent]`` type expected by
    ``SummarizationService``.
    """
    return lambda: mock_agent


# ---------------------------------------------------------------------------
# Mock fixtures for Phase 3 (Integration Tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_deep_research_instance(
    mocker: pytest.MockFixture, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Patch ``src.main.DeepResearchAgent`` and return the mock instance.

    Sets safe environment variables before the module-level import so that
    the real module-level ``app = create_app()`` side-effect uses harmless
    defaults.
    """
    monkeypatch.setenv("ENABLE_NOTES", "false")
    monkeypatch.setenv("SEARCH_API", "duckduckgo")
    monkeypatch.setenv("LOCAL_LLM", "test-model")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    mock_cls = mocker.patch("src.main.DeepResearchAgent")
    instance: MagicMock = mock_cls.return_value
    instance.run.return_value = SummaryStateOutput(
        running_summary="# Mock Report",
        report_markdown="# Mock Report",
        todo_items=[],
    )
    instance.run_stream.return_value = iter([
        {"type": "status", "message": "初始化研究流程"},
        {"type": "done"},
    ])
    return instance


@pytest.fixture
def client(mock_deep_research_instance: MagicMock) -> TestClient:
    """Return a ``TestClient`` whose ``DeepResearchAgent`` is fully mocked.

    Import of ``create_app`` must happen *after* the patch is installed so
    that the app instance uses the mocked class.
    """
    # Imported here so pytest collection does not trigger the real import
    from fastapi.testclient import TestClient
    from src.main import create_app

    return TestClient(create_app())
