"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response, state.research_topic)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or state.research_topic).strip()

            if not query:
                query = state.research_topic

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""

        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str, research_topic: str = "") -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""

        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if not tasks:
            tool_payload = self._extract_tool_payload(text)
            if tool_payload and isinstance(tool_payload.get("tasks"), list):
                for item in tool_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)

        if not tasks:
            tasks = self._extract_tasks_from_note_calls(text, research_topic)

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """Try to locate and parse a JSON object or array from the text."""

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """Parse the first TOOL_CALL expression in the output."""

        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None

    def _extract_tasks_from_note_calls(
        self,
        text: str,
        research_topic: str,
    ) -> List[dict[str, Any]]:
        """Extract tasks from TOOL_CALL:note patterns when JSON parsing fails.

        The planner agent often records each task via a note tool call like:
          [TOOL_CALL:note:{"action":"create","task_id":1,"title":"任务 1: xxx",
                           "tags":["deep_research","task_1"],"content":"..."}]
        This method parses those calls and reconstructs the task list.
        """
        tasks: List[dict[str, Any]] = []
        seen_ids: set[int] = set()

        for match in TOOL_CALL_PATTERN.finditer(text):
            tool_name = match.group("tool").lower()
            if tool_name != "note":
                continue

            body = match.group("body")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue
            if payload.get("action") not in ("create", "update"):
                continue

            tags = payload.get("tags", [])
            if isinstance(tags, list):
                tag_ids = {t for t in tags if isinstance(t, str) and t.startswith("task_")}
            elif isinstance(tags, str):
                tag_ids = {tags} if tags.startswith("task_") else set()
            else:
                tag_ids = set()

            if not tag_ids:
                continue

            task_id_str = next(iter(tag_ids)).replace("task_", "")
            try:
                task_id = int(task_id_str)
            except ValueError:
                continue

            if task_id in seen_ids:
                continue
            seen_ids.add(task_id)

            title = str(payload.get("title") or f"任务{task_id}").strip()
            intent = str(payload.get("intent") or "聚焦主题的关键问题").strip()

            content = payload.get("content", "")
            query = self._guess_query(content, title, research_topic)

            tasks.append({
                "title": title,
                "intent": intent,
                "query": query,
            })

        tasks.sort(key=lambda t: _extract_task_number(t.get("title", "")))
        return tasks

    @staticmethod
    def _guess_query(content: str, title: str, research_topic: str) -> str:
        """Best-effort query extraction from note content, falling back to title or topic."""
        if not content:
            return f"{research_topic} {title}" if research_topic else title

        content_lower = content.lower()
        for keyword in ("检索方向", "检索关键词", "查询", "query", "搜索"):
            idx = content_lower.find(keyword)
            if idx == -1:
                continue
            snippet = content[idx + len(keyword):].strip().strip("：:，, ")
            if snippet:
                return snippet

        # Last resort: use the first meaningful sentence of content
        cleaned = content.strip().strip("待办任务：:，, ")
        if cleaned:
            return cleaned[:120]
        return f"{research_topic} {title}" if research_topic else title


def _extract_task_number(title: str) -> int:
    """Extract the numeric task number from a title like '任务 1: xxx'."""
    m = re.search(r"任务\s*(\d+)", title)
    return int(m.group(1)) if m else 999
