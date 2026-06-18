"""A deep research assistant powered by HelloAgents."""
from dotenv import load_dotenv
from pathlib import Path

# Load .env before ANY other imports that may depend on environment variables.
# This is critical because src/__init__.py is executed before main.py when
# uvicorn does "from src.main import app".
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

__version__ = "0.0.1"

from .agent import DeepResearchAgent
from .config import Configuration, SearchAPI
from .models import SummaryState, SummaryStateInput, SummaryStateOutput, TodoItem

__all__ = [
    "DeepResearchAgent",
    "Configuration",
    "SearchAPI",
    "SummaryState",
    "SummaryStateInput",
    "SummaryStateOutput",
    "TodoItem",
]

