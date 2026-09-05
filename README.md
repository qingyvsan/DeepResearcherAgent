# Deep Research Assistant — Multi-Agent Web Research System

> **From a single topic to a structured research report — powered by a collaborative multi-agent pipeline.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.x-4FC08D?logo=vue.js)](https://vuejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A deep research assistant that automatically **plans, searches, summarizes, and synthesizes** research on any topic. Built on a multi-agent architecture where specialized agents collaborate to produce structured, well-sourced reports — all streamed in real-time to the browser.

---

## 🎯 What It Does

```
User: "What's the state of multi-agent systems in 2025?"
                    │
                    ▼
    ┌──────────────────────────────────────┐
    │         PLANNING EXPERT              │
    │  Decompose topic into 3-5 sub-tasks  │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│Search  │   │Search  │   │Search  │   │Search  │
│Task 1  │   │Task 2  │   │Task 3  │   │Task 4  │
│(Tavily)│   │(Tavily)│   │(Tavily)│   │(Tavily)│
└───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
    │            │            │            │
    ▼            ▼            ▼            ▼
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│Summary │   │Summary │   │Summary │   │Summary │
│Expert  │   │Expert  │   │Expert  │   │Expert  │
└───┬────┘   └───┬────┘   └───┬────┘   └───┬────┘
    └──────────────┼──────────────┴──────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │        REPORT WRITING EXPERT         │
    │  Synthesize all notes → structured   │
    │  report with citations               │
    └──────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Multi-Agent Collaboration Pattern

The system uses a **Supervisor-Worker** pattern with 4 specialized agents:

| Agent               | Role                      | Key Capability                                |
| ------------------- | ------------------------- | --------------------------------------------- |
| **Planning Expert** | Decompose complex topics  | Generates 3-5 focused sub-tasks               |
| **Search Agent**    | Web information retrieval | Tavily API / DuckDuckGo fallback              |
| **Summary Expert**  | Extract key insights      | Distills search results into structured notes |
| **Report Writer**   | Synthesize final output   | Merges all notes with citations               |

### Real-Time Streaming (SSE)

The entire research pipeline streams events to the frontend via **Server-Sent Events**:

```
status → todo_list → task_status → sources → 
task_summary_chunk → final_report → done
```

Users see the research unfold in real-time — not a loading spinner.

---

## 🔑 Key Features

### 1. Intelligent Topic Decomposition

The Planning Expert doesn't just split by keywords — it reasons about the research domain:

- Identifies subtopics, competing perspectives, and knowledge gaps
- Structures tasks for maximum coverage
- Avoids overlapping research areas

### 2. Multi-Engine Search with Fallback

```
Primary: Tavily API (AI-optimized search)
   │
   └── Fallback: DuckDuckGo (no API key required)
```

### 3. Tool Event Tracking

Every agent action is tracked as a tool event — providing full observability:

- What was searched
- Which sources were retrieved
- What notes were created
- Full audit trail for the final report

### 4. Model Provider Flexibility

Supports multiple LLM backends through a unified interface:

| Provider   | Setup                                |
| ---------- | ------------------------------------ |
| OpenAI     | Set `LLM_PROVIDER=openai` + API key  |
| Ollama     | Set `LLM_PROVIDER=ollama` (local)    |
| LM Studio  | Set `LLM_PROVIDER=lmstudio` (local)  |
| Custom API | Set `LLM_PROVIDER=custom` + base URL |

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Backend

```bash
cd backend
cp .env.example .env
# Edit .env:
#   LLM_PROVIDER=openai
#   LLM_MODEL_ID=gpt-4o
#   LLM_API_KEY=sk-...
#   SEARCH_API=tavily
#   TAVILY_API_KEY=tvly-...

uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### API Endpoints

| Method | Path               | Description                                |
| ------ | ------------------ | ------------------------------------------ |
| `POST` | `/research`        | Synchronous research (returns full report) |
| `POST` | `/research/stream` | Streaming research (SSE events)            |
| `GET`  | `/healthz`         | Health check                               |

```bash
# Sync research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Latest advances in RAG systems", "search_api": "tavily"}'

# Streaming research
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "Multi-agent collaboration patterns", "search_api": "tavily"}'
```

---

## 📁 Project Structure

```
agentProjet/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI entry point (REST + SSE)
│   │   ├── agent.py             # DeepResearchAgent coordinator
│   │   ├── config.py            # Pydantic settings model
│   │   ├── models.py            # Data models (TodoItem, SummaryState)
│   │   ├── prompts.py           # Agent prompt templates
│   │   ├── utils.py             # Utility functions
│   │   └── services/
│   │       ├── planner.py       # Research planning agent
│   │       ├── search.py        # Search dispatch (Tavily/DuckDuckGo)
│   │       ├── summarizer.py    # Task summarization agent
│   │       ├── reporter.py      # Report writing agent
│   │       ├── tool_events.py   # Tool call event tracking
│   │       └── notes.py         # Notes tool
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.vue              # Main page component
│   │   ├── services/api.ts      # SSE streaming API client
│   │   └── main.ts              # Entry point
│   └── package.json
└── README.md
```

---

## 🛠️ Tech Stack

| Component           | Technology                  | Purpose                    |
| ------------------- | --------------------------- | -------------------------- |
| **Agent Framework** | HelloAgents 0.2.9           | Multi-agent orchestration  |
| **Backend**         | FastAPI + uvicorn           | Async REST + SSE streaming |
| **LLM**             | OpenAI / Ollama / LM Studio | Multi-provider support     |
| **Search**          | Tavily API + DuckDuckGo     | Web search with fallback   |
| **Frontend**        | Vue 3 + TypeScript + Vite   | Reactive streaming UI      |
| **Package Mgmt**    | uv (Python) + npm (Node)    | Fast, reproducible builds  |

---

## 🎓 Design Decisions

### Why SSE instead of WebSocket?

Research is a **one-directional** data flow: server → client. SSE is simpler than WebSocket (no handshake upgrade, no bidirectional protocol overhead), works through all proxies, and auto-reconnects on drop. WebSocket would add complexity for no benefit.

### Why HelloAgents framework?

Lightweight, Python-native, and designed for the exact multi-agent pattern used here (plan → execute → summarize → report). Compared to LangChain/LangGraph, HelloAgents has significantly less abstraction overhead.

### Why separate Summary Expert per task?

Having a dedicated summarizer for each search task (rather than one global summarizer) means each task produces **focused, context-rich notes**. The Report Writer then synthesizes notes that already have domain-specific structure.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

*Research shouldn't be manual. Let agents do the heavy lifting while you focus on insights.*
