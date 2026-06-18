# Deep Research Assistant

基于 HelloAgents 框架的深度研究助手，支持多 Agent 协作完成网络调研并生成结构化报告。

## 项目结构

```
agentProjet/
├── backend/                    # FastAPI 后端
│   ├── src/
│   │   ├── main.py            # FastAPI 入口，REST & SSE 端点
│   │   ├── agent.py           # DeepResearchAgent 协调器
│   │   ├── config.py          # Pydantic 配置模型
│   │   ├── models.py          # 数据模型（TodoItem, SummaryState）
│   │   ├── prompts.py         # Agent 提示词模板
│   │   ├── utils.py           # 工具函数
│   │   └── services/
│   │       ├── planner.py     # 研究规划 Agent
│   │       ├── search.py      # 搜索调度（Tavily/DuckDuckGo）
│   │       ├── summarizer.py  # 任务总结 Agent
│   │       ├── reporter.py    # 报告撰写 Agent
│   │       ├── tool_events.py # 工具调用事件追踪
│   │       └── notes.py       # 笔记工具
│   └── pyproject.toml
├── frontend/                   # Vue 3 + TypeScript 前端
│   ├── src/
│   │   ├── App.vue            # 主页面组件
│   │   ├── services/api.ts    # SSE 流式 API 封装
│   │   └── main.ts            # 入口
│   └── package.json
└── README.md
```

## 工作流程

```
用户输入研究主题
       │
       ▼
┌─────────────────┐
│  研究规划 Expert  │  → 将主题拆解为 3~5 个待办任务
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  任务执行循环    │ ──▶  │ 网络搜索 Tavily │
│  (并行执行)      │      └──────┬───────┘
└────────┬────────┘             │
         │                      ▼
         │              ┌──────────────┐
         │              │ 任务总结 Expert│ → 写入笔记
         │              └──────────────┘
         ▼
┌─────────────────┐
│  报告撰写 Expert  │  → 综合所有任务笔记生成报告
└─────────────────┘
```

## 技术栈

| 组件       | 技术                                       |
| ---------- | ------------------------------------------ |
| 后端框架   | Python 3.12+, FastAPI, uvicorn             |
| Agent 框架 | HelloAgents 0.2.9                          |
| 大模型     | 支持 OpenAI/Ollama/LMStudio/custom API     |
| 搜索引擎   | Tavily API / DuckDuckGo                    |
| 前端       | Vue 3 + TypeScript + Vite                  |
| 包管理     | uv (后端), npm (前端)                      |

## 快速开始

### 环境要求

- Python >= 3.10
- Node.js >= 18
- uv (推荐) 或 pip

### 1. 后端配置

```bash
cd backend

# 复制环境变量模板并编辑
cp .env.example .env
# 编辑 .env，填入 API Key 和搜索引擎配置

# 安装依赖
uv sync

# 启动服务
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 2. 环境变量说明

```env
# 模型提供者 (ollama / lmstudio / custom)
LLM_PROVIDER=custom

# 模型名称
LLM_MODEL_ID=gpt-4o

# API 密钥
LLM_API_KEY=sk-xxxxx

# 服务地址（custom 模式）
LLM_BASE_URL=https://api.openai.com/v1

# 搜索引擎 (tavily / duckduckgo)
SEARCH_API=tavily
TAVILY_API_KEY=tvly-xxxxx
```

### 3. 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在 `http://localhost:5173`，自动代理 API 请求到 `http://localhost:8000`。

## API 接口

### `POST /research` — 同步研究

```json
{
  "topic": "Python 异步编程",
  "search_api": "tavily"
}
```

### `POST /research/stream` — 流式研究 (SSE)

流式返回各阶段事件：`status` → `todo_list` → `task_status` → `sources` → `task_summary_chunk` → `final_report` → `done`

### `GET /healthz` — 健康检查

## 修复记录

详见 [源代码注释](backend/src/agent.py) 和 [提交历史](https://github.com/qingyvsan/-/commits/main)。
