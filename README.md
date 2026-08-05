# 食韵智析 CookHero

食韵智析是一个基于 LLM 与 Multi-Agent 的 AI 原生健康管理平台。系统围绕饮食计划、打卡记录、营养分析和菜谱问答构建智能体中枢，后端采用 FastAPI，前端采用 React + Vite，知识检索链路按 Advanced RAG 方案实现。

## 核心能力

- ReAct 风格智能体：主控 Agent 通过 Thought → Action → Observation 调用统一 ToolRegistry 中的知识库、饮食计划、餐食记录和营养分析工具。
- 混合 RAG：查询改写、元数据过滤、L1/L2 两级缓存、Dense + BM25 混合检索、Weighted/RRF 融合、Rerank、Parent Document 还原。

- 多源知识：公共菜谱源 `recipes` 与个人知识源 `personal` 共用同一套 Pipeline，并通过 `user_id` 隔离；请求未提供 `user_id` 时会跳过 `personal` 源。
- 工具生态：提供 local / MCP / subagent 统一注册中心，主控 Agent 可调度多种工具，也可调度子代理，并支持按会话限制可用工具。
- 多模态分析：通过 ModelScope 视觉模型接口识别食物图片，输出菜名、食材、营养估计和建议。

- 生产级链路：父文档、饮食计划、餐食打卡和营养分析报告落 PostgreSQL，L1 检索缓存和元数据字典镜像走 Redis，Dense/Sparse 混合检索与 L2 语义缓存走 Milvus，查询改写、Self-Querying 和智能助手回答均调用真实 LLM。



## 目录结构

```text
.
├── backend
│   ├── app
│   │   ├── agent              # ReAct Agent 与 Tool 封装
│   │   ├── core               # 配置
│   │   ├── database           # 文档、饮食计划、打卡与营养报告仓储
│   │   ├── mcp                # MCP Tool Provider 边界
│   │   ├── rag                # RAG Pipeline、缓存、Rerank
│   │   ├── services           # RAG、视觉识别与饮食智能服务编排
│   │   └── main.py            # FastAPI 入口
│   ├── scripts                # HowToCook 同步与解析
│   └── tests
├── frontend
│   └── src                    # React 工作台
└── infra                      # Docker Compose
```



## 本地启动

### 1. 启动基础设施

```bash
docker compose -f infra/docker-compose.yml up -d
```

### 2. 配置后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填写 `LLM_API_KEY`、`EMBEDDING_API_KEY`、`SILICONFLOW_API_KEY` 和 `MODELSCOPE_API_KEY`。`RANKER_STRATEGY` 可设为 `weighted` 或 `rrf`。

### 3. 同步并入库菜谱

```bash
python scripts/sync_data.py --target ../data/HowToCook
python scripts/ingest_howtocook.py --source ../data/HowToCook/dishes
```

默认推荐显式执行 `ingest_howtocook.py` 完成入库。若希望服务启动时自动加载正式 HowToCook 数据，可在 `.env` 中设置 `AUTO_SEED_HOWTOCOOK_DATA=true`；此模式只读取 `data/HowToCook/dishes` 或 `../data/HowToCook/dishes`，不会使用 `data/sample_recipes` 作为生产数据源，缺少正式目录时会启动失败。

### 4. 启动后端

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

问答接口：

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"番茄炒蛋怎么做","sources":["recipes"],"enabled_tools":["knowledge_base_search"]}'
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。


服务启动时会执行 MCP `initialize`、`tools/list`，并把发现的远端工具及其 `inputSchema` 按 `provider="mcp"` 注册到同一个 ToolRegistry，主控 Agent 可在 ReAct 工具规划中看到参数结构并选择调用。

```

## 生产依赖

`infra/docker-compose.yml` 提供 PostgreSQL、Redis、Milvus Standalone 的基础编排：

```bash
docker compose -f infra/docker-compose.yml up -d
```

复制后端环境变量：

```bash
cp backend/.env.example backend/.env
```

按需填写：

- `POSTGRES_DSN`
- `REDIS_URL`
- `MILVUS_URI`
- `LLM_API_KEY`
- `EMBEDDING_API_KEY`
- `SILICONFLOW_API_KEY`
- `RERANK_MIN_SCORE`
- `MODELSCOPE_API_KEY`
- `RANKER_STRATEGY` (`weighted` 或 `rrf`)
- `MCP_SERVERS_CONFIG`（可选，stdio MCP server 配置文件）


