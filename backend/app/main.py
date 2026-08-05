from __future__ import annotations

from contextlib import asynccontextmanager
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid5, UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent.react_agent import DietReActAgent
from app.agent.registry import RegisteredTool, SubagentRegistry, ToolRegistry
from app.agent.subagents import build_default_subagent_registry, register_subagent_tools
from app.agent.tools import build_default_tool_registry
from app.core.config import Settings, get_settings
from app.core.llm import OpenAICompatibleClient
from app.database.document_repository import PostgresDocumentRepository, RedisMetadataDictionaryCache
from app.database.user_repository import UserRepository
from app.mcp.client import MCPToolProvider
from app.rag.cache.backends import MilvusVectorCache, RedisKeywordCache
from app.rag.cache.cache_manager import CacheManager
from app.rag.pipeline.retrieval import MilvusHybridRetriever
from app.rag.rerankers.siliconflow_reranker import SiliconFlowReranker
from app.rag.document import ParsedDocument
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DietPlanRequest,
    DietPlanResponse,
    MealCheckinRequest,
    MealCheckinResponse,
    NutritionAnalysisRequest,
    NutritionAnalysisResponse,
    PersonalDocumentRequest,
    PersonalDocumentResponse,
    SourceOut,
    ToolOut,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
)
from app.services.diet_service import DietService
from app.services.vision_service import ModelScopeVisionService
from app.services.rag_service import RAGService
from scripts.howtocook_loader import HowToCookLoader


@dataclass(slots=True)
class ApplicationState:
    settings: Settings
    llm_client: OpenAICompatibleClient
    repository: PostgresDocumentRepository
    user_repository: UserRepository
    rag_service: RAGService
    diet_service: DietService
    tool_registry: ToolRegistry
    subagent_registry: SubagentRegistry
    mcp_provider: MCPToolProvider
    agent: DietReActAgent


def build_application_state(settings: Settings) -> ApplicationState:
    llm_client = OpenAICompatibleClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        fast_model=settings.llm_model_fast,
        reasoning_model=settings.llm_model_reasoning,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
        embedding_model=settings.embedding_model,
    )
    metadata_cache = RedisMetadataDictionaryCache(settings.redis_url) if settings.cache_enabled else None
    repository = PostgresDocumentRepository(settings.postgres_dsn, metadata_cache=metadata_cache)
    user_repository = UserRepository(settings.postgres_dsn)
    retrieval = MilvusHybridRetriever(
        milvus_uri=settings.milvus_uri,
        embedding_client=llm_client,
        embedding_dim=settings.embedding_dim,
        collection_by_source={
            "recipes": settings.recipes_collection,
            "personal": settings.personal_collection,
        },
        ranker_strategy=settings.ranker_strategy,
        rrf_k=settings.rrf_k,
    )
    cache_manager = None
    if settings.cache_enabled:
        keyword_backend = RedisKeywordCache(settings.redis_url)
        vector_backend = (
            MilvusVectorCache(
                milvus_uri=settings.milvus_uri,
                collection_name=settings.retrieval_cache_collection,
                embedding_client=llm_client,
                embedding_dim=settings.embedding_dim,
            )
            if settings.cache_l2_enabled
            else None
        )
        cache_manager = CacheManager(
            keyword_backend=keyword_backend,
            vector_backend=vector_backend,
            ttl_seconds=settings.cache_ttl_seconds,
            l2_threshold=settings.l2_similarity_threshold,
        )
    reranker = SiliconFlowReranker(
        api_key=settings.siliconflow_api_key,
        model=settings.siliconflow_rerank_model,
        enabled=settings.rerank_enabled,
        min_score=settings.rerank_min_score,
    )
    rag_service = RAGService(
        settings=settings,
        repository=repository,
        llm_client=llm_client,
        retrieval=retrieval,
        cache_manager=cache_manager,
        reranker=reranker,
    )
    diet_service = DietService(user_repository, llm_client)
    food_image_analyzer = None
    if settings.modelscope_api_key:

        async def food_image_analyzer(*, image_url: str | None, image_base64: str | None, user_goal: str | None):
            return await ModelScopeVisionService(settings).analyze_food(
                image_url=image_url, image_base64=image_base64, user_goal=user_goal
            )

    tool_registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=food_image_analyzer
    )
    subagent_registry = build_default_subagent_registry()
    register_subagent_tools(tool_registry, subagent_registry, llm_client)
    mcp_provider = MCPToolProvider()
    agent = DietReActAgent(tool_registry, llm_client)
    return ApplicationState(
        settings,
        llm_client,
        repository,
        user_repository,
        rag_service,
        diet_service,
        tool_registry,
        subagent_registry,
        mcp_provider,
        agent,
    )


def _normalize_mcp_config_text(value: object, field_name: str, server_name: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError(
            f"MCP_SERVERS_CONFIG invalid: server {server_name} {field_name} must be a string"
        )
    normalized = value.strip()
    if not normalized:
        raise RuntimeError(
            f"MCP_SERVERS_CONFIG invalid: server {server_name} {field_name} is required"
        )
    if normalized.startswith("replace-with-"):
        raise RuntimeError(
            f"MCP_SERVERS_CONFIG invalid: server {server_name} {field_name} must not be a placeholder"
        )
    return normalized


async def load_configured_mcp_tools(state: ApplicationState) -> None:
    if not state.settings.mcp_servers_config:
        return
    config_path = Path(state.settings.mcp_servers_config)
    if not config_path.exists():
        raise RuntimeError(f"MCP_SERVERS_CONFIG does not exist: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("MCP_SERVERS_CONFIG invalid: file must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("MCP_SERVERS_CONFIG invalid: root must be an object")
    servers = payload.get("servers", {})
    if not isinstance(servers, dict):
        raise RuntimeError("MCP_SERVERS_CONFIG invalid: servers must be an object")
    for server_name, spec in servers.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"MCP_SERVERS_CONFIG invalid: server {server_name} must be an object")
        command = spec.get("command")
        if command is None:
            raise RuntimeError(f"MCP_SERVERS_CONFIG invalid: server {server_name} command is required")
        command = _normalize_mcp_config_text(command, "command", server_name)
        args = spec.get("args", [])
        if not isinstance(args, list):
            raise RuntimeError(f"MCP_SERVERS_CONFIG invalid: server {server_name} args must be a list")
        normalized_args = [
            _normalize_mcp_config_text(item, f"args[{index}]", server_name)
            for index, item in enumerate(args)
        ]
        env = spec.get("env", {})
        if not isinstance(env, dict):
            raise RuntimeError(f"MCP_SERVERS_CONFIG invalid: server {server_name} env must be an object")
        normalized_env = {}
        for key, value in env.items():
            env_key = _normalize_mcp_config_text(key, "env key", server_name)
            normalized_env[env_key] = _normalize_mcp_config_text(value, f"env.{env_key}", server_name)
        await state.mcp_provider.connect_stdio_server(
            server_name=str(server_name),
            command=command,
            args=normalized_args,
            env=normalized_env,
        )
    for tool in state.mcp_provider.list_tools():
        name = tool["name"]

        async def handler(arguments: dict, *, tool_name: str = name) -> dict:
            return await state.mcp_provider.call(tool_name, arguments)

        state.tool_registry.register(
            RegisteredTool(
                name=name,
                description=tool["description"],
                provider="mcp",
                handler=handler,
                input_schema=dict(tool.get("input_schema") or {}),
            )
        )


async def seed_local_knowledge(state: ApplicationState) -> None:
    candidates = [
        Path("data/HowToCook/dishes"),
        Path("../data/HowToCook/dishes"),
    ]
    source = next((path for path in candidates if path.exists()), None)
    if not source:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"HowToCook dishes directory not found: {searched}")
    await state.rag_service.index_parsed_documents(HowToCookLoader(source).load())


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state = build_application_state(settings)
    state.rag_service.initialize_storage()
    state.user_repository.create_schema()
    if state.rag_service.cache_manager and state.rag_service.cache_manager.vector_backend:
        state.rag_service.cache_manager.vector_backend.create_collection()
    await load_configured_mcp_tools(state)
    if settings.auto_seed_howtocook_data:
        await seed_local_knowledge(state)
    app.state.cookhero = state
    try:
        yield
    finally:
        await state.mcp_provider.close()


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str | int]:
    state: ApplicationState = app.state.cookhero
    return {"status": "ok", "documents": len(state.repository.list_documents())}


@app.get(f"{get_settings().api_prefix}/tools", response_model=list[ToolOut])
async def list_tools() -> list[ToolOut]:
    state: ApplicationState = app.state.cookhero
    return [ToolOut(**tool) for tool in state.tool_registry.list_tools()]


def _normalize_source_score(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _normalize_source_text(value: object, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip()
    return normalized or default


def _first_source_text(item: dict[str, Any], fields: tuple[str, ...], default: str = "") -> str:
    for field in fields:
        value = _normalize_source_text(item.get(field))
        if value:
            return value
    return default


def _normalize_chat_sources(sources: object, action: str) -> list[SourceOut]:
    if not isinstance(sources, list):
        return []
    normalized: list[SourceOut] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        title = _first_source_text(
            item, ("title", "dish_name", "name", "source", "url"), action
        )
        source = _first_source_text(item, ("source", "url", "href"), title)
        score = _normalize_source_score(
            item.get("score") if "score" in item else item.get("rerank_score")
        )
        normalized.append(
            SourceOut(
                title=title,
                dish_name=_normalize_source_text(item.get("dish_name")),
                category=_normalize_source_text(item.get("category")),
                difficulty=_normalize_source_text(item.get("difficulty")),
                source=source,
                score=score,
                data_source=_first_source_text(item, ("data_source", "provider"), action),
            )
        )
    return normalized


@app.post(f"{get_settings().api_prefix}/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    state: ApplicationState = app.state.cookhero
    try:
        agent_result = await state.agent.run(
            payload.message,
            user_id=payload.user_id,
            sources=payload.sources,
            enabled_tools=payload.enabled_tools,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    observation = agent_result["observation"]
    return ChatResponse(
        answer=agent_result["answer"],
        thought=agent_result["thought"],
        action=agent_result["action"],
        observation=observation,
        rewritten_query=observation["rewritten_query"],
        metadata_expression=observation["metadata_expression"],
        sources=_normalize_chat_sources(observation.get("sources"), agent_result["action"]),
        trace=observation["trace"],
        context_preview=observation["context"][:1200],
    )


PERSONAL_NAMESPACE = UUID("fe906966-67cc-4f5b-947e-8cbb0d799ca8")


@app.post(f"{get_settings().api_prefix}/knowledge/personal", response_model=PersonalDocumentResponse)
async def upload_personal_document(payload: PersonalDocumentRequest) -> PersonalDocumentResponse:
    state: ApplicationState = app.state.cookhero
    source = f"personal/{payload.user_id}/{payload.title}.md"
    doc_id = str(uuid5(PERSONAL_NAMESPACE, source))
    document = ParsedDocument(
        doc_id=doc_id,
        dish_name=payload.title,
        category=payload.category,
        difficulty=payload.difficulty,
        content=payload.content,
        source=source,
        data_source="personal",
        source_type="personal",
        user_id=payload.user_id,
        is_dish_index=False,
    )
    try:
        await state.rag_service.index_parsed_documents([document])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PersonalDocumentResponse(doc_id=doc_id, indexed=True)


@app.post(f"{get_settings().api_prefix}/vision/food", response_model=VisionAnalyzeResponse)
async def analyze_food(payload: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    state: ApplicationState = app.state.cookhero
    try:
        service = ModelScopeVisionService(state.settings)
        return await service.analyze_food(
            image_url=payload.image_url,
            image_base64=payload.image_base64,
            user_goal=payload.user_goal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(f"{get_settings().api_prefix}/diet/plans", response_model=DietPlanResponse)
async def create_diet_plan(payload: DietPlanRequest) -> DietPlanResponse:
    state: ApplicationState = app.state.cookhero
    try:
        context = ""
        if payload.context_query:
            retrieval = await state.rag_service.retrieve(
                query=payload.context_query, user_id=payload.user_id, sources=["recipes", "personal"]
            )
            context = retrieval.context
        plan = await state.diet_service.create_plan(
            user_id=payload.user_id, goal=payload.goal, days=payload.days, context=context
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DietPlanResponse(**asdict(plan))


def _required_tool_result_text(result: dict[str, Any], field: str) -> str:
    value = result.get(field)
    if value is None:
        raise RuntimeError(f"meal_checkin tool result invalid: {field} is required")
    if not isinstance(value, str):
        raise RuntimeError(f"meal_checkin tool result invalid: {field} must be text")
    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"meal_checkin tool result invalid: {field} is required")
    return normalized


def _meal_checkin_response_from_tool_result(result: object) -> MealCheckinResponse:
    if not isinstance(result, dict):
        raise RuntimeError("meal_checkin tool result invalid: result must be an object")
    image_analysis = result.get("image_analysis")
    if image_analysis is None:
        normalized_image_analysis: dict[str, Any] = {}
    elif isinstance(image_analysis, dict):
        normalized_image_analysis = dict(image_analysis)
    else:
        raise RuntimeError("meal_checkin tool result invalid: image_analysis must be an object")
    return MealCheckinResponse(
        checkin_id=_required_tool_result_text(result, "checkin_id"),
        user_id=_required_tool_result_text(result, "user_id"),
        meal_time=_required_tool_result_text(result, "meal_time"),
        description=_required_tool_result_text(result, "description"),
        image_analysis=normalized_image_analysis,
    )


@app.post(f"{get_settings().api_prefix}/checkins", response_model=MealCheckinResponse)
async def create_meal_checkin(payload: MealCheckinRequest) -> MealCheckinResponse:
    state: ApplicationState = app.state.cookhero
    try:
        result = await state.tool_registry.call("meal_checkin", payload.model_dump(exclude_none=True))
        return _meal_checkin_response_from_tool_result(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(f"{get_settings().api_prefix}/nutrition/analysis", response_model=NutritionAnalysisResponse)
async def analyze_nutrition(payload: NutritionAnalysisRequest) -> NutritionAnalysisResponse:
    state: ApplicationState = app.state.cookhero
    try:
        report = await state.diet_service.analyze_nutrition(payload.user_id, payload.date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return NutritionAnalysisResponse(**asdict(report))
