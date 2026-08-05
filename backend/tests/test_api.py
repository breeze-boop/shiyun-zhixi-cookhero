from dataclasses import asdict

import pytest

from app.schemas import SourceOut, VisionAnalyzeResponse
from tests.fakes import build_test_rag_service


@pytest.mark.asyncio
async def test_chat_response_serializes_sources() -> None:
    service = build_test_rag_service()
    from pathlib import Path
    from scripts.howtocook_loader import HowToCookLoader

    await service.index_parsed_documents(HowToCookLoader(Path("../data/sample_recipes/dishes")).load())
    result = await service.retrieve("番茄炒蛋怎么做", sources=["recipes"])

    sources = [SourceOut(**asdict(source)) for source in result.sources]

    assert sources[0].dish_name == "番茄炒蛋"


@pytest.mark.asyncio
async def test_chat_route_passes_enabled_tools_to_agent(monkeypatch) -> None:
    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        def __init__(self) -> None:
            self.kwargs = None

        async def run(self, message, **kwargs):
            self.kwargs = kwargs
            return {
                "thought": "用户需要计划",
                "action": "diet_planning_expert",
                "answer": "ok",
                "observation": {
                    "rewritten_query": "番茄炒蛋",
                    "metadata_expression": None,
                    "sources": [],
                    "trace": ["tool:diet_planning_expert"],
                    "context": "",
                    "tool_result": {"content": "ok"},
                },
            }

    fake_agent = FakeAgent()
    monkeypatch.setattr(app.state, "cookhero", type("State", (), {"agent": fake_agent})(), raising=False)

    response = await chat(ChatRequest(message="制定计划", user_id="u1", enabled_tools=["diet_planning_expert"]))

    assert response.thought == "用户需要计划"
    assert response.action == "diet_planning_expert"
    assert response.observation["trace"] == ["tool:diet_planning_expert"]
    assert fake_agent.kwargs == {
        "user_id": "u1",
        "sources": None,
        "enabled_tools": ["diet_planning_expert"],
    }


@pytest.mark.asyncio
async def test_chat_route_normalizes_external_tool_sources(monkeypatch) -> None:
    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            return {
                "thought": "调用百科 MCP 工具",
                "action": "mcp百科搜索",
                "answer": "外部搜索结果",
                "observation": {
                    "rewritten_query": "番茄炒蛋",
                    "metadata_expression": None,
                    "sources": [
                        {
                            "title": "番茄炒蛋百科",
                            "url": "https://example.com/tomato-egg",
                            "snippet": "外部 MCP 返回的来源结构",
                        }
                    ],
                    "trace": ["tool:mcp百科搜索"],
                    "context": "百科内容",
                    "tool_result": {"content": "百科内容"},
                },
            }

    monkeypatch.setattr(
        app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False
    )

    response = await chat(
        ChatRequest(message="查百科", user_id="u1", enabled_tools=["mcp百科搜索"])
    )

    assert response.sources[0].title == "番茄炒蛋百科"
    assert response.sources[0].source == "https://example.com/tomato-egg"
    assert response.sources[0].data_source == "mcp百科搜索"
    assert response.observation["sources"][0]["snippet"] == "外部 MCP 返回的来源结构"


@pytest.mark.asyncio
async def test_chat_route_normalizes_blank_external_source_fields(monkeypatch) -> None:
    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            return {
                "thought": "调用百科 MCP 工具",
                "action": "mcp百科搜索",
                "answer": "外部搜索结果",
                "observation": {
                    "rewritten_query": "番茄炒蛋",
                    "metadata_expression": None,
                    "sources": [
                        {
                            "title": "   ",
                            "url": "   ",
                            "dish_name": "  ",
                            "category": "  ",
                            "difficulty": "  ",
                            "data_source": "   ",
                        }
                    ],
                    "trace": ["tool:mcp百科搜索"],
                    "context": "百科内容",
                    "tool_result": {"content": "百科内容"},
                },
            }

    monkeypatch.setattr(
        app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False
    )

    response = await chat(
        ChatRequest(message="查百科", user_id="u1", enabled_tools=["mcp百科搜索"])
    )

    assert response.sources[0].title == "mcp百科搜索"
    assert response.sources[0].source == "mcp百科搜索"
    assert response.sources[0].dish_name == ""
    assert response.sources[0].category == ""
    assert response.sources[0].difficulty == ""
    assert response.sources[0].data_source == "mcp百科搜索"


@pytest.mark.asyncio
async def test_chat_route_drops_invalid_external_source_scores(monkeypatch) -> None:
    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            return {
                "thought": "调用百科 MCP 工具",
                "action": "mcp百科搜索",
                "answer": "外部搜索结果",
                "observation": {
                    "rewritten_query": "番茄炒蛋",
                    "metadata_expression": None,
                    "sources": [
                        {
                            "title": "番茄炒蛋百科",
                            "url": "https://example.com/tomato-egg",
                            "score": "high",
                        }
                    ],
                    "trace": ["tool:mcp百科搜索"],
                    "context": "百科内容",
                    "tool_result": {"content": "百科内容"},
                },
            }

    monkeypatch.setattr(
        app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False
    )

    response = await chat(
        ChatRequest(message="查百科", user_id="u1", enabled_tools=["mcp百科搜索"])
    )

    assert response.sources[0].title == "番茄炒蛋百科"
    assert response.sources[0].source == "https://example.com/tomato-egg"
    assert response.sources[0].score is None


@pytest.mark.asyncio
async def test_chat_route_drops_malformed_external_source_items_and_fields(monkeypatch) -> None:
    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            return {
                "thought": "调用百科 MCP 工具",
                "action": "mcp百科搜索",
                "answer": "外部搜索结果",
                "observation": {
                    "rewritten_query": "番茄炒蛋",
                    "metadata_expression": None,
                    "sources": [
                        ["not", "a", "source"],
                        {
                            "title": {"bad": "title"},
                            "url": "https://example.com/tomato-egg",
                            "dish_name": ["bad"],
                            "category": {"bad": "category"},
                            "difficulty": 123,
                            "data_source": {"bad": "provider"},
                        },
                    ],
                    "trace": ["tool:mcp百科搜索"],
                    "context": "百科内容",
                    "tool_result": {"content": "百科内容"},
                },
            }

    monkeypatch.setattr(
        app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False
    )

    response = await chat(
        ChatRequest(message="查百科", user_id="u1", enabled_tools=["mcp百科搜索"])
    )

    assert len(response.sources) == 1
    assert response.sources[0].title == "https://example.com/tomato-egg"
    assert response.sources[0].source == "https://example.com/tomato-egg"
    assert response.sources[0].dish_name == ""
    assert response.sources[0].category == ""
    assert response.sources[0].difficulty == ""
    assert response.sources[0].data_source == "mcp百科搜索"


@pytest.mark.asyncio
async def test_chat_route_returns_400_when_no_tools_are_enabled(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            raise PermissionError("no tools are enabled for this session")

    monkeypatch.setattr(app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await chat(ChatRequest(message="制定计划", user_id="u1", enabled_tools=[]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "no tools are enabled for this session"


@pytest.mark.asyncio
async def test_chat_route_returns_400_when_tool_arguments_are_invalid(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            raise ValueError("unknown source selection: web")

    monkeypatch.setattr(app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await chat(ChatRequest(message="番茄炒蛋", user_id="u1"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "unknown source selection: web"


@pytest.mark.asyncio
async def test_chat_route_returns_503_when_agent_runtime_dependency_fails(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, chat
    from app.schemas import ChatRequest

    class FakeAgent:
        async def run(self, message, **kwargs):
            raise RuntimeError("LLM_API unavailable")

    monkeypatch.setattr(app.state, "cookhero", type("State", (), {"agent": FakeAgent()})(), raising=False)

    with pytest.raises(HTTPException) as exc_info:
        await chat(ChatRequest(message="番茄炒蛋", user_id="u1"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "LLM_API unavailable"


@pytest.mark.asyncio
async def test_diet_plan_route_returns_503_when_retrieval_runtime_dependency_fails(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, create_diet_plan
    from app.schemas import DietPlanRequest

    class FakeRAGService:
        async def retrieve(self, **kwargs):
            raise RuntimeError("Milvus unavailable")

    class FakeDietService:
        async def create_plan(self, **kwargs):
            raise AssertionError("diet service should not run after retrieval failure")

    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"rag_service": FakeRAGService(), "diet_service": FakeDietService()})(),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_diet_plan(
            DietPlanRequest(user_id="u1", goal="减脂高蛋白", days=7, context_query="番茄炒蛋")
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Milvus unavailable"


@pytest.mark.asyncio
async def test_nutrition_route_returns_503_when_analysis_runtime_dependency_fails(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, analyze_nutrition
    from app.schemas import NutritionAnalysisRequest

    class FakeDietService:
        async def analyze_nutrition(self, user_id: str, date: str):
            raise RuntimeError("LLM_API unavailable")

    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"diet_service": FakeDietService()})(),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await analyze_nutrition(NutritionAnalysisRequest(user_id="u1", date="2026-07-30"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "LLM_API unavailable"


@pytest.mark.asyncio
async def test_personal_document_route_returns_503_when_indexing_runtime_dependency_fails(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, upload_personal_document
    from app.schemas import PersonalDocumentRequest

    class FakeRAGService:
        async def index_parsed_documents(self, documents):
            raise RuntimeError("Milvus unavailable")

    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"rag_service": FakeRAGService()})(),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await upload_personal_document(
            PersonalDocumentRequest(user_id="u1", title="控糖早餐", content="少糖，增加蛋白质")
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Milvus unavailable"


@pytest.mark.asyncio
async def test_personal_document_route_returns_400_when_indexing_validation_fails(monkeypatch) -> None:
    from fastapi import HTTPException

    from app.main import app, upload_personal_document
    from app.schemas import PersonalDocumentRequest

    class FakeRAGService:
        async def index_parsed_documents(self, documents):
            raise ValueError("personal documents require a non-GLOBAL user_id")

    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"rag_service": FakeRAGService()})(),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await upload_personal_document(
            PersonalDocumentRequest(user_id="GLOBAL", title="控糖早餐", content="少糖，增加蛋白质")
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "personal documents require a non-GLOBAL user_id"


async def fake_food_image_analyzer(*, image_url: str | None, image_base64: str | None, user_goal: str | None):
    return VisionAnalyzeResponse(
        dish_name="番茄炒蛋",
        ingredients=["番茄", "鸡蛋"],
        nutrition={"protein": "18g"},
        advice=["少油烹饪"],
        confidence=0.91,
    )


@pytest.mark.asyncio
async def test_checkin_route_analyzes_image_before_saving(monkeypatch) -> None:
    from app.agent.tools import build_default_tool_registry
    from app.main import app, create_meal_checkin
    from app.schemas import MealCheckinRequest
    from app.services.diet_service import DietService
    from tests.fakes import FakeLLMClient, FakeUserRepository

    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    tool_registry = build_default_tool_registry(
        rag_service,
        diet_service,
        user_repository,
        food_image_analyzer=fake_food_image_analyzer,
    )
    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"tool_registry": tool_registry, "user_repository": user_repository})(),
        raising=False,
    )

    response = await create_meal_checkin(
        MealCheckinRequest(
            user_id="u1",
            meal_time="dinner",
            description="晚餐图片",
            image_url="https://example.com/meal.jpg",
            user_goal="少油",
        )
    )

    assert response.image_analysis["dish_name"] == "番茄炒蛋"
    assert user_repository.checkins[0].image_analysis["nutrition"] == {"protein": "18g"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_result", "detail"),
    [
        (
            {
                "checkin_id": 123,
                "user_id": "u1",
                "meal_time": "dinner",
                "description": "晚餐",
                "image_analysis": {},
            },
            "meal_checkin tool result invalid: checkin_id must be text",
        ),
        (
            {
                "checkin_id": "c1",
                "user_id": 123,
                "meal_time": "dinner",
                "description": "晚餐",
                "image_analysis": {},
            },
            "meal_checkin tool result invalid: user_id must be text",
        ),
        (
            {
                "checkin_id": "c1",
                "user_id": "u1",
                "meal_time": "dinner",
                "description": "晚餐",
                "image_analysis": [],
            },
            "meal_checkin tool result invalid: image_analysis must be an object",
        ),
    ],
)
async def test_checkin_route_rejects_malformed_tool_results(monkeypatch, tool_result, detail) -> None:
    from fastapi import HTTPException

    from app.main import app, create_meal_checkin
    from app.schemas import MealCheckinRequest

    class FakeToolRegistry:
        async def call(self, name, arguments):
            assert name == "meal_checkin"
            return tool_result

    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"tool_registry": FakeToolRegistry()})(),
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_meal_checkin(
            MealCheckinRequest(user_id="u1", meal_time="dinner", description="晚餐")
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == detail


@pytest.mark.asyncio
async def test_tools_route_returns_registered_tool_catalog(monkeypatch) -> None:
    from app.agent.registry import RegisteredTool, ToolRegistry
    from app.main import app, list_tools

    async def handler(arguments: dict) -> dict:
        return {"content": "ok"}

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            name="knowledge_base_search",
            description="检索知识库",
            provider="local",
            handler=handler,
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    )
    registry.register(
        RegisteredTool(
            name="mcp_echo",
            description="MCP echo",
            provider="mcp",
            handler=handler,
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )
    )
    registry.register(
        RegisteredTool(
            name="diet_planning_expert",
            description="饮食计划专家",
            provider="subagent",
            handler=handler,
            input_schema={"type": "object", "required": ["message"]},
        )
    )
    monkeypatch.setattr(
        app.state,
        "cookhero",
        type("State", (), {"tool_registry": registry})(),
        raising=False,
    )

    response = await list_tools()

    tools = {tool.name: tool for tool in response}
    assert tools["knowledge_base_search"].provider == "local"
    assert tools["mcp_echo"].provider == "mcp"
    assert tools["diet_planning_expert"].provider == "subagent"
    assert tools["mcp_echo"].input_schema["properties"]["text"]["type"] == "string"


def test_chat_request_rejects_unknown_sources() -> None:
    from pydantic import ValidationError

    from app.schemas import ChatRequest

    assert ChatRequest(message="番茄炒蛋", sources=[]).sources == []
    with pytest.raises(ValidationError):
        ChatRequest(message="番茄炒蛋", sources=["web"])


def test_api_request_models_reject_blank_required_text() -> None:
    from pydantic import ValidationError

    from app.schemas import (
        ChatRequest,
        DietPlanRequest,
        MealCheckinRequest,
        NutritionAnalysisRequest,
        PersonalDocumentRequest,
    )

    cases = [
        lambda: ChatRequest(message="   "),
        lambda: PersonalDocumentRequest(user_id="u1", title="   ", content="少糖"),
        lambda: PersonalDocumentRequest(user_id="u1", title="控糖早餐", content="   "),
        lambda: DietPlanRequest(user_id="u1", goal="   "),
        lambda: MealCheckinRequest(user_id="u1", meal_time="dinner", description="   "),
        lambda: NutritionAnalysisRequest(user_id="u1", date="   "),
    ]

    for build_request in cases:
        with pytest.raises(ValidationError):
            build_request()


def test_api_request_models_strip_required_text() -> None:
    from app.schemas import ChatRequest, DietPlanRequest, PersonalDocumentRequest

    assert ChatRequest(message="  番茄炒蛋  ").message == "番茄炒蛋"
    assert DietPlanRequest(user_id=" u1 ", goal=" 减脂 ").user_id == "u1"
    assert PersonalDocumentRequest(user_id=" u1 ", title=" 控糖早餐 ", content=" 少糖 ").title == "控糖早餐"


def test_diet_plan_request_rejects_non_integer_days() -> None:
    from pydantic import ValidationError

    from app.schemas import DietPlanRequest

    for invalid_days in (True, 1.5, "3"):
        with pytest.raises(ValidationError):
            DietPlanRequest(user_id="u1", goal="减脂", days=invalid_days)


def test_nutrition_analysis_request_rejects_invalid_dates() -> None:
    from pydantic import ValidationError

    from app.schemas import NutritionAnalysisRequest

    for invalid_date in ("2026/07/30", "2026-7-30", "2026-02-30"):
        with pytest.raises(ValidationError):
            NutritionAnalysisRequest(user_id="u1", date=invalid_date)


def test_vision_analyze_request_requires_image_input() -> None:
    from pydantic import ValidationError

    from app.schemas import VisionAnalyzeRequest

    for payload in ({"user_goal": "少油"}, {"image_url": "   ", "image_base64": "   "}):
        with pytest.raises(ValidationError):
            VisionAnalyzeRequest(**payload)


def test_api_request_models_normalize_optional_text_fields() -> None:
    from app.schemas import ChatRequest, DietPlanRequest, MealCheckinRequest, VisionAnalyzeRequest

    chat = ChatRequest(message="番茄炒蛋", user_id="   ")
    diet = DietPlanRequest(user_id="u1", goal="减脂", context_query="  番茄炒蛋  ")
    blank_diet = DietPlanRequest(user_id="u1", goal="减脂", context_query="   ")
    vision = VisionAnalyzeRequest(image_url="  https://example.com/meal.jpg  ", image_base64="   ", user_goal="  少油  ")
    checkin = MealCheckinRequest(
        user_id="u1",
        meal_time="dinner",
        description="晚餐",
        image_url="   ",
        image_base64=" Zm9vZA== ",
        user_goal="   ",
    )

    assert chat.user_id is None
    assert diet.context_query == "番茄炒蛋"
    assert blank_diet.context_query is None
    assert vision.image_url == "https://example.com/meal.jpg"
    assert vision.image_base64 is None
    assert vision.user_goal == "少油"
    assert checkin.image_url is None
    assert checkin.image_base64 == "Zm9vZA=="
    assert checkin.user_goal is None


def test_personal_document_metadata_rejects_blank_values() -> None:
    from pydantic import ValidationError

    from app.schemas import PersonalDocumentRequest

    with pytest.raises(ValidationError):
        PersonalDocumentRequest(user_id="u1", title="控糖早餐", content="少糖", category="   ")
    with pytest.raises(ValidationError):
        PersonalDocumentRequest(user_id="u1", title="控糖早餐", content="少糖", difficulty="   ")


def test_chat_request_normalizes_source_and_tool_lists() -> None:
    from pydantic import ValidationError

    from app.schemas import ChatRequest

    request = ChatRequest(
        message="番茄炒蛋",
        sources=[" recipes ", "   "],
        enabled_tools=[" knowledge_base_search ", "   "],
    )

    assert request.sources == ["recipes"]
    assert request.enabled_tools == ["knowledge_base_search"]

    assert ChatRequest(message="番茄炒蛋", sources=["   "]).sources == []
    with pytest.raises(ValidationError):
        ChatRequest(message="番茄炒蛋", sources=[" web "])
    with pytest.raises(ValidationError, match="source selection must be text"):
        ChatRequest(message="番茄炒蛋", sources=[123])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="enabled tool name must be text"):
        ChatRequest(message="番茄炒蛋", enabled_tools=[123])  # type: ignore[list-item]
