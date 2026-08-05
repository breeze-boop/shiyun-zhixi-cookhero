import pytest

from app.agent.registry import RegisteredTool, ToolRegistry
from app.agent.react_agent import DietReActAgent
from app.schemas import VisionAnalyzeResponse
from app.agent.tools import KnowledgeBaseSearchTool, build_default_tool_registry
from app.services.diet_service import DietService
from scripts.howtocook_loader import HowToCookLoader
from tests.fakes import FakeLLMClient, FakeUserRepository, build_test_rag_service


@pytest.mark.asyncio
async def test_knowledge_base_tool_honors_requested_sources() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader("../data/sample_recipes/dishes").load())

    payload = await KnowledgeBaseSearchTool(service).execute("番茄炒蛋怎么做", sources=["personal"])

    assert payload["sources"] == []
    assert "hybrid:recipes" not in "|".join(payload["trace"])


class NoCallRAGService:
    async def retrieve(self, **_kwargs):
        raise AssertionError("invalid tool text inputs must not call RAG")


@pytest.mark.asyncio
async def test_knowledge_base_tool_rejects_non_string_query_before_rag_call() -> None:
    tool = KnowledgeBaseSearchTool(NoCallRAGService())

    with pytest.raises(ValueError, match="query must be text"):
        await tool.handle({"query": 123})


@pytest.mark.asyncio
async def test_agent_routes_diet_plan_request_through_registered_tool() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=FakeLLMClient())

    result = await agent.run("帮我制定3天减脂高蛋白饮食计划", user_id="u1")

    assert result["action"] == "diet_plan"
    assert result["observation"]["tool_result"]["goal"] == "减脂高蛋白"
    assert result["observation"]["tool_result"]["days"] == 3
    assert user_repository.plans[0].user_id == "u1"


@pytest.mark.asyncio
async def test_agent_rejects_session_with_no_enabled_tools() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=FakeLLMClient())

    with pytest.raises(PermissionError, match="no tools are enabled"):
        await agent.run("帮我制定3天减脂高蛋白饮食计划", user_id="u1", enabled_tools=[])

    assert user_repository.plans == []


class FailingPlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            raise RuntimeError("LLM_API unavailable")
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_agent_rejects_empty_tool_scope_before_llm_planning() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FailingPlanningLLM())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=FailingPlanningLLM())

    with pytest.raises(PermissionError, match="no tools are enabled"):
        await agent.run("帮我制定3天减脂高蛋白饮食计划", user_id="u1", enabled_tools=[])

    assert user_repository.plans == []


@pytest.mark.asyncio
async def test_agent_rejects_unknown_enabled_tool_names() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=FakeLLMClient())

    with pytest.raises(PermissionError, match="unknown enabled tool: typo_tool"):
        await agent.run("番茄炒蛋怎么做", user_id="u1", enabled_tools=["typo_tool"])

    assert user_repository.plans == []


def test_default_local_tools_expose_input_schemas_for_llm_planning() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    tools = {tool["name"]: tool for tool in registry.list_tools()}

    assert tools["knowledge_base_search"]["input_schema"]["required"] == ["query"]
    assert tools["diet_plan"]["input_schema"]["required"] == ["user_id", "goal"]
    assert tools["meal_checkin"]["input_schema"]["required"] == ["user_id", "meal_time", "description"]
    assert tools["nutrition_analysis"]["input_schema"]["required"] == ["user_id", "date"]


async def fake_food_image_analyzer(*, image_url: str | None, image_base64: str | None, user_goal: str | None):
    return VisionAnalyzeResponse(
        dish_name="番茄炒蛋",
        ingredients=["番茄", "鸡蛋"],
        nutrition={"protein": "18g"},
        advice=["少油烹饪"],
        confidence=0.91,
    )


@pytest.mark.asyncio
async def test_food_image_analysis_tool_is_available_when_analyzer_is_configured() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=fake_food_image_analyzer
    )

    tools = {tool["name"]: tool for tool in registry.list_tools()}
    result = await registry.call("food_image_analysis", {"image_url": "https://example.com/meal.jpg", "user_goal": "少油"})
    base64_result = await registry.call("food_image_analysis", {"image_base64": "Zm9vZA==", "user_goal": "少油"})

    schema = tools["food_image_analysis"]["input_schema"]

    assert {item["required"][0] for item in schema["anyOf"]} == {"image_url", "image_base64"}
    assert "required" not in schema
    assert result["dish_name"] == "番茄炒蛋"
    assert base64_result["dish_name"] == "番茄炒蛋"
    assert result["nutrition"] == {"protein": "18g"}


@pytest.mark.asyncio
async def test_food_image_analysis_tool_rejects_missing_image_input() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=fake_food_image_analyzer
    )

    with pytest.raises(ValueError, match="image_url or image_base64 is required"):
        await registry.call("food_image_analysis", {"user_goal": "少油"})


async def fail_food_image_analyzer(*, image_url: str | None, image_base64: str | None, user_goal: str | None):
    raise AssertionError("invalid image inputs must not call ModelScope analyzer")


@pytest.mark.asyncio
async def test_food_image_analysis_tool_rejects_non_string_image_url_before_analysis() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=fail_food_image_analyzer
    )

    with pytest.raises(ValueError, match="image_url must be text"):
        await registry.call("food_image_analysis", {"image_url": 123})


@pytest.mark.asyncio
async def test_meal_checkin_tool_normalizes_image_inputs_before_analysis() -> None:
    captured: dict[str, str | None] = {}

    async def capturing_food_image_analyzer(
        *, image_url: str | None, image_base64: str | None, user_goal: str | None
    ):
        captured.update(
            {
                "image_url": image_url,
                "image_base64": image_base64,
                "user_goal": user_goal,
            }
        )
        return VisionAnalyzeResponse(
            dish_name="番茄炒蛋",
            ingredients=["番茄", "鸡蛋"],
            nutrition={"protein": "18g"},
            advice=["少油烹饪"],
            confidence=0.91,
        )

    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(
        rag_service,
        diet_service,
        user_repository,
        food_image_analyzer=capturing_food_image_analyzer,
    )

    await registry.call(
        "meal_checkin",
        {
            "user_id": "u1",
            "meal_time": "dinner",
            "description": "晚餐图片",
            "image_url": " https://example.com/meal.jpg ",
            "image_base64": "   ",
            "user_goal": " 少油 ",
        },
    )

    assert captured == {
        "image_url": "https://example.com/meal.jpg",
        "image_base64": None,
        "user_goal": "少油",
    }


@pytest.mark.asyncio
async def test_meal_checkin_tool_analyzes_image_before_saving_when_image_is_provided() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=fake_food_image_analyzer
    )

    result = await registry.call(
        "meal_checkin",
        {
            "user_id": "u1",
            "meal_time": "dinner",
            "description": "晚餐图片",
            "image_url": "https://example.com/meal.jpg",
            "user_goal": "少油",
        },
    )

    assert result["image_analysis"]["dish_name"] == "番茄炒蛋"
    assert user_repository.checkins[0].image_analysis["nutrition"] == {"protein": "18g"}


@pytest.mark.asyncio
async def test_agent_preserves_empty_sources_for_knowledge_search() -> None:
    service = build_test_rag_service()
    await service.index_parsed_documents(HowToCookLoader("../data/sample_recipes/dishes").load())
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=FakeLLMClient())

    result = await agent.run(
        "番茄炒蛋怎么做", sources=[], enabled_tools=["knowledge_base_search"]
    )

    assert result["observation"]["sources"] == []
    assert not any(item.startswith("hybrid:") for item in result["observation"]["trace"])


@pytest.mark.asyncio
async def test_local_tools_normalize_whitespace_source_names() -> None:
    rag_service = build_test_rag_service()
    await rag_service.index_parsed_documents(HowToCookLoader("../data/sample_recipes/dishes").load())
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    payload = await registry.call(
        "knowledge_base_search",
        {"query": "番茄炒蛋怎么做", "sources": [" recipes ", "   "]},
    )

    assert payload["sources"][0]["data_source"] == "recipes"
    assert any(item.startswith("hybrid:recipes") for item in payload["trace"])
    assert not any(item.startswith("hybrid:personal") for item in payload["trace"])


@pytest.mark.asyncio
async def test_local_tools_reject_unknown_sources() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    with pytest.raises(ValueError, match="unknown source"):
        await registry.call(
            "knowledge_base_search",
            {"query": "番茄炒蛋", "sources": ["recipes", "web"]},
        )
    with pytest.raises(ValueError, match="unknown source"):
        await registry.call(
            "diet_plan",
            {"user_id": "u1", "goal": "减脂", "sources": ["web"]},
        )


@pytest.mark.asyncio
async def test_local_tools_reject_invalid_source_selection_shape_before_rag_call() -> None:
    tool = KnowledgeBaseSearchTool(NoCallRAGService())

    with pytest.raises(ValueError, match="sources must be a list"):
        await tool.handle({"query": "番茄炒蛋", "sources": "recipes"})

    with pytest.raises(ValueError, match="source selection must be text"):
        await tool.handle({"query": "番茄炒蛋", "sources": ["recipes", 123]})


@pytest.mark.asyncio
async def test_diet_plan_tool_rejects_days_outside_schema_range() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    for invalid_days in (31, True, 1.5, "3"):
        with pytest.raises(ValueError, match="days must be between 1 and 30"):
            await registry.call(
                "diet_plan",
                {"user_id": "u1", "goal": "减脂", "days": invalid_days},
            )

    assert user_repository.plans == []


@pytest.mark.asyncio
async def test_local_tools_reject_missing_required_arguments() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    with pytest.raises(ValueError, match="query is required"):
        await registry.call("knowledge_base_search", {})
    with pytest.raises(ValueError, match="user_id is required"):
        await registry.call("diet_plan", {"goal": "减脂"})
    with pytest.raises(ValueError, match="goal is required"):
        await registry.call("diet_plan", {"user_id": "u1", "days": 3})
    with pytest.raises(ValueError, match="user_id is required"):
        await registry.call("meal_checkin", {"meal_time": "dinner", "description": "晚餐"})
    with pytest.raises(ValueError, match="meal_time is required"):
        await registry.call("meal_checkin", {"user_id": "u1", "description": "晚餐"})
    with pytest.raises(ValueError, match="description is required"):
        await registry.call("meal_checkin", {"user_id": "u1", "meal_time": "dinner"})
    with pytest.raises(ValueError, match="user_id is required"):
        await registry.call("nutrition_analysis", {"date": "2026-07-30"})
    with pytest.raises(ValueError, match="date is required"):
        await registry.call("nutrition_analysis", {"user_id": "u1"})

    assert user_repository.plans == []
    assert user_repository.checkins == []
    assert user_repository.reports == []


class UnknownSourcePlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "尝试使用未知知识源",
                "action": "knowledge_base_search",
                "action_input": {"query": "番茄炒蛋", "sources": ["web"]},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class CrossUserPlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "尝试改写会话上下文",
                "action": "knowledge_base_search",
                "action_input": {
                    "query": "训练日晚餐偏好",
                    "user_id": "u2",
                    "sources": ["personal"],
                },
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class RuntimePlanningFailureLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            raise RuntimeError("LLM_API request failed: network down")
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class MalformedActionInputPlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "返回了非对象参数",
                "action": "knowledge_base_search",
                "action_input": "番茄炒蛋",
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class MalformedThoughtPlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": 123,
                "action": "knowledge_base_search",
                "action_input": {"query": "番茄炒蛋"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_agent_rejects_non_text_planning_thought() -> None:
    async def capture_search(arguments: dict) -> dict:
        raise AssertionError("malformed LLM thought must not reach tool execution")

    registry = ToolRegistry()
    registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", capture_search))
    agent = DietReActAgent(tool_registry=registry, llm_client=MalformedThoughtPlanningLLM())

    with pytest.raises(RuntimeError, match="LLM tool planning response invalid: thought must be text"):
        await agent.run("番茄炒蛋怎么做", user_id="u1", enabled_tools=["knowledge_base_search"])


@pytest.mark.asyncio
async def test_agent_rejects_non_object_action_input_without_default_tool_fallback() -> None:
    async def capture_search(arguments: dict) -> dict:
        raise AssertionError("malformed LLM planning must not call a default tool")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool("knowledge_base_search", "检索知识库", "local", capture_search)
    )
    agent = DietReActAgent(
        tool_registry=registry, llm_client=MalformedActionInputPlanningLLM()
    )

    with pytest.raises(RuntimeError, match="LLM tool planning response invalid"):
        await agent.run(
            "番茄炒蛋怎么做", user_id="u1", enabled_tools=["knowledge_base_search"]
        )


@pytest.mark.asyncio
async def test_agent_rejects_malformed_planning_even_when_another_tool_is_enabled() -> None:
    async def enabled_subagent(arguments: dict) -> dict:
        raise AssertionError("malformed LLM planning must not call an enabled fallback tool")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool("diet_planning_expert", "饮食计划专家", "subagent", enabled_subagent)
    )
    agent = DietReActAgent(
        tool_registry=registry, llm_client=MalformedActionInputPlanningLLM()
    )

    with pytest.raises(RuntimeError, match="LLM tool planning response invalid"):
        await agent.run(
            "帮我制定饮食计划", user_id="u1", enabled_tools=["diet_planning_expert"]
        )


@pytest.mark.asyncio
async def test_agent_does_not_fallback_when_planning_llm_runtime_dependency_fails() -> None:
    async def successful_search(arguments: dict) -> dict:
        return {
            "context": "fallback should not be used",
            "sources": [],
            "trace": ["tool:knowledge_base_search"],
            "rewritten_query": arguments["query"],
            "metadata_expression": None,
        }

    registry = ToolRegistry()
    registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", successful_search))
    agent = DietReActAgent(tool_registry=registry, llm_client=RuntimePlanningFailureLLM())

    with pytest.raises(RuntimeError, match="LLM_API request failed"):
        await agent.run("番茄炒蛋怎么做", user_id="u1")


class DietPlanRuntimeFailureLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "用户需要饮食计划",
                "action": "diet_plan",
                "action_input": {"goal": "减脂高蛋白"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class MCPToolFailureLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "需要调用外部百科 MCP 工具",
                "action": "百科搜索",
                "action_input": {"query": "番茄炒蛋"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_agent_does_not_fallback_when_selected_tool_runtime_dependency_fails() -> None:
    async def successful_search(arguments: dict) -> dict:
        return {
            "context": "fallback should not be used",
            "sources": [],
            "trace": ["tool:knowledge_base_search"],
            "rewritten_query": arguments["query"],
            "metadata_expression": None,
        }

    async def failing_diet_plan(arguments: dict) -> dict:
        raise RuntimeError("Milvus unavailable")

    registry = ToolRegistry()
    registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", successful_search))
    registry.register(RegisteredTool("diet_plan", "饮食计划", "local", failing_diet_plan))
    agent = DietReActAgent(tool_registry=registry, llm_client=DietPlanRuntimeFailureLLM())

    with pytest.raises(RuntimeError, match="Milvus unavailable"):
        await agent.run("帮我制定饮食计划", user_id="u1")


@pytest.mark.asyncio
async def test_agent_wraps_non_runtime_selected_tool_failures_without_fallback() -> None:
    async def successful_search(arguments: dict) -> dict:
        return {
            "context": "fallback should not be used",
            "sources": [],
            "trace": ["tool:knowledge_base_search"],
            "rewritten_query": arguments["query"],
            "metadata_expression": None,
        }

    async def failing_mcp_tool(arguments: dict) -> dict:
        raise ConnectionError("MCP stdio pipe closed")

    registry = ToolRegistry()
    registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", successful_search))
    registry.register(RegisteredTool("百科搜索", "外部百科 MCP", "mcp", failing_mcp_tool))
    agent = DietReActAgent(tool_registry=registry, llm_client=MCPToolFailureLLM())

    with pytest.raises(RuntimeError, match="tool execution failed: 百科搜索") as exc_info:
        await agent.run("查一下番茄炒蛋百科", user_id="u1")
    assert isinstance(exc_info.value.__cause__, ConnectionError)


class ExternalToolObservationLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "工具调度器" in system_prompt:
            return {
                "thought": "需要调用外部 MCP 百科工具",
                "action": "百科搜索",
                "action_input": {"query": "番茄炒蛋"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_agent_normalizes_external_tool_observation_shapes() -> None:
    async def external_tool(arguments: dict) -> dict:
        return {
            "content": [
                {"type": "text", "text": "番茄炒蛋百科第一段"},
                {"type": "text", "text": "番茄炒蛋百科第二段"},
                {"type": "image", "data": "ignored"},
            ],
            "sources": {"title": "不是列表的来源结构"},
            "trace": "mcp:百科搜索",
            "rewritten_query": ["not", "text"],
            "metadata_expression": {"expr": "dish_name == 番茄炒蛋"},
        }

    registry = ToolRegistry()
    registry.register(RegisteredTool("百科搜索", "外部百科 MCP", "mcp", external_tool))
    agent = DietReActAgent(tool_registry=registry, llm_client=ExternalToolObservationLLM())

    result = await agent.run("查一下番茄炒蛋百科", user_id="u1", enabled_tools=["百科搜索"])

    assert result["observation"]["context"] == "番茄炒蛋百科第一段\n番茄炒蛋百科第二段"
    assert result["observation"]["sources"] == []
    assert result["observation"]["trace"] == ["mcp:百科搜索"]
    assert result["observation"]["rewritten_query"] == ""
    assert result["observation"]["metadata_expression"] is None


@pytest.mark.asyncio
async def test_agent_ignores_non_text_mcp_content_blocks() -> None:
    async def external_tool(arguments: dict) -> dict:
        return {
            "content": [
                {"type": "image", "text": "图片替代说明不能作为检索上下文"},
                {"type": "text", "text": "番茄炒蛋百科正文"},
            ],
            "sources": [],
            "trace": ["mcp:百科搜索"],
        }

    registry = ToolRegistry()
    registry.register(RegisteredTool("百科搜索", "外部百科 MCP", "mcp", external_tool))
    agent = DietReActAgent(tool_registry=registry, llm_client=ExternalToolObservationLLM())

    result = await agent.run("查一下番茄炒蛋百科", user_id="u1", enabled_tools=["百科搜索"])

    assert result["observation"]["context"] == "番茄炒蛋百科正文"


@pytest.mark.asyncio
async def test_agent_does_not_fallback_when_tool_arguments_are_invalid() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    agent = DietReActAgent(tool_registry=registry, llm_client=UnknownSourcePlanningLLM())

    with pytest.raises(ValueError, match="unknown source"):
        await agent.run("番茄炒蛋怎么做", user_id="u1")


@pytest.mark.asyncio
async def test_agent_session_context_overrides_llm_tool_input() -> None:
    captured_arguments = {}

    async def capture_tool(arguments: dict) -> dict:
        captured_arguments.update(arguments)
        return {
            "context": "ok",
            "sources": [],
            "trace": ["tool:knowledge_base_search"],
            "rewritten_query": arguments["query"],
            "metadata_expression": None,
        }

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(
            "knowledge_base_search",
            "检索知识库",
            "local",
            capture_tool,
            {"type": "object", "required": ["query"]},
        )
    )
    agent = DietReActAgent(tool_registry=registry, llm_client=CrossUserPlanningLLM())

    await agent.run(
        "训练日晚餐偏好",
        user_id="u1",
        sources=["recipes"],
        enabled_tools=["knowledge_base_search"],
    )

    assert captured_arguments["user_id"] == "u1"
    assert captured_arguments["sources"] == ["recipes"]


@pytest.mark.asyncio
async def test_tool_registry_normalizes_tool_names_and_enabled_scope() -> None:
    captured_arguments = {}

    async def handler(arguments: dict) -> dict:
        captured_arguments.update(arguments)
        return {"content": "ok"}

    registry = ToolRegistry()
    registry.register(RegisteredTool(" echo ", " Echo tool ", "mcp", handler))

    tools = registry.list_tools(enabled=[" echo ", "   "])
    result = await registry.call(" echo ", {"query": "番茄炒蛋"}, enabled=[" echo "])

    assert tools == [
        {
            "name": "echo",
            "description": "Echo tool",
            "provider": "mcp",
            "input_schema": {},
        }
    ]
    assert result == {"content": "ok"}
    assert captured_arguments == {"query": "番茄炒蛋"}


@pytest.mark.asyncio
async def test_tool_registry_rejects_non_list_enabled_scope() -> None:
    async def handler(arguments: dict) -> dict:
        return {"content": "ok"}

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "Echo tool", "mcp", handler))

    with pytest.raises(ValueError, match="enabled tools must be a list"):
        registry.list_tools(enabled="echo")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="enabled tools must be a list"):
        await registry.call("echo", {}, enabled="echo")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_registry_rejects_non_text_tool_names_and_enabled_entries() -> None:
    async def handler(arguments: dict) -> dict:
        return {"content": "ok"}

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="tool name must be text"):
        registry.register(RegisteredTool(123, "Echo tool", "mcp", handler))  # type: ignore[arg-type]

    registry.register(RegisteredTool("echo", "Echo tool", "mcp", handler))

    with pytest.raises(ValueError, match="enabled tool name must be text"):
        registry.list_tools(enabled=[123])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="tool name must be text"):
        await registry.call(123, {}, enabled=["echo"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_registry_rejects_non_object_arguments_before_handler() -> None:
    async def handler(arguments: dict) -> dict:
        raise AssertionError("non-object tool arguments must not reach handler")

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "Echo tool", "mcp", handler))

    with pytest.raises(ValueError, match="tool arguments must be an object"):
        await registry.call("echo", ["query", "番茄炒蛋"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_tool_registry_rejects_non_object_handler_results() -> None:
    async def handler(arguments: dict) -> list[str]:
        return ["not", "an", "object"]

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "Echo tool", "mcp", handler))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tool result must be an object"):
        await registry.call("echo", {"query": "番茄炒蛋"})


@pytest.mark.asyncio
async def test_tool_registry_rejects_duplicate_tool_names() -> None:
    async def local_handler(arguments: dict) -> dict:
        return {"content": "local"}

    async def mcp_handler(arguments: dict) -> dict:
        return {"content": "mcp"}

    registry = ToolRegistry()
    registry.register(RegisteredTool("knowledge_base_search", "local search", "local", local_handler))

    with pytest.raises(ValueError, match="tool already registered: knowledge_base_search"):
        registry.register(RegisteredTool("knowledge_base_search", "mcp search", "mcp", mcp_handler))

    result = await registry.call("knowledge_base_search", {})

    assert result == {"content": "local"}


def test_tool_registry_rejects_invalid_tool_catalog_metadata() -> None:
    async def handler(arguments: dict) -> dict:
        return {"content": "ok"}

    registry = ToolRegistry()

    with pytest.raises(ValueError, match="tool provider must be one of"):
        registry.register(RegisteredTool("web_search", "外部搜索", "remote", handler))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tool description is required"):
        registry.register(RegisteredTool("blank_description", "   ", "mcp", handler))

    with pytest.raises(ValueError, match="tool input_schema must be an object"):
        registry.register(
            RegisteredTool("bad_schema", "坏 schema", "mcp", handler, ["query"])  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="tool input_schema required entries must be text"):
        registry.register(
            RegisteredTool(
                "bad_required_schema",
                "坏 required",
                "mcp",
                handler,
                {"type": "object", "required": ["query", 123]},
            )  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="tool input_schema property names must be text"):
        registry.register(
            RegisteredTool(
                "bad_properties_schema",
                "坏 properties",
                "mcp",
                handler,
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, 123: {"type": "string"}},
                },
            )  # type: ignore[dict-item]
        )

    assert registry.list_tools() == []


@pytest.mark.asyncio
async def test_meal_checkin_tool_rejects_non_object_image_analysis_before_saving() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    with pytest.raises(ValueError, match="image_analysis must be an object"):
        await registry.call(
            "meal_checkin",
            {
                "user_id": "u1",
                "meal_time": "dinner",
                "description": "番茄炒蛋和米饭",
                "image_analysis": "not-an-object",
            },
        )

    assert user_repository.checkins == []


@pytest.mark.asyncio
async def test_meal_checkin_tool_rejects_malformed_image_analysis_before_saving() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    with pytest.raises(ValueError, match="image_analysis"):
        await registry.call(
            "meal_checkin",
            {
                "user_id": "u1",
                "meal_time": "dinner",
                "description": "番茄炒蛋和米饭",
                "image_analysis": {
                    "dish_name": "番茄炒蛋",
                    "ingredients": ["番茄", "鸡蛋"],
                    "nutrition": {"protein": "18g"},
                    "advice": ["少油烹饪"],
                    "confidence": True,
                },
            },
        )

    assert user_repository.checkins == []


@pytest.mark.asyncio
async def test_meal_checkin_tool_strips_text_arguments_before_saving() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    diet_service = DietService(repository=user_repository, llm_client=FakeLLMClient())
    registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    await registry.call(
        "meal_checkin",
        {
            "user_id": " u1 ",
            "meal_time": " dinner ",
            "description": " 番茄炒蛋和米饭 ",
        },
    )

    checkin = user_repository.checkins[0]
    assert checkin.user_id == "u1"
    assert checkin.meal_time == "dinner"
    assert checkin.description == "番茄炒蛋和米饭"
