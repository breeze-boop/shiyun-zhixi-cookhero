import pytest

from app.agent.registry import RegisteredTool, SubagentDefinition, SubagentRegistry, ToolRegistry
from app.agent.subagents import SubagentTool, build_default_subagent_registry, register_subagent_tools
from app.schemas import VisionAnalyzeResponse
from app.agent.tools import build_default_tool_registry
from app.services.diet_service import DietService
from tests.fakes import FakeLLMClient, FakeUserRepository, build_test_rag_service


@pytest.mark.asyncio
async def test_subagent_experts_are_registered_as_tools_and_call_limited_toolsets() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FakeLLMClient()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    subagent_registry = build_default_subagent_registry()

    register_subagent_tools(tool_registry, subagent_registry, llm_client)

    subagent_tools = [tool for tool in tool_registry.list_tools() if tool["provider"] == "subagent"]
    assert {tool["name"] for tool in subagent_tools} >= {
        "diet_planning_expert",
        "meal_record_expert",
        "nutrition_analysis_expert",
    }

    result = await tool_registry.call(
        "diet_planning_expert",
        {"message": "帮我制定3天减脂高蛋白饮食计划", "user_id": "u1"},
        enabled=["diet_planning_expert"],
    )

    assert result["subagent"] == "diet_planning_expert"
    assert result["delegated_action"] == "diet_plan"
    assert result["tool_result"]["goal"] == "减脂高蛋白"
    assert user_repository.plans[0].user_id == "u1"


@pytest.mark.asyncio
async def test_agent_can_run_with_only_a_subagent_enabled() -> None:
    from app.agent.react_agent import DietReActAgent

    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FakeLLMClient()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)
    agent = DietReActAgent(tool_registry, llm_client)

    result = await agent.run(
        "帮我制定3天减脂高蛋白饮食计划",
        user_id="u1",
        enabled_tools=["diet_planning_expert"],
    )

    assert result["action"] == "diet_planning_expert"
    assert result["observation"]["tool_result"]["delegated_action"] == "diet_plan"
    assert user_repository.plans[0].user_id == "u1"


def test_subagent_registry_rejects_invalid_catalog_metadata() -> None:
    registry = SubagentRegistry()

    with pytest.raises(ValueError, match="subagent name is required"):
        registry.register(SubagentDefinition("   ", "饮食计划专家", ["diet_plan"]))

    with pytest.raises(ValueError, match="subagent purpose is required"):
        registry.register(SubagentDefinition("diet_planning_expert", "   ", ["diet_plan"]))

    with pytest.raises(ValueError, match="subagent tool_names must be a list"):
        registry.register(
            SubagentDefinition("bad_tools", "坏工具列表", "diet_plan")  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="subagent tool name is required"):
        registry.register(SubagentDefinition("blank_tool", "坏工具名称", ["diet_plan", "   "]))

    with pytest.raises(ValueError, match="subagent input_schema must be an object"):
        registry.register(
            SubagentDefinition("bad_schema", "坏 schema", ["diet_plan"], ["message"])  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="subagent input_schema required entries must be text"):
        registry.register(
            SubagentDefinition(
                "bad_required_schema",
                "坏 required",
                ["diet_plan"],
                {"type": "object", "required": ["message", 123]},
            )  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="subagent input_schema property names must be text"):
        registry.register(
            SubagentDefinition(
                "bad_properties_schema",
                "坏 properties",
                ["diet_plan"],
                {
                    "type": "object",
                    "properties": {"message": {"type": "string"}, 123: {"type": "string"}},
                },
            )  # type: ignore[dict-item]
        )

    with pytest.raises(ValueError, match="subagent name must be text"):
        registry.register(SubagentDefinition(123, "坏名称", ["diet_plan"]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="subagent tool name must be text"):
        registry.register(SubagentDefinition("bad_tool_type", "坏工具名称", ["diet_plan", 123]))  # type: ignore[list-item]

    registry.register(SubagentDefinition("diet_planning_expert", "饮食计划专家", ["diet_plan"]))
    with pytest.raises(ValueError, match="subagent already registered: diet_planning_expert"):
        registry.register(SubagentDefinition(" diet_planning_expert ", "重复专家", ["diet_plan"]))

    assert [item.name for item in registry.definitions()] == ["diet_planning_expert"]


def test_subagent_tools_expose_input_schemas_for_llm_planning() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FakeLLMClient()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)

    tools = {tool["name"]: tool for tool in tool_registry.list_tools()}
    assert tools["diet_planning_expert"]["input_schema"]["required"] == ["message", "user_id"]
    assert "goal" in tools["diet_planning_expert"]["input_schema"]["properties"]
    assert tools["meal_record_expert"]["input_schema"]["required"] == ["message", "user_id", "meal_time"]
    assert tools["nutrition_analysis_expert"]["input_schema"]["required"] == ["message", "user_id", "date"]


@pytest.mark.asyncio
async def test_subagent_tools_reject_schema_required_arguments() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FakeLLMClient()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)

    for name in ["diet_planning_expert", "meal_record_expert", "nutrition_analysis_expert"]:
        with pytest.raises(ValueError, match="message is required"):
            await tool_registry.call(name, {}, enabled=[name])

    with pytest.raises(ValueError, match="user_id is required"):
        await tool_registry.call(
            "diet_planning_expert",
            {"message": "帮我制定饮食计划"},
            enabled=["diet_planning_expert"],
        )
    with pytest.raises(ValueError, match="meal_time is required"):
        await tool_registry.call(
            "meal_record_expert",
            {"message": "记录晚餐", "user_id": "u1"},
            enabled=["meal_record_expert"],
        )
    with pytest.raises(ValueError, match="date is required"):
        await tool_registry.call(
            "nutrition_analysis_expert",
            {"message": "分析营养", "user_id": "u1"},
            enabled=["nutrition_analysis_expert"],
        )

    assert user_repository.plans == []
    assert user_repository.checkins == []
    assert user_repository.reports == []


class FailingSubagentPlanningLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            raise AssertionError("invalid subagent required inputs must not call LLM planning")
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_subagent_tools_reject_non_text_required_arguments_before_llm_planning() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FailingSubagentPlanningLLM()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)

    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)

    with pytest.raises(ValueError, match="message must be text"):
        await tool_registry.call(
            "diet_planning_expert",
            {"message": ["制定计划"], "user_id": "u1"},
            enabled=["diet_planning_expert"],
        )
    with pytest.raises(ValueError, match="user_id must be text"):
        await tool_registry.call(
            "diet_planning_expert",
            {"message": "制定计划", "user_id": 123},
            enabled=["diet_planning_expert"],
        )
    with pytest.raises(ValueError, match="meal_time must be text"):
        await tool_registry.call(
            "meal_record_expert",
            {"message": "记录晚餐", "user_id": "u1", "meal_time": ["dinner"]},
            enabled=["meal_record_expert"],
        )

    assert user_repository.plans == []
    assert user_repository.checkins == []
    assert user_repository.reports == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "schema_type", "bad_value", "error"),
    [
        ("days", "integer", "3", "days must be an integer"),
        ("score", "number", "0.5", "score must be a number"),
        ("sources", "array", "recipes", "sources must be an array"),
        ("image_analysis", "object", [], "image_analysis must be an object"),
        ("confirmed", "boolean", "true", "confirmed must be a boolean"),
    ],
)
async def test_subagent_tools_reject_non_matching_required_argument_types_before_llm_planning(
    field_name: str,
    schema_type: str,
    bad_value: object,
    error: str,
) -> None:
    async def capture_tool(arguments: dict) -> dict:
        raise AssertionError("invalid subagent inputs must not reach tool execution")

    tool_registry = ToolRegistry()
    tool_registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", capture_tool))
    subagent = SubagentTool(
        SubagentDefinition(
            name="strict_subagent",
            purpose="验证 required 参数 schema 类型。",
            tool_names=["knowledge_base_search"],
            input_schema={
                "type": "object",
                "properties": {field_name: {"type": schema_type}},
                "required": [field_name],
            },
        ),
        tool_registry,
        FailingSubagentPlanningLLM(),
    )

    with pytest.raises(ValueError, match=error):
        await subagent.handle({field_name: bad_value})


async def fake_food_image_analyzer(*, image_url: str | None, image_base64: str | None, user_goal: str | None):
    return VisionAnalyzeResponse(
        dish_name="番茄炒蛋",
        ingredients=["番茄", "鸡蛋"],
        nutrition={"protein": "18g"},
        advice=["少油烹饪"],
        confidence=0.91,
    )


@pytest.mark.asyncio
async def test_meal_record_subagent_passes_image_fields_to_checkin_tool() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = FakeLLMClient()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(
        rag_service, diet_service, user_repository, food_image_analyzer=fake_food_image_analyzer
    )
    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)

    await tool_registry.call(
        "meal_record_expert",
        {
            "message": "记录这张晚餐图片",
            "user_id": "u1",
            "meal_time": "dinner",
            "image_url": "https://example.com/meal.jpg",
            "user_goal": "少油",
        },
        enabled=["meal_record_expert"],
    )

    assert user_repository.checkins[0].image_analysis["dish_name"] == "番茄炒蛋"


class HallucinatedVisionToolLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            return {
                "thought": "尝试先调用视觉识别",
                "action": "food_image_analysis",
                "action_input": {"image_url": "https://example.com/meal.jpg"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_meal_record_subagent_rejects_unavailable_llm_selected_tool_without_default_fallback() -> None:
    rag_service = build_test_rag_service()
    user_repository = FakeUserRepository()
    llm_client = HallucinatedVisionToolLLM()
    diet_service = DietService(repository=user_repository, llm_client=llm_client)
    tool_registry = build_default_tool_registry(rag_service, diet_service, user_repository)
    register_subagent_tools(tool_registry, build_default_subagent_registry(), llm_client)

    with pytest.raises(RuntimeError, match="LLM subagent planning response invalid"):
        await tool_registry.call(
            "meal_record_expert",
            {
                "message": "记录这张晚餐图片",
                "user_id": "u1",
                "meal_time": "dinner",
                "image_url": "https://example.com/meal.jpg",
            },
            enabled=["meal_record_expert"],
        )

    assert user_repository.checkins == []


def test_meal_record_subagent_schema_exposes_image_inputs() -> None:
    subagent = next(
        item
        for item in build_default_subagent_registry().definitions()
        if item.name == "meal_record_expert"
    )

    assert "image_url" in subagent.input_schema["properties"]
    assert "image_base64" in subagent.input_schema["properties"]
    assert "user_goal" in subagent.input_schema["properties"]


class RuntimeSubagentPlanningFailureLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            raise RuntimeError("LLM_API request failed: network down")
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_subagent_does_not_fallback_when_planning_llm_runtime_dependency_fails() -> None:
    async def successful_search(arguments: dict) -> dict:
        return {
            "context": "fallback should not be used",
            "sources": [],
            "trace": ["tool:knowledge_base_search"],
            "rewritten_query": arguments["query"],
            "metadata_expression": None,
        }

    tool_registry = ToolRegistry()
    tool_registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", successful_search))
    subagent = SubagentTool(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["knowledge_base_search"],
            input_schema={"type": "object", "required": ["message"]},
        ),
        tool_registry,
        RuntimeSubagentPlanningFailureLLM(),
    )

    with pytest.raises(RuntimeError, match="LLM_API request failed"):
        await subagent.handle({"message": "训练日晚餐偏好", "user_id": "u1"})


class MalformedSubagentActionInputLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            return {
                "thought": "返回了非对象参数",
                "action": "knowledge_base_search",
                "action_input": "训练日晚餐偏好",
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


class MalformedSubagentThoughtLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            return {
                "thought": 123,
                "action": "knowledge_base_search",
                "action_input": {"query": "训练日晚餐偏好"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_subagent_rejects_non_text_planning_thought() -> None:
    async def capture_tool(arguments: dict) -> dict:
        raise AssertionError("malformed subagent thought must not reach tool execution")

    tool_registry = ToolRegistry()
    tool_registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", capture_tool))
    subagent = SubagentTool(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["knowledge_base_search"],
            input_schema={"type": "object", "required": ["message"]},
        ),
        tool_registry,
        MalformedSubagentThoughtLLM(),
    )

    with pytest.raises(RuntimeError, match="LLM subagent planning response invalid: thought must be text"):
        await subagent.handle({"message": "训练日晚餐偏好", "user_id": "u1"})


@pytest.mark.asyncio
async def test_subagent_rejects_non_object_action_input_without_default_tool_fallback() -> None:
    async def capture_tool(arguments: dict) -> dict:
        raise AssertionError("malformed subagent LLM planning must not call a default tool")

    tool_registry = ToolRegistry()
    tool_registry.register(
        RegisteredTool("knowledge_base_search", "检索知识库", "local", capture_tool)
    )
    subagent = SubagentTool(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["knowledge_base_search"],
            input_schema={"type": "object", "required": ["message"]},
        ),
        tool_registry,
        MalformedSubagentActionInputLLM(),
    )

    with pytest.raises(RuntimeError, match="LLM subagent planning response invalid"):
        await subagent.handle({"message": "训练日晚餐偏好", "user_id": "u1"})


class ExternalToolSubagentLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            return {
                "thought": "调用限定知识工具",
                "action": "knowledge_base_search",
                "action_input": {"query": "番茄炒蛋"},
            }
        return await super().complete_json(
            system_prompt, user_prompt, model=model, temperature=temperature
        )


@pytest.mark.asyncio
async def test_subagent_normalizes_inner_tool_observation_shapes() -> None:
    async def external_search(arguments: dict) -> dict:
        return {
            "content": [
                {"type": "text", "text": "训练日晚餐偏好第一段"},
                {"type": "text", "text": "训练日晚餐偏好第二段"},
                {"type": "image", "data": "ignored"},
            ],
            "sources": {"title": "不是列表的来源结构"},
            "trace": "tool:knowledge_base_search",
            "rewritten_query": ["not", "text"],
            "metadata_expression": {"expr": "category == 个人饮食"},
        }

    tool_registry = ToolRegistry()
    tool_registry.register(RegisteredTool("knowledge_base_search", "检索知识库", "local", external_search))
    subagent = SubagentTool(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["knowledge_base_search"],
            input_schema={"type": "object", "required": ["message"]},
        ),
        tool_registry,
        ExternalToolSubagentLLM(),
    )

    result = await subagent.handle({"message": "训练日晚餐偏好", "user_id": "u1"})

    assert result["context"] == "训练日晚餐偏好第一段\n训练日晚餐偏好第二段"
    assert result["sources"] == []
    assert result["trace"] == ["subagent:diet_planning_expert", "tool:knowledge_base_search"]
    assert result["rewritten_query"] == ""
    assert result["metadata_expression"] is None


class CrossUserSubagentLLM(FakeLLMClient):
    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str = "fast",
        temperature: float = 0.0,
    ) -> dict:
        if "Subagent 专家" in system_prompt:
            return {
                "thought": "尝试改写外层上下文",
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


@pytest.mark.asyncio
async def test_subagent_outer_context_overrides_llm_tool_input() -> None:
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

    tool_registry = ToolRegistry()
    tool_registry.register(
        RegisteredTool(
            "knowledge_base_search",
            "检索知识库",
            "local",
            capture_tool,
            {"type": "object", "required": ["query"]},
        )
    )
    subagent = SubagentTool(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["knowledge_base_search"],
            input_schema={"type": "object", "required": ["message"]},
        ),
        tool_registry,
        CrossUserSubagentLLM(),
    )

    await subagent.handle({"message": "训练日晚餐偏好", "user_id": "u1", "sources": ["recipes"]})

    assert captured_arguments["user_id"] == "u1"
    assert captured_arguments["sources"] == ["recipes"]

