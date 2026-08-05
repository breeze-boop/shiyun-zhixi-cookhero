from __future__ import annotations

from dataclasses import asdict
from typing import Any, Awaitable, Callable

from app.agent.registry import RegisteredTool, ToolRegistry
from app.database.user_repository import UserRepository
from app.services.diet_service import DietService
from app.services.rag_service import RAGService
from app.schemas import VisionAnalyzeResponse


FoodImageAnalyzer = Callable[..., Awaitable[VisionAnalyzeResponse]]


ALLOWED_SOURCES = {"recipes", "personal"}


def _normalize_sources(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("sources must be a list")
    sources: list[str] = []
    seen_sources: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("source selection must be text")
        source = item.strip()
        if source and source not in seen_sources:
            sources.append(source)
            seen_sources.add(source)
    unknown = sorted(set(sources) - ALLOWED_SOURCES)
    if unknown:
        raise ValueError(f"unknown source selection: {', '.join(unknown)}")
    return sources


def _normalize_days(value: Any) -> int:
    if value is None:
        return 7
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("days must be between 1 and 30")
    if value < 1 or value > 30:
        raise ValueError("days must be between 1 and 30")
    return value


def _required_text(arguments: dict[str, Any], field: str) -> str:
    value = arguments.get(field)
    if value is None:
        raise ValueError(f"{field} is required")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    return normalized


def _optional_text(arguments: dict[str, Any], field: str, default: str | None = None) -> str | None:
    value = arguments.get(field)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    normalized = value.strip()
    return normalized or default


def _normalize_optional_text_value(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    normalized = value.strip()
    return normalized or None


def _normalize_image_inputs(arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    normalized_url = _normalize_optional_text_value(arguments.get("image_url"), "image_url")
    normalized_base64 = _normalize_optional_text_value(arguments.get("image_base64"), "image_base64")
    if not normalized_url and not normalized_base64:
        raise ValueError("image_url or image_base64 is required")
    return normalized_url, normalized_base64


def _optional_object(arguments: dict[str, Any], field: str) -> dict[str, Any]:
    value = arguments.get(field)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _optional_image_analysis(arguments: dict[str, Any]) -> dict[str, Any]:
    image_analysis = _optional_object(arguments, "image_analysis")
    if not image_analysis:
        return {}
    try:
        return VisionAnalyzeResponse.model_validate(image_analysis).model_dump()
    except ValueError as exc:
        raise ValueError("image_analysis must match food image analysis schema") from exc


class KnowledgeBaseSearchTool:
    name = "knowledge_base_search"
    description = "检索公共菜谱和个人知识库，返回可供智能体回答的上下文。"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户的菜谱或饮食知识查询"},
            "user_id": {"type": "string", "description": "个人知识库隔离使用的用户 ID"},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["recipes", "personal"]},
                "description": "要检索的数据源，省略时同时检索公共菜谱和个人知识",
            },
        },
        "required": ["query"],
    }

    def __init__(self, rag_service: RAGService) -> None:
        self.rag_service = rag_service

    async def execute(
        self, query: str, user_id: str | None = None, sources: list[str] | None = None
    ) -> dict[str, Any]:
        result = await self.rag_service.retrieve(query=query, user_id=user_id, sources=sources)
        return {
            "context": result.context,
            "sources": [asdict(source) for source in result.sources],
            "trace": result.trace,
            "rewritten_query": result.rewritten_query,
            "metadata_expression": result.metadata_expression,
        }

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.execute(
            query=_required_text(arguments, "query"),
            user_id=_optional_text(arguments, "user_id"),
            sources=_normalize_sources(arguments.get("sources")),
        )


class DietPlanTool:
    name = "diet_plan"
    description = "基于用户目标和可选知识库上下文生成饮食计划。"
    input_schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "计划所属用户 ID"},
            "goal": {"type": "string", "description": "饮食目标，例如减脂高蛋白、控糖"},
            "days": {"type": "integer", "minimum": 1, "maximum": 30, "description": "计划天数"},
            "context_query": {"type": "string", "description": "用于检索菜谱上下文的查询"},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["recipes", "personal"]},
            },
        },
        "required": ["user_id", "goal"],
    }

    def __init__(self, rag_service: RAGService, diet_service: DietService) -> None:
        self.rag_service = rag_service
        self.diet_service = diet_service

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        user_id = _required_text(arguments, "user_id")
        goal = _required_text(arguments, "goal")
        days = _normalize_days(arguments.get("days"))
        context_query = _optional_text(arguments, "context_query", goal) or goal
        sources = _normalize_sources(arguments.get("sources"))
        retrieval = await self.rag_service.retrieve(
            query=context_query,
            user_id=user_id,
            sources=["recipes", "personal"] if sources is None else sources,
        )
        plan = await self.diet_service.create_plan(
            user_id=user_id, goal=goal, days=days, context=retrieval.context
        )
        return {
            **asdict(plan),
            "context": plan.content,
            "sources": [asdict(source) for source in retrieval.sources],
            "trace": [*retrieval.trace, "tool:diet_plan"],
            "rewritten_query": retrieval.rewritten_query,
            "metadata_expression": retrieval.metadata_expression,
        }


class FoodImageAnalysisTool:
    name = "food_image_analysis"
    description = "通过 ModelScope 视觉模型识别食物图片，返回菜名、食材、营养估计和建议。"
    input_schema = {
        "type": "object",
        "properties": {
            "image_url": {"type": "string", "description": "可公开访问的食物图片 URL"},
            "image_base64": {"type": "string", "description": "食物图片 base64 内容"},
            "user_goal": {"type": "string", "description": "用户饮食目标，用于生成建议"},
        },
        "anyOf": [{"required": ["image_url"]}, {"required": ["image_base64"]}],
    }

    def __init__(self, analyzer: FoodImageAnalyzer) -> None:
        self.analyzer = analyzer

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        image_url, image_base64 = _normalize_image_inputs(arguments)
        result = await self.analyzer(
            image_url=image_url,
            image_base64=image_base64,
            user_goal=_optional_text(arguments, "user_goal"),
        )
        payload = result.model_dump()
        return {
            **payload,
            "context": f"{result.dish_name}：" + "、".join(result.ingredients),
            "sources": [],
            "trace": ["tool:food_image_analysis"],
            "rewritten_query": result.dish_name,
            "metadata_expression": None,
        }


class MealCheckinTool:
    name = "meal_checkin"
    description = "保存用户餐食打卡记录和图片营养识别结果。"
    input_schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "打卡所属用户 ID"},
            "meal_time": {"type": "string", "description": "餐次或时间，例如 breakfast/lunch/dinner"},
            "description": {"type": "string", "description": "餐食文字描述"},
            "image_url": {"type": "string", "description": "需要识别的餐食图片 URL"},
            "image_base64": {"type": "string", "description": "需要识别的餐食图片 base64 内容"},
            "user_goal": {"type": "string", "description": "用户饮食目标，用于图片营养建议"},
            "image_analysis": {"type": "object", "description": "视觉识别得到的营养结构化结果"},
        },
        "required": ["user_id", "meal_time", "description"],
    }

    def __init__(self, repository: UserRepository, food_image_analyzer: FoodImageAnalyzer | None = None) -> None:
        self.repository = repository
        self.food_image_analyzer = food_image_analyzer

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        image_analysis = _optional_image_analysis(arguments)
        has_image_input = "image_url" in arguments or "image_base64" in arguments
        if not image_analysis and self.food_image_analyzer and has_image_input:
            image_url, image_base64 = _normalize_image_inputs(arguments)
            analysis = await self.food_image_analyzer(
                image_url=image_url,
                image_base64=image_base64,
                user_goal=_optional_text(arguments, "user_goal"),
            )
            image_analysis = analysis.model_dump()
        checkin = self.repository.create_meal_checkin(
            user_id=_required_text(arguments, "user_id"),
            meal_time=_required_text(arguments, "meal_time"),
            description=_required_text(arguments, "description"),
            image_analysis=image_analysis,
        )
        return {
            **asdict(checkin),
            "context": checkin.description,
            "sources": [],
            "trace": ["tool:meal_checkin"],
            "rewritten_query": checkin.description,
            "metadata_expression": None,
        }


class NutritionAnalysisTool:
    name = "nutrition_analysis"
    description = "基于近期餐食打卡生成营养分析报告。"
    input_schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "要分析的用户 ID"},
            "date": {"type": "string", "description": "报告日期，格式 YYYY-MM-DD"},
        },
        "required": ["user_id", "date"],
    }

    def __init__(self, diet_service: DietService) -> None:
        self.diet_service = diet_service

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        report = await self.diet_service.analyze_nutrition(
            user_id=_required_text(arguments, "user_id"),
            date=_required_text(arguments, "date"),
        )
        return {
            **asdict(report),
            "context": report.content,
            "sources": [],
            "trace": ["tool:nutrition_analysis"],
            "rewritten_query": report.date,
            "metadata_expression": None,
        }


def build_default_tool_registry(
    rag_service: RAGService,
    diet_service: DietService,
    user_repository: UserRepository,
    food_image_analyzer: FoodImageAnalyzer | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    search = KnowledgeBaseSearchTool(rag_service)
    diet_plan = DietPlanTool(rag_service, diet_service)
    meal_checkin = MealCheckinTool(user_repository, food_image_analyzer)
    nutrition_analysis = NutritionAnalysisTool(diet_service)
    food_image_analysis = FoodImageAnalysisTool(food_image_analyzer) if food_image_analyzer else None
    registry.register(
        RegisteredTool(search.name, search.description, "local", search.handle, search.input_schema)
    )
    registry.register(
        RegisteredTool(diet_plan.name, diet_plan.description, "local", diet_plan.handle, diet_plan.input_schema)
    )
    registry.register(
        RegisteredTool(meal_checkin.name, meal_checkin.description, "local", meal_checkin.handle, meal_checkin.input_schema)
    )
    registry.register(
        RegisteredTool(
            nutrition_analysis.name,
            nutrition_analysis.description,
            "local",
            nutrition_analysis.handle,
            nutrition_analysis.input_schema,
        )
    )
    if food_image_analysis:
        registry.register(
            RegisteredTool(
                food_image_analysis.name,
                food_image_analysis.description,
                "local",
                food_image_analysis.handle,
                food_image_analysis.input_schema,
            )
        )
    return registry
