from __future__ import annotations

import json
import math
from typing import Any

from app.agent.registry import RegisteredTool, SubagentDefinition, SubagentRegistry, ToolRegistry
from app.agent.react_agent import DietReActAgent
from app.core.llm import OpenAICompatibleClient


class SubagentTool:
    def __init__(
        self,
        definition: SubagentDefinition,
        tool_registry: ToolRegistry,
        llm_client: OpenAICompatibleClient,
    ) -> None:
        self.definition = definition
        self.tool_registry = tool_registry
        self.llm_client = llm_client

    async def handle(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._validate_required_arguments(arguments)
        available_tools = self._available_tools()
        available_tool_names = [tool["name"] for tool in available_tools]
        plan = await self._choose_inner_tool(arguments, available_tools)
        action = plan["action"]
        thought = self._normalize_thought(plan.get("thought"))
        action_input = dict(plan["action_input"])
        for key in (
            "message",
            "query",
            "user_id",
            "sources",
            "date",
            "goal",
            "days",
            "meal_time",
            "description",
            "image_analysis",
            "image_url",
            "image_base64",
            "user_goal",
        ):
            if key in arguments:
                action_input[key] = arguments[key]
        action_input.setdefault("query", arguments.get("message") or arguments.get("query") or "")
        if action == "meal_checkin":
            action_input.setdefault("description", arguments.get("description") or arguments.get("message") or arguments.get("query"))

        tool_result = await self.tool_registry.call(
            action, action_input, enabled=available_tool_names
        )
        observation = DietReActAgent._normalize_observation(action, tool_result)
        return {
            "subagent": self.definition.name,
            "purpose": self.definition.purpose,
            "delegated_action": action,
            "thought": thought,
            "tool_result": tool_result,
            "context": observation["context"],
            "sources": observation["sources"],
            "trace": [f"subagent:{self.definition.name}", *observation["trace"]],
            "rewritten_query": observation["rewritten_query"],
            "metadata_expression": observation["metadata_expression"],
        }

    def _validate_required_arguments(self, arguments: dict[str, Any]) -> None:
        required = self.definition.input_schema.get("required") or []
        properties = self.definition.input_schema.get("properties") or {}
        if not isinstance(required, list):
            raise ValueError("subagent input_schema required must be a list")
        if not isinstance(properties, dict):
            raise ValueError("subagent input_schema properties must be an object")
        for field in required:
            if not isinstance(field, str):
                raise ValueError("subagent input_schema required entries must be text")
            field_name = field.strip()
            if not field_name:
                raise ValueError("subagent input_schema required entries must not be blank")
            value = arguments.get(field_name)
            if value is None:
                raise ValueError(f"{field_name} is required")
            schema = properties.get(field_name)
            schema_type = schema.get("type") if isinstance(schema, dict) else None
            self._validate_required_argument_type(field_name, value, schema_type)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{field_name} is required")

    @staticmethod
    def _validate_required_argument_type(field_name: str, value: object, schema_type: object) -> None:
        if schema_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be text")
            return
        if schema_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
            return
        if schema_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            return
        if schema_type == "array":
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be an array")
            return
        if schema_type == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{field_name} must be an object")
            return
        if schema_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{field_name} must be a boolean")

    async def _choose_inner_tool(
        self, arguments: dict[str, Any], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system_prompt = (
            "你是食韵智析的 Subagent 专家。"
            "只能在你的限定工具集中选择一个工具。"
            "只输出 JSON，字段为 thought、action、action_input。"
        )
        user_prompt = json.dumps(
            {
                "subagent": self.definition.name,
                "purpose": self.definition.purpose,
                "arguments": arguments,
                "tools": tools,
            },
            ensure_ascii=False,
        )
        available_tool_names = [tool["name"] for tool in tools]
        if not available_tool_names:
            raise PermissionError(f"subagent has no registered tools: {self.definition.name}")
        try:
            plan = await self.llm_client.complete_json(
                system_prompt, user_prompt, model="fast", temperature=0.0
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("LLM subagent planning response invalid") from exc
        self._validate_plan(plan, set(available_tool_names))
        return plan

    @staticmethod
    def _validate_plan(plan: dict[str, Any], available_tool_names: set[str]) -> None:
        action = plan.get("action")
        if not isinstance(action, str) or not action.strip():
            raise RuntimeError("LLM subagent planning response invalid: action is required")
        if action not in available_tool_names:
            raise RuntimeError("LLM subagent planning response invalid: action is not available")
        if not isinstance(plan.get("action_input"), dict):
            raise RuntimeError("LLM subagent planning response invalid: action_input must be an object")
        thought = plan.get("thought")
        if thought is not None and not isinstance(thought, str):
            raise RuntimeError("LLM subagent planning response invalid: thought must be text")

    @staticmethod
    def _normalize_thought(value: object) -> str:
        if value is None:
            return "子代理选择限定工具完成任务。"
        if not isinstance(value, str):
            raise RuntimeError("LLM subagent planning response invalid: thought must be text")
        normalized = value.strip()
        return normalized or "子代理选择限定工具完成任务。"

    def _available_tools(self) -> list[dict[str, Any]]:
        listed_tools = {tool["name"]: tool for tool in self.tool_registry.list_tools()}
        return [
            listed_tools[name]
            for name in self.definition.tool_names
            if name in listed_tools
        ]


def build_default_subagent_registry() -> SubagentRegistry:
    registry = SubagentRegistry()
    registry.register(
        SubagentDefinition(
            name="diet_planning_expert",
            purpose="根据用户目标、公共菜谱和个人知识生成饮食计划。",
            tool_names=["diet_plan", "knowledge_base_search"],
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "用户原始饮食计划需求"},
                    "user_id": {"type": "string", "description": "计划所属用户 ID"},
                    "goal": {"type": "string", "description": "饮食目标"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 30},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["recipes", "personal"]},
                    },
                },
                "required": ["message", "user_id"],
            },
        )
    )
    registry.register(
        SubagentDefinition(
            name="meal_record_expert",
            purpose="处理餐食打卡记录，并在需要时检索菜谱知识补全描述。",
            tool_names=["meal_checkin", "food_image_analysis", "knowledge_base_search"],
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "用户原始餐食记录需求"},
                    "user_id": {"type": "string", "description": "打卡所属用户 ID"},
                    "meal_time": {"type": "string", "description": "餐次或时间"},
                    "description": {"type": "string", "description": "餐食描述"},
                    "image_url": {"type": "string", "description": "需要识别的餐食图片 URL"},
                    "image_base64": {"type": "string", "description": "需要识别的餐食图片 base64 内容"},
                    "user_goal": {"type": "string", "description": "用户饮食目标，用于图片营养建议"},
                    "image_analysis": {"type": "object", "description": "视觉识别营养结果"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["recipes", "personal"]},
                    },
                },
                "required": ["message", "user_id", "meal_time"],
            },
        )
    )
    registry.register(
        SubagentDefinition(
            name="nutrition_analysis_expert",
            purpose="基于近期打卡和知识库上下文生成精细化营养分析。",
            tool_names=["nutrition_analysis", "knowledge_base_search"],
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "用户原始营养分析需求"},
                    "user_id": {"type": "string", "description": "要分析的用户 ID"},
                    "date": {"type": "string", "description": "报告日期，格式 YYYY-MM-DD"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["recipes", "personal"]},
                    },
                },
                "required": ["message", "user_id", "date"],
            },
        )
    )
    return registry


def register_subagent_tools(
    tool_registry: ToolRegistry,
    subagent_registry: SubagentRegistry,
    llm_client: OpenAICompatibleClient,
) -> None:
    for subagent in subagent_registry.definitions():
        runtime = SubagentTool(subagent, tool_registry, llm_client)
        tool_registry.register(
            RegisteredTool(
                name=subagent.name,
                description=subagent.purpose,
                provider="subagent",
                handler=runtime.handle,
                input_schema=subagent.input_schema,
            )
        )
