from __future__ import annotations

import json
from typing import Any

from app.agent.registry import RegisteredTool, ToolRegistry
from app.agent.tools import KnowledgeBaseSearchTool
from app.core.llm import OpenAICompatibleClient


class DietReActAgent:
    """ReAct-style controller for diet planning and nutrition questions."""

    def __init__(
        self, tool_registry: ToolRegistry | KnowledgeBaseSearchTool, llm_client: OpenAICompatibleClient
    ) -> None:
        if isinstance(tool_registry, ToolRegistry):
            self.tool_registry = tool_registry
        else:
            self.tool_registry = ToolRegistry()
            self.tool_registry.register(
                RegisteredTool(tool_registry.name, tool_registry.description, "local", tool_registry.handle)
            )
        self.llm_client = llm_client

    async def run(
        self,
        message: str,
        user_id: str | None = None,
        sources: list[str] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        plan = await self._choose_tool(message, user_id, sources, enabled_tools)
        action = plan["action"]
        thought = self._normalize_thought(
            plan.get("thought"), "根据用户意图选择合适工具并基于证据回答。"
        )
        arguments = dict(plan["action_input"])
        arguments.setdefault("query", message)
        arguments.setdefault("message", message)
        if user_id:
            arguments["user_id"] = user_id
        if sources is not None:
            arguments["sources"] = sources

        try:
            tool_result = await self.tool_registry.call(action, arguments, enabled=enabled_tools)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"tool execution failed: {action}") from exc

        observation = self._normalize_observation(action, tool_result)
        answer = await self._compose_answer(message, observation)
        return {
            "thought": thought,
            "action": action,
            "observation": observation,
            "answer": answer,
        }

    async def _choose_tool(
        self,
        message: str,
        user_id: str | None,
        sources: list[str] | None,
        enabled_tools: list[str] | None,
    ) -> dict[str, Any]:
        tools = self.tool_registry.list_tools(enabled_tools)
        if not tools:
            raise PermissionError("no tools are enabled for this session")
        system_prompt = (
            "你是食韵智析 ReAct 主控 Agent 的工具调度器。"
            "只输出 JSON，字段为 thought、action、action_input。"
            "action 必须来自可用工具列表；不确定时选择 knowledge_base_search。"
        )
        user_prompt = json.dumps(
            {
                "message": message,
                "user_id": user_id,
                "sources": sources,
                "tools": tools,
            },
            ensure_ascii=False,
        )
        try:
            plan = await self.llm_client.complete_json(
                system_prompt, user_prompt, model="fast", temperature=0.0
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("LLM tool planning response invalid") from exc
        self._validate_plan(plan, {tool["name"] for tool in tools})
        return plan

    @staticmethod
    def _validate_plan(plan: dict[str, Any], available_tool_names: set[str]) -> None:
        action = plan.get("action")
        if not isinstance(action, str) or not action.strip():
            raise RuntimeError("LLM tool planning response invalid: action is required")
        if action not in available_tool_names:
            raise RuntimeError("LLM tool planning response invalid: action is not enabled")
        if not isinstance(plan.get("action_input"), dict):
            raise RuntimeError("LLM tool planning response invalid: action_input must be an object")
        thought = plan.get("thought")
        if thought is not None and not isinstance(thought, str):
            raise RuntimeError("LLM tool planning response invalid: thought must be text")

    @staticmethod
    def _normalize_thought(value: object, default: str) -> str:
        if value is None:
            return default
        if not isinstance(value, str):
            raise RuntimeError("LLM tool planning response invalid: thought must be text")
        normalized = value.strip()
        return normalized or default

    @staticmethod
    def _normalize_text(value: object) -> str:
        if isinstance(value, str):
            return value
        return ""

    @classmethod
    def _normalize_context(cls, tool_result: dict[str, Any]) -> str:
        context = tool_result.get("context")
        if isinstance(context, str):
            return context
        content = tool_result.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    block_type = item.get("type")
                    if block_type is None or block_type == "text":
                        parts.append(item["text"])
            return "\n".join(part for part in parts if part.strip())
        return ""

    @staticmethod
    def _normalize_sources(sources: object) -> list[Any]:
        if not isinstance(sources, list):
            return []
        return list(sources)

    @staticmethod
    def _normalize_trace(action: str, trace: object) -> list[str]:
        if isinstance(trace, str):
            normalized = trace.strip()
            return [normalized] if normalized else [f"tool:{action}"]
        if not isinstance(trace, list):
            return [f"tool:{action}"]
        normalized_items = [item.strip() for item in trace if isinstance(item, str) and item.strip()]
        return normalized_items or [f"tool:{action}"]

    @staticmethod
    def _normalize_metadata_expression(metadata_expression: object) -> str | None:
        if metadata_expression is None:
            return None
        if isinstance(metadata_expression, str):
            normalized = metadata_expression.strip()
            return normalized or None
        return None

    @classmethod
    def _normalize_observation(cls, action: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "context": cls._normalize_context(tool_result),
            "sources": cls._normalize_sources(tool_result.get("sources")),
            "trace": cls._normalize_trace(action, tool_result.get("trace")),
            "rewritten_query": cls._normalize_text(tool_result.get("rewritten_query")),
            "metadata_expression": cls._normalize_metadata_expression(tool_result.get("metadata_expression")),
            "tool_result": tool_result,
        }

    async def _compose_answer(self, message: str, observation: dict[str, Any]) -> str:
        system_prompt = (
            "你是食韵智析的饮食健康智能助手。必须基于工具 Observation 和检索上下文回答，"
            "不要编造上下文中不存在的菜谱细节。输出中文，给出步骤、营养注意点和可追问方向。"
        )
        user_prompt = (
            f"用户问题：{message}\n\n"
            f"Observation：\n{json.dumps(observation, ensure_ascii=False)}"
        )
        return await self.llm_client.complete_text(
            system_prompt, user_prompt, model="reasoning", temperature=0.2
        )
