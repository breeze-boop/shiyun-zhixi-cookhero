from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, cast


ToolProvider = Literal["local", "mcp", "subagent"]
ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _normalize_input_schema_object(input_schema: object, label: str) -> dict[str, Any]:
    if not isinstance(input_schema, dict):
        raise ValueError(f"{label} input_schema must be an object")
    normalized = dict(input_schema)
    required = normalized.get("required")
    if required is not None:
        if not isinstance(required, list):
            raise ValueError(f"{label} input_schema required must be a list")
        for item in required:
            if not isinstance(item, str):
                raise ValueError(f"{label} input_schema required entries must be text")
            if not item.strip():
                raise ValueError(f"{label} input_schema required entries must not be blank")
    properties = normalized.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise ValueError(f"{label} input_schema properties must be an object")
        for name in properties:
            if not isinstance(name, str):
                raise ValueError(f"{label} input_schema property names must be text")
            if not name.strip():
                raise ValueError(f"{label} input_schema property names must not be blank")
    return normalized


@dataclass(slots=True)
class RegisteredTool:
    name: str
    description: str
    provider: ToolProvider
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    @staticmethod
    def _normalize_name(name: object) -> str:
        if not isinstance(name, str):
            raise ValueError("tool name must be text")
        normalized = name.strip()
        if not normalized:
            raise ValueError("tool name is required")
        return normalized

    @staticmethod
    def _normalize_description(description: object) -> str:
        if not isinstance(description, str):
            raise ValueError("tool description is required")
        normalized = description.strip()
        if not normalized:
            raise ValueError("tool description is required")
        return normalized

    @staticmethod
    def _normalize_provider(provider: object) -> ToolProvider:
        if provider not in {"local", "mcp", "subagent"}:
            raise ValueError("tool provider must be one of: local, mcp, subagent")
        return cast(ToolProvider, provider)

    @staticmethod
    def _normalize_input_schema(input_schema: object) -> dict[str, Any]:
        return _normalize_input_schema_object(input_schema, "tool")

    @classmethod
    def _normalize_enabled(cls, enabled: list[str] | None) -> set[str] | None:
        if enabled is None:
            return None
        if not isinstance(enabled, list):
            raise ValueError("enabled tools must be a list")
        normalized: set[str] = set()
        for item in enabled:
            if not isinstance(item, str):
                raise ValueError("enabled tool name must be text")
            name = item.strip()
            if name:
                normalized.add(name)
        return normalized

    def register(self, tool: RegisteredTool) -> None:
        name = self._normalize_name(tool.name)
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=self._normalize_description(tool.description),
            provider=self._normalize_provider(tool.provider),
            handler=tool.handler,
            input_schema=self._normalize_input_schema(tool.input_schema),
        )

    def list_tools(self, enabled: list[str] | None = None) -> list[dict[str, Any]]:
        selected = set(self._tools.keys()) if enabled is None else self._normalize_enabled(enabled)
        assert selected is not None
        unknown = sorted(selected - set(self._tools.keys()))
        if unknown:
            raise PermissionError(f"unknown enabled tool: {', '.join(unknown)}")
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "provider": tool.provider,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
            if tool.name in selected
        ]

    async def call(
        self, name: str, arguments: dict[str, Any], enabled: list[str] | None = None
    ) -> dict[str, Any]:
        normalized_name = self._normalize_name(name)
        selected = self._normalize_enabled(enabled)
        if selected is not None and normalized_name not in selected:
            raise PermissionError(f"tool is not enabled for this session: {normalized_name}")
        if normalized_name not in self._tools:
            raise KeyError(f"tool not registered: {normalized_name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        result = await self._tools[normalized_name].handler(dict(arguments))
        if not isinstance(result, dict):
            raise ValueError("tool result must be an object")
        return dict(result)


@dataclass(slots=True)
class SubagentDefinition:
    name: str
    purpose: str
    tool_names: list[str]
    input_schema: dict[str, Any] = field(default_factory=dict)


class SubagentRegistry:
    def __init__(self) -> None:
        self._subagents: dict[str, SubagentDefinition] = {}

    @staticmethod
    def _normalize_name(name: object) -> str:
        if not isinstance(name, str):
            raise ValueError("subagent name must be text")
        normalized = name.strip()
        if not normalized:
            raise ValueError("subagent name is required")
        return normalized

    @staticmethod
    def _normalize_purpose(purpose: object) -> str:
        if not isinstance(purpose, str):
            raise ValueError("subagent purpose is required")
        normalized = purpose.strip()
        if not normalized:
            raise ValueError("subagent purpose is required")
        return normalized

    @staticmethod
    def _normalize_tool_names(tool_names: object) -> list[str]:
        if not isinstance(tool_names, list):
            raise ValueError("subagent tool_names must be a list")
        normalized: list[str] = []
        for item in tool_names:
            if not isinstance(item, str):
                raise ValueError("subagent tool name must be text")
            name = item.strip()
            if not name:
                raise ValueError("subagent tool name is required")
            normalized.append(name)
        return normalized

    @staticmethod
    def _normalize_input_schema(input_schema: object) -> dict[str, Any]:
        return _normalize_input_schema_object(input_schema, "subagent")

    def register(self, subagent: SubagentDefinition) -> None:
        name = self._normalize_name(subagent.name)
        if name in self._subagents:
            raise ValueError(f"subagent already registered: {name}")
        self._subagents[name] = SubagentDefinition(
            name=name,
            purpose=self._normalize_purpose(subagent.purpose),
            tool_names=self._normalize_tool_names(subagent.tool_names),
            input_schema=self._normalize_input_schema(subagent.input_schema),
        )

    def definitions(self) -> list[SubagentDefinition]:
        return list(self._subagents.values())

    def list_subagents(self) -> list[dict[str, object]]:
        return [
            {"name": item.name, "purpose": item.purpose, "tool_names": item.tool_names}
            for item in self._subagents.values()
        ]
