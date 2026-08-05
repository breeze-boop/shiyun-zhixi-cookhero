from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class MCPTool:
    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPProtocolError(RuntimeError):
    pass


class MCPStdioClient:
    def __init__(
        self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1

    async def initialize(self) -> dict[str, Any]:
        await self._ensure_process()
        result = await self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "cookhero", "version": "0.1.0"},
            },
        )
        await self._notify("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise MCPProtocolError("MCP tools/list response invalid: tools must be a list")
        normalized_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise MCPProtocolError("MCP tools/list response invalid: tool must be an object")
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                raise MCPProtocolError("MCP tools/list response invalid: tool name is required")
            input_schema = tool.get("inputSchema")
            if input_schema is not None and not isinstance(input_schema, dict):
                raise MCPProtocolError("MCP tools/list response invalid: inputSchema must be an object")
            description = tool.get("description")
            if description is not None and not isinstance(description, str):
                raise MCPProtocolError("MCP tools/list response invalid: tool description must be text")
            normalized_tools.append(
                {
                    **tool,
                    "name": name.strip(),
                    "description": description.strip() if isinstance(description, str) else description,
                }
            )
        return normalized_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object")
        result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        self._validate_tool_call_result(result)
        return result

    @staticmethod
    def _validate_tool_call_result(result: dict[str, Any]) -> None:
        content = result.get("content")
        if not isinstance(content, list):
            raise MCPProtocolError("MCP tools/call response invalid: content must be a list")
        for item in content:
            if not isinstance(item, dict):
                raise MCPProtocolError("MCP tools/call response invalid: content item must be an object")
            content_type = item.get("type")
            if not isinstance(content_type, str) or not content_type.strip():
                raise MCPProtocolError("MCP tools/call response invalid: content item type is required")
            if content_type == "text" and not isinstance(item.get("text"), str):
                raise MCPProtocolError("MCP tools/call response invalid: text content requires text")
        structured_content = result.get("structuredContent")
        if structured_content is not None and not isinstance(structured_content, dict):
            raise MCPProtocolError("MCP tools/call response invalid: structuredContent must be an object")
        is_error = result.get("isError")
        if is_error is not None and not isinstance(is_error, bool):
            raise MCPProtocolError("MCP tools/call response invalid: isError must be a boolean")
        if is_error is True:
            message = next(
                (
                    item.get("text", "").strip()
                    for item in content
                    if item.get("type") == "text" and item.get("text", "").strip()
                ),
                "MCP tool returned an error result",
            )
            raise MCPProtocolError(message)

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self._process = None

    async def _ensure_process(self) -> None:
        if self._process is not None:
            return
        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
        )

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        await self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        response = await self._read_response(request_id)
        if "error" in response:
            error = response["error"]
            if not isinstance(error, dict):
                raise MCPProtocolError("MCP response invalid: error must be an object")
            message = error.get("message")
            error_message = (
                message.strip() if isinstance(message, str) and message.strip() else str(error)
            )
            raise MCPProtocolError(error_message)
        result = response.get("result") or {}
        if not isinstance(result, dict):
            raise MCPProtocolError("MCP response invalid: result must be an object")
        return dict(result)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._write({"jsonrpc": "2.0", "method": method, "params": params})

    async def _write(self, message: dict[str, Any]) -> None:
        await self._ensure_process()
        assert self._process is not None and self._process.stdin is not None
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise MCPProtocolError("MCP server closed stdout before responding")
            try:
                payload = json.loads(line.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise MCPProtocolError("MCP server response is not valid UTF-8") from exc
            except json.JSONDecodeError as exc:
                raise MCPProtocolError("MCP server response contains invalid JSON") from exc
            messages = payload if isinstance(payload, list) else [payload]
            for response in messages:
                if isinstance(response, dict) and response.get("id") == request_id:
                    return response


class MCPToolProvider:
    """Registry facade for MCP-compatible external tools."""

    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}
        self._clients: list[MCPStdioClient] = []

    @staticmethod
    def _normalize_name(name: object) -> str:
        if not isinstance(name, str):
            raise ValueError("MCP tool name must be text")
        normalized = name.strip()
        if not normalized:
            raise ValueError("MCP tool name is required")
        return normalized

    @staticmethod
    def _normalize_description(description: object) -> str:
        if not isinstance(description, str):
            raise ValueError("MCP tool description is required")
        normalized = description.strip()
        if not normalized:
            raise ValueError("MCP tool description is required")
        return normalized

    @staticmethod
    def _normalize_input_schema(input_schema: object) -> dict[str, Any]:
        if not isinstance(input_schema, dict):
            raise ValueError("MCP tool input_schema must be an object")
        normalized = dict(input_schema)
        required = normalized.get("required")
        if required is not None:
            if not isinstance(required, list):
                raise ValueError("MCP tool input_schema required must be a list")
            for item in required:
                if not isinstance(item, str):
                    raise ValueError("MCP tool input_schema required entries must be text")
                if not item.strip():
                    raise ValueError("MCP tool input_schema required entries must not be blank")
        properties = normalized.get("properties")
        if properties is not None:
            if not isinstance(properties, dict):
                raise ValueError("MCP tool input_schema properties must be an object")
            for name in properties:
                if not isinstance(name, str):
                    raise ValueError("MCP tool input_schema property names must be text")
                if not name.strip():
                    raise ValueError("MCP tool input_schema property names must not be blank")
        return normalized

    def register(self, tool: MCPTool) -> None:
        name = self._normalize_name(tool.name)
        if name in self._tools:
            raise ValueError(f"MCP tool already registered: {name}")
        self._tools[name] = MCPTool(
            name=name,
            description=self._normalize_description(tool.description),
            handler=tool.handler,
            input_schema=self._normalize_input_schema(tool.input_schema),
        )

    async def connect_stdio_server(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        client = MCPStdioClient(command, args, env)
        await client.initialize()
        self._clients.append(client)
        for tool in await client.list_tools():
            name = tool["name"]
            description = tool.get("description") or f"MCP tool from {server_name}"
            input_schema = tool.get("inputSchema") or {}

            async def handler(arguments: dict[str, Any], *, tool_name: str = name, mcp_client: MCPStdioClient = client) -> dict[str, Any]:
                return await mcp_client.call_tool(tool_name, arguments)

            self.register(
                MCPTool(
                    name=name,
                    description=description,
                    handler=handler,
                    input_schema=input_schema,
                )
            )

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized_name = self._normalize_name(name)
        if normalized_name not in self._tools:
            raise KeyError(f"MCP tool not registered: {normalized_name}")
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object")
        result = await self._tools[normalized_name].handler(dict(arguments))
        if not isinstance(result, dict):
            raise ValueError("MCP tool result must be an object")
        return dict(result)

    async def close(self) -> None:
        for client in self._clients:
            await client.close()
        self._clients = []
