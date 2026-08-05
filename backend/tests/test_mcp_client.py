import sys
from pathlib import Path

import pytest

from app.mcp.client import MCPProtocolError, MCPStdioClient, MCPTool, MCPToolProvider


@pytest.mark.asyncio
async def test_stdio_mcp_client_lists_and_calls_tools() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        tools = await client.list_tools()
        result = await client.call_tool("echo", {"dish": "番茄炒蛋"})
    finally:
        await client.close()

    assert tools[0]["name"] == "echo"
    assert result["content"][0]["text"] == '{"dish": "番茄炒蛋"}'


@pytest.mark.asyncio
async def test_stdio_mcp_client_handles_notifications_before_batched_response() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_batch_server.py"
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        tools = await client.list_tools()
        result = await client.call_tool("batch_echo", {"query": "百科搜索"})
    finally:
        await client.close()

    assert tools[0]["name"] == "batch_echo"
    assert result["content"][0]["text"] == '{"query": "百科搜索"}'


@pytest.mark.asyncio
async def test_stdio_mcp_client_wraps_invalid_json_stdout(tmp_path) -> None:
    server = tmp_path / "invalid_json_mcp_server.py"
    server.write_text(
        'import sys\n'
        'sys.stdin.readline()\n'
        'print("not json", flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        with pytest.raises(MCPProtocolError, match="invalid JSON"):
            await client.initialize()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_non_object_error_payload(tmp_path) -> None:
    server = tmp_path / "non_object_error_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"error":"boom"}), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        with pytest.raises(MCPProtocolError, match="error must be an object"):
            await client.initialize()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_malformed_tool_list(tmp_path) -> None:
    server = tmp_path / "malformed_tools_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"result":{}}), flush=True)\n'
        'sys.stdin.readline()\n'
        'tools_list = json.loads(sys.stdin.readline())\n'
        'response = {"jsonrpc":"2.0","id":tools_list["id"],"result":'
        '{"tools":[{"description":"missing name"}]}}\n'
        'print(json.dumps(response), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        with pytest.raises(MCPProtocolError, match="tools/list response invalid"):
            await client.list_tools()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_non_string_tool_description(tmp_path) -> None:
    server = tmp_path / "non_string_description_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"result":{}}), flush=True)\n'
        'sys.stdin.readline()\n'
        'tools_list = json.loads(sys.stdin.readline())\n'
        'response = {"jsonrpc":"2.0","id":tools_list["id"],"result":'
        '{"tools":[{"name":"bad_description","description":123,"inputSchema":{"type":"object"}}]}}\n'
        'print(json.dumps(response), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        with pytest.raises(MCPProtocolError, match="tool description must be text"):
            await client.list_tools()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_non_object_result_payload(tmp_path) -> None:
    server = tmp_path / "non_object_result_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"result":[["tools", []]]}), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        with pytest.raises(MCPProtocolError, match="result must be an object"):
            await client.initialize()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_malformed_tool_call_result(tmp_path) -> None:
    server = tmp_path / "malformed_tool_call_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"result":{}}), flush=True)\n'
        'sys.stdin.readline()\n'
        'tool_call = json.loads(sys.stdin.readline())\n'
        'response = {"jsonrpc":"2.0","id":tool_call["id"],"result":{"isError":False}}\n'
        'print(json.dumps(response), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        with pytest.raises(MCPProtocolError, match="tools/call response invalid"):
            await client.call_tool("broken", {"query": "百科搜索"})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_surfaces_tool_call_is_error_result(tmp_path) -> None:
    server = tmp_path / "tool_error_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'initialize = json.loads(sys.stdin.readline())\n'
        'print(json.dumps({"jsonrpc":"2.0","id":initialize["id"],"result":{}}), flush=True)\n'
        'sys.stdin.readline()\n'
        'tool_call = json.loads(sys.stdin.readline())\n'
        'response = {"jsonrpc":"2.0","id":tool_call["id"],"result":'
        '{"content":[{"type":"text","text":"remote tool failed"}],"isError":True}}\n'
        'print(json.dumps(response), flush=True)\n',
        encoding="utf-8",
    )
    client = MCPStdioClient(command=sys.executable, args=["-u", str(server)])

    try:
        await client.initialize()
        with pytest.raises(MCPProtocolError, match="remote tool failed"):
            await client.call_tool("broken", {"query": "百科搜索"})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stdio_mcp_client_rejects_non_object_arguments_before_request() -> None:
    client = MCPStdioClient(command=sys.executable)

    async def fail_request(method, params):
        raise AssertionError("non-object MCP arguments must not reach tools/call request")

    client._request = fail_request  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="MCP tool arguments must be an object"):
        await client.call_tool("echo", ["query", "百科搜索"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mcp_tool_provider_rejects_non_text_call_names_before_lookup() -> None:
    async def handler(arguments: dict) -> dict:
        return {"content": []}

    provider = MCPToolProvider()
    provider.register(MCPTool("echo", "Echo arguments", handler))

    with pytest.raises(ValueError, match="MCP tool name must be text"):
        await provider.call(123, {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mcp_tool_provider_rejects_non_object_arguments_before_handler() -> None:
    async def handler(arguments: dict) -> dict:
        raise AssertionError("non-object MCP arguments must not reach handler")

    provider = MCPToolProvider()
    provider.register(MCPTool("echo", "Echo arguments", handler))

    with pytest.raises(ValueError, match="MCP tool arguments must be an object"):
        await provider.call("echo", ["query", "百科搜索"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mcp_tool_provider_rejects_non_object_handler_results() -> None:
    async def handler(arguments: dict) -> list[str]:
        return ["not", "an", "object"]

    provider = MCPToolProvider()
    provider.register(MCPTool("echo", "Echo arguments", handler))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="MCP tool result must be an object"):
        await provider.call("echo", {"query": "百科搜索"})


def test_mcp_tool_provider_rejects_invalid_tool_catalog_metadata() -> None:
    async def handler(arguments: dict) -> dict:
        return {"content": []}

    provider = MCPToolProvider()

    with pytest.raises(ValueError, match="MCP tool name is required"):
        provider.register(MCPTool("   ", "外部搜索", handler))

    with pytest.raises(ValueError, match="MCP tool name must be text"):
        provider.register(MCPTool(123, "外部搜索", handler))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="MCP tool description is required"):
        provider.register(MCPTool("web_search", "   ", handler))

    with pytest.raises(ValueError, match="MCP tool input_schema must be an object"):
        provider.register(
            MCPTool("bad_schema", "坏 schema", handler, ["query"])  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="MCP tool input_schema required entries must be text"):
        provider.register(
            MCPTool(
                "bad_required_schema",
                "坏 required",
                handler,
                {"type": "object", "required": ["query", 123]},
            )  # type: ignore[list-item]
        )

    with pytest.raises(ValueError, match="MCP tool input_schema properties must be an object"):
        provider.register(
            MCPTool(
                "bad_properties_schema",
                "坏 properties",
                handler,
                {"type": "object", "properties": ["query"]},
            )  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="MCP tool input_schema property names must be text"):
        provider.register(
            MCPTool(
                "bad_property_name_schema",
                "坏 property name",
                handler,
                {"type": "object", "properties": {123: {"type": "string"}}},
            )  # type: ignore[dict-item]
        )

    with pytest.raises(ValueError, match="MCP tool input_schema property names must not be blank"):
        provider.register(
            MCPTool(
                "blank_property_name_schema",
                "空 property name",
                handler,
                {"type": "object", "properties": {"   ": {"type": "string"}}},
            )
        )

    assert provider.list_tools() == []


@pytest.mark.asyncio
async def test_mcp_tool_provider_registers_stdio_tools() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    provider = MCPToolProvider()

    await provider.connect_stdio_server("local", sys.executable, ["-u", str(server)])
    try:
        tools = provider.list_tools()
        result = await provider.call("echo", {"query": "百科搜索"})
    finally:
        await provider.close()

    assert tools == [
        {
            "name": "echo",
            "description": "Echo arguments",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert result["content"][0]["text"] == '{"query": "百科搜索"}'


@pytest.mark.asyncio
async def test_mcp_tool_provider_normalizes_remote_tool_names(tmp_path) -> None:
    server = tmp_path / "spaced_tool_mcp_server.py"
    server.write_text(
        'import json, sys\n'
        'for line in sys.stdin:\n'
        '    request = json.loads(line)\n'
        '    method = request.get("method")\n'
        '    request_id = request.get("id")\n'
        '    if method == "notifications/initialized":\n'
        '        continue\n'
        '    if method == "initialize":\n'
        '        result = {"protocolVersion":"2025-06-18","capabilities":{"tools":{}}}\n'
        '    elif method == "tools/list":\n'
        '        result = {"tools":[{"name":" echo ","description":" Spaced echo ","inputSchema":{"type":"object"}}]}\n'
        '    elif method == "tools/call":\n'
        '        result = {"content":[{"type":"text","text":request["params"]["name"]}],"isError":False}\n'
        '    else:\n'
        '        print(json.dumps({"jsonrpc":"2.0","id":request_id,"error":{"message":"not found"}}), flush=True)\n'
        '        continue\n'
        '    print(json.dumps({"jsonrpc":"2.0","id":request_id,"result":result}), flush=True)\n',
        encoding="utf-8",
    )
    provider = MCPToolProvider()

    await provider.connect_stdio_server("local", sys.executable, ["-u", str(server)])
    try:
        tools = provider.list_tools()
        result = await provider.call("echo", {})
    finally:
        await provider.close()

    assert tools[0]["name"] == "echo"
    assert tools[0]["description"] == "Spaced echo"
    assert result["content"][0]["text"] == "echo"


@pytest.mark.asyncio
async def test_mcp_tool_provider_rejects_duplicate_remote_tool_names() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    provider = MCPToolProvider()

    await provider.connect_stdio_server("first", sys.executable, ["-u", str(server)])
    try:
        with pytest.raises(ValueError, match="MCP tool already registered: echo"):
            await provider.connect_stdio_server("second", sys.executable, ["-u", str(server)])
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_configured_mcp_tools_reject_invalid_servers_shape(tmp_path) -> None:
    from app.core.config import Settings
    from app.main import build_application_state, load_configured_mcp_tools

    config = tmp_path / "mcp_servers.json"
    config.write_text('{"servers":[]}', encoding="utf-8")
    settings = Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key="test-key",
        cache_enabled=False,
        mcp_servers_config=str(config),
    )
    state = build_application_state(settings)

    with pytest.raises(RuntimeError, match="MCP_SERVERS_CONFIG invalid: servers must be an object"):
        await load_configured_mcp_tools(state)


@pytest.mark.asyncio
async def test_configured_mcp_tools_rejects_non_string_args_before_process_start(tmp_path) -> None:
    from app.core.config import Settings
    from app.main import build_application_state, load_configured_mcp_tools

    config = tmp_path / "mcp_servers.json"
    config.write_text(
        '{"servers":{"encyclopedia":{"command":"python","args":[null]}}}',
        encoding="utf-8",
    )
    settings = Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key="test-key",
        cache_enabled=False,
        mcp_servers_config=str(config),
    )
    state = build_application_state(settings)

    async def fail_connect(**_kwargs):
        raise AssertionError("invalid MCP args must not start a stdio server")

    state.mcp_provider.connect_stdio_server = fail_connect

    with pytest.raises(RuntimeError, match=r"server encyclopedia args\[0\] must be a string"):
        await load_configured_mcp_tools(state)


@pytest.mark.asyncio
async def test_configured_mcp_tools_reject_placeholder_values_before_process_start(tmp_path) -> None:
    from app.core.config import Settings
    from app.main import build_application_state, load_configured_mcp_tools

    config = tmp_path / "mcp_servers.json"
    config.write_text(
        '{"servers":{"encyclopedia":{"command":"replace-with-mcp-server-command",'
        '"args":["--stdio"],"env":{"API_KEY":"replace-with-provider-key"}}}}',
        encoding="utf-8",
    )
    settings = Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key="test-key",
        cache_enabled=False,
        mcp_servers_config=str(config),
    )
    state = build_application_state(settings)

    with pytest.raises(RuntimeError, match="server encyclopedia command must not be a placeholder"):
        await load_configured_mcp_tools(state)

    assert state.mcp_provider.list_tools() == []


@pytest.mark.asyncio
async def test_configured_mcp_tools_are_registered_for_agent_use(tmp_path) -> None:
    from app.core.config import Settings
    from app.main import build_application_state, load_configured_mcp_tools

    server = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    config = tmp_path / "mcp_servers.json"
    config.write_text(
        '{"servers":{"local":{"command":"' + sys.executable + '","args":["-u","' + str(server) + '"]}}}',
        encoding="utf-8",
    )
    settings = Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key="test-key",
        cache_enabled=False,
        mcp_servers_config=str(config),
    )
    state = build_application_state(settings)

    await load_configured_mcp_tools(state)
    try:
        result = await state.tool_registry.call("echo", {"query": "百科搜索"})
    finally:
        await state.mcp_provider.close()

    echo_tool = next(tool for tool in state.tool_registry.list_tools() if tool["name"] == "echo")

    assert echo_tool["provider"] == "mcp"
    assert echo_tool["input_schema"] == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    assert result["content"][0]["text"] == '{"query": "百科搜索"}'
