from __future__ import annotations

import json
import sys


def response_for(request: dict) -> dict | None:
    request_id = request.get("id")
    method = request.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "batch", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "batch_echo",
                        "description": "Echo arguments from a batched response",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ]
            },
        }
    if method == "tools/call":
        arguments = request.get("params", {}).get("arguments", {})
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(arguments, ensure_ascii=False)}
                ],
                "isError": False,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": "method not found"},
    }


for line in sys.stdin:
    payload = line.strip()
    if not payload:
        continue
    request = json.loads(payload)
    response = response_for(request)
    if response is None:
        continue
    sys.stdout.write(
        json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}},
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stdout.write(json.dumps([response], ensure_ascii=False) + "\n")
    sys.stdout.flush()
