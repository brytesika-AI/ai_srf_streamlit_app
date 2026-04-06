"""
Minimal stdio MCP bridge for AI-SRF.

Supports:
- loading server config from repo root
- starting a configured MCP server over stdio
- initialize / tools/list / tools/call
- best-effort live web search across configured search-capable servers
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_CANDIDATES = [
    ROOT_DIR / "mcp_servers.json",
    ROOT_DIR / "mcp_servers.example.json",
]


def load_mcp_config() -> dict[str, Any]:
    for path in CONFIG_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"mcpServers": {}}


def get_configured_servers() -> dict[str, Any]:
    return load_mcp_config().get("mcpServers", {})


class MCPError(RuntimeError):
    pass


class StdioMCPClient:
    def __init__(self, server_name: str, server_cfg: dict[str, Any]):
        self.server_name = server_name
        self.server_cfg = server_cfg
        self.proc: subprocess.Popen[bytes] | None = None
        self._next_id = 1

    def __enter__(self) -> "StdioMCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        cmd = [self.server_cfg["command"], *self.server_cfg.get("args", [])]
        env = os.environ.copy()
        env.update(self.server_cfg.get("env", {}))
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        init = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-srf", "version": "0.1.0"},
            },
        )
        if init.get("error"):
            raise MCPError(f"{self.server_name} initialize failed: {init['error']}")
        self.notify("notifications/initialized", {})

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.time() + 20
        while time.time() < deadline:
            msg = self._recv()
            if msg.get("id") == req_id:
                return msg
        raise MCPError(f"{self.server_name} timed out waiting for {method}")

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.request("tools/list", {})
        if response.get("error"):
            raise MCPError(f"{self.server_name} tools/list failed: {response['error']}")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self.request("tools/call", {"name": tool_name, "arguments": arguments})
        if response.get("error"):
            raise MCPError(f"{self.server_name} tools/call failed: {response['error']}")
        return response.get("result", {})

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise MCPError(f"{self.server_name} is not running")
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise MCPError(f"{self.server_name} is not running")
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.proc.stdout.read(1)
            if not chunk:
                err = b""
                if self.proc.stderr:
                    err = self.proc.stderr.read1(4096)
                raise MCPError(f"{self.server_name} closed stdout. stderr={err.decode(errors='ignore')}")
            header += chunk
        raw_header, _ = header.split(b"\r\n\r\n", 1)
        content_length = None
        for line in raw_header.decode("ascii", errors="ignore").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
                break
        if content_length is None:
            raise MCPError(f"{self.server_name} sent response without Content-Length")
        body = self.proc.stdout.read(content_length)
        return json.loads(body.decode("utf-8"))


def search_live_web(query: str, limit: int = 5) -> dict[str, Any]:
    servers = get_configured_servers()
    if not servers:
        return {
            "status": "unavailable",
            "reason": "No MCP servers configured.",
            "sources": [],
        }

    attempts: list[str] = []
    candidates = ["exa", "firecrawl", "fetch"]
    for server_name in candidates:
        cfg = servers.get(server_name)
        if not cfg:
            continue
        try:
            with StdioMCPClient(server_name, cfg) as client:
                tools = client.list_tools()
                tool_names = [t.get("name", "") for t in tools]
                attempts.append(f"{server_name}:{','.join(tool_names[:6])}")
                for tool_name in tool_names:
                    lowered = tool_name.lower()
                    if "search" in lowered:
                        result = client.call_tool(tool_name, {"query": query, "limit": limit})
                        return {
                            "status": "ok",
                            "server": server_name,
                            "tool": tool_name,
                            "query": query,
                            "result": result,
                        }
                if server_name == "fetch":
                    for tool_name in tool_names:
                        if "fetch" in tool_name.lower():
                            result = client.call_tool(tool_name, {"url": query})
                            return {
                                "status": "ok",
                                "server": server_name,
                                "tool": tool_name,
                                "query": query,
                                "result": result,
                            }
        except Exception as exc:
            attempts.append(f"{server_name}:error={exc}")

    return {
        "status": "unavailable",
        "reason": "No configured MCP search-capable tool succeeded.",
        "attempts": attempts,
        "sources": [],
    }


def probe_mcp_servers() -> dict[str, Any]:
    servers = get_configured_servers()
    checks: list[dict[str, Any]] = []

    for server_name, cfg in servers.items():
        record: dict[str, Any] = {
            "server": server_name,
            "command": cfg.get("command"),
            "args": cfg.get("args", []),
            "status": "unknown",
            "tool_count": 0,
        }
        try:
            with StdioMCPClient(server_name, cfg) as client:
                tools = client.list_tools()
                record["status"] = "ok"
                record["tool_count"] = len(tools)
                record["tools"] = [tool.get("name", "") for tool in tools[:8]]
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        checks.append(record)

    ok_count = sum(1 for c in checks if c["status"] == "ok")
    return {
        "configured_count": len(checks),
        "healthy_count": ok_count,
        "servers": checks,
    }
