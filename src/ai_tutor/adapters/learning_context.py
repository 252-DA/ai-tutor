from typing import Any

from ai_runtime.mcp import MCPToolClient


class MCPCourseTools:
    """Gọi công cụ của mcp-service. Một lỗi công cụ ném ra MCPToolError cho vòng lặp xử lý."""

    def __init__(self, client: MCPToolClient) -> None:
        self._client = client

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._client.call_tool(name, arguments)
