"""Tool registry: JSON-schema descriptions plus handlers for the LLM.

The registry is the single place that defines what the assistant can do, what
the model is told about it, and what actually executes. Keeping them together
is what makes the UI's tool trace trustworthy.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

Handler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class ToolResult:
    """Outcome of one tool invocation, shaped for the UI trace."""

    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_ms: int | None = None
    started_at: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": _jsonable(self.result),
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


def _jsonable(value: Any) -> Any:
    """Coerce tool output into something ``json.dumps`` will accept."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def to_openai_schema(self) -> dict[str, Any]:
        """Shape expected by llama-cpp-python's ``tools`` argument."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool, converting any failure into a reportable message."""
        started = time.monotonic()
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                arguments=arguments,
                error=f"no such tool: {name}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        try:
            value = await tool.handler(arguments or {})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            return ToolResult(
                name=name,
                arguments=arguments,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        return ToolResult(
            name=name,
            arguments=arguments,
            result=value,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
