from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str]
    error: str | None = None


@dataclass
class TollError:
    code: str
    message: str
    retryable: bool = False


@dataclass
class ToolResult:
    tool_call_id: str
    ok: bool
    error: TollError | None


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None