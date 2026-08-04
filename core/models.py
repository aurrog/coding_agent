from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    ok: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None

    @classmethod
    def success(
        cls,
        tool_call_id: str,
        data: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            ok=True,
            data=data or {},
        )

    @classmethod
    def failure(
        cls,
        tool_call_id: str,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> "ToolResult":
        return cls(
            tool_call_id=tool_call_id,
            ok=False,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
            ),
        )

    def to_message(self) -> dict[str, Any]:
        payload = {
            "ok": self.ok,
            "data": self.data if self.ok else None,
            "error": (
                asdict(self.error)
                if self.error is not None
                else None
            ),
        }
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None

    def to_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments or {},
                            ensure_ascii=False,
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class AgentStatus(Enum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentResult:
    status: AgentStatus
    final_text: str | None
    iterations: int
    tool_calls_count: int
    error: str | None = None

    @classmethod
    def completed(
        cls,
        final_text: str,
        iterations: int,
        tool_calls_count: int,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.COMPLETED,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
        )

    @classmethod
    def failed(
        cls,
        *,
        iterations: int,
        tool_calls_count: int,
        error: str,
        final_text: str | None = None,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.FAILED,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
            error=error,
        )

    @classmethod
    def incomplete(
        cls,
        *,
        iterations: int,
        tool_calls_count: int,
        reason: str,
        final_text: str | None = None,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.INCOMPLETE,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
            error=reason,
        )
