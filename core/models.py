from dataclasses import asdict, dataclass, field
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
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    reported: bool = False

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=(
                self.completion_tokens + other.completion_tokens
            ),
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=(
                self.reasoning_tokens + other.reasoning_tokens
            ),
            reported=self.reported or other.reported,
        )


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TokenUsage = field(default_factory=TokenUsage)

    def to_assistant_message(
        self,
        *,
        compact_tool_arguments: bool = False,
    ) -> dict[str, Any]:
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
                            (
                                _compact_tool_arguments(call.arguments or {})
                                if compact_tool_arguments
                                else call.arguments or {}
                            ),
                            ensure_ascii=False,
                        ),
                    },
                }
                for call in self.tool_calls
            ]
        return message


def _compact_tool_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    compacted = dict(arguments)
    for name in {"content", "old_text", "new_text"}:
        value = compacted.get(name)
        if isinstance(value, str):
            compacted[name] = f"<compacted {len(value)} characters>"

    edits = compacted.get("edits")
    if isinstance(edits, list):
        compacted["edits"] = [
            {
                "old_text": (
                    f"<compacted {len(edits)} edit operations>"
                ),
                "new_text": "<compacted after tool execution>",
                "expected_replacements": 1,
            }
        ]
    return compacted


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
    usage: TokenUsage = field(default_factory=TokenUsage)

    @classmethod
    def completed(
        cls,
        final_text: str,
        iterations: int,
        tool_calls_count: int,
        usage: TokenUsage | None = None,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.COMPLETED,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
            usage=usage or TokenUsage(),
        )

    @classmethod
    def failed(
        cls,
        *,
        iterations: int,
        tool_calls_count: int,
        error: str,
        final_text: str | None = None,
        usage: TokenUsage | None = None,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.FAILED,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
            error=error,
            usage=usage or TokenUsage(),
        )

    @classmethod
    def incomplete(
        cls,
        *,
        iterations: int,
        tool_calls_count: int,
        reason: str,
        final_text: str | None = None,
        usage: TokenUsage | None = None,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.INCOMPLETE,
            final_text=final_text,
            iterations=iterations,
            tool_calls_count=tool_calls_count,
            error=reason,
            usage=usage or TokenUsage(),
        )
