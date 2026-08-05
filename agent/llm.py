import json
from typing import Any, Protocol

from openai import OpenAI

from config import LLMSettings
from core.models import ModelResponse, TokenUsage, ToolCall


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        ...


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings):
        self._model = settings.model
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools
        response = self._client.chat.completions.create(**request)
        return self._normalize_response(response)

    @staticmethod
    def _normalize_response(response: Any) -> ModelResponse:
        if not response.choices:
            raise ValueError("Model response has no choices")

        choice = response.choices[0]
        message = choice.message
        raw_tool_calls = message.tool_calls or []
        tool_calls: list[ToolCall] = []

        for raw_call in raw_tool_calls:
            raw_arguments = raw_call.function.arguments
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object"
                    )
            except (json.JSONDecodeError, ValueError) as exc:
                tool_calls.append(
                    ToolCall(
                        id=str(raw_call.id),
                        name=str(raw_call.function.name),
                        arguments=None,
                        error=(
                            f"Invalid tool arguments: {exc}. "
                            f"Received: {raw_arguments!r}"
                        ),
                    )
                )
                continue

            tool_calls.append(
                ToolCall(
                    id=str(raw_call.id),
                    name=str(raw_call.function.name),
                    arguments=arguments,
                )
            )

        return ModelResponse(
            content=(
                str(message.content)
                if message.content is not None
                else None
            ),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=OpenAICompatibleLLM._normalize_usage(response),
        )

    @staticmethod
    def _normalize_usage(response: Any) -> TokenUsage:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return TokenUsage()

        prompt_tokens = int(
            getattr(raw_usage, "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(raw_usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(raw_usage, "total_tokens", 0)
            or prompt_tokens + completion_tokens
        )
        prompt_details = getattr(
            raw_usage,
            "prompt_tokens_details",
            None,
        )
        completion_details = getattr(
            raw_usage,
            "completion_tokens_details",
            None,
        )

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=int(
                getattr(prompt_details, "cached_tokens", 0) or 0
            ),
            reasoning_tokens=int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            ),
            reported=True,
        )
