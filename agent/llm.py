import json
from typing import Any, Protocol

from openai import OpenAI

from config import LLMSettings
from core.models import ModelResponse, ToolCall


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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
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
        )
