from typing import Protocol
from openai import OpenAI
import json

from core.models import ModelResponse, ToolCall
from config import LLMSettings



class LLMClient(Protocol):
    def complete(self, messages: list, tools: list[dict]) -> ModelResponse:
        ...


class OpenAICompatibleLLM:
    def __init__(self, settings: LLMSettings):
        self._model = settings.model
        self._client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries
        )

    def complete(self, messages, tools) -> ModelResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )

        return self._normalize_response(response)
    
    def _normalize_response(self, response):
        choice = response.choices[0]
        message=choice.message


        if choice.finish_reason != "tool_calls":
            return ModelResponse(
                content=str(message.content or ""),
                tool_calls=[],
                finish_reason=choice.finish_reason or "stop",
            )

        tool_calls=[]
        for toolcall in message.tool_calls:
            raw_arguments = toolcall.function.arguments
            try:
                arguments=json.loads(raw_arguments)

                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be a JSON object')
                

                tool_calls.append(
                    ToolCall(
                        id=toolcall.id,
                        name=str(toolcall.function.name),
                        arguments=arguments
                    )
                )
            except (json.JSONDecodeError, ValueError) as e:
                tool_calls.append(
                    ToolCall(
                        id=toolcall.id,
                        name=str(toolcall.function.name),
                        arguments=None,
                        error=(
                            f'Invalid tool arguments: {e}',
                            f'Recieved: {raw_arguments!r}'
                        )
                    )
                )
        return ModelResponse(
            content=None,
            tool_calls=tool_calls,
            finish_reason='tool_calls'
        )

