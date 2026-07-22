# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# API_KEY=os.getenv('API_KEY')
# BASE_URL=os.getenv('BASE_URL')

# client = OpenAI(
#     base_url=BASE_URL,
#     api_key=API_KEY,
# )

# def chat(messages):

#     resp = client.chat.completions.create(
#         model="deepseek/deepseek-v4-flash",
#         messages=messages
#     )
#     return resp.choices[0].message.content


# def chat_with_tools(messages, tools):
#     resp=client.chat.completions.create(
#         model="deepseek/deepseek-v4-flash", 
#         messages=messages,
#         tools=tools
#     )
#     return resp

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

        tool_cals=[]
        for toolcall in message.tool_calls:

            try:
                arguments=json.loads(toolcall.function.arguments)

                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be a JSON object')
                

                tool_cals.append(
                    ToolCall(
                        id=toolcall.id,
                        name=str(toolcall.function.name)
                        arguments=arguments
                    )
                )
            except json.JSONDecodeError as e:
                

                    

                # tool_cals.append(
                #     ToolCall(
                #         id=toolcall.id,
                #         name=str(toolcall.function.name),
                #         arguments=json.loads(toolcall.function.arguments)
                #     )
                # )


