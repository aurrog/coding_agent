from collections.abc import Iterable
import unittest

from agent.runner import AgentRunner
from config import AgentContext, AgentSettings
from core.models import (
    AgentStatus,
    ModelResponse,
    ToolCall,
)
from security.policy import ToolPolicy
from tools.base import BaseTool, ToolRisk
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: Iterable[ModelResponse]):
        self._responses = iter(responses)
        self.received_messages: list[list[dict]] = []

    def complete(self, messages, tools):
        self.received_messages.append(list(messages))
        return next(self._responses)


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns the supplied value."
    risk = ToolRisk.READ_ONLY
    arguments_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return {"value": arguments["value"]}


def make_runner(llm, *, max_iterations=3, max_tool_calls=3):
    registry = ToolRegistry(ToolPolicy("read_only"))
    registry.register(EchoTool())
    return AgentRunner(
        llm=llm,
        tools=registry,
        settings=AgentSettings(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_context_characters=100_000,
        ),
        context=AgentContext(
            workspace_root="workspace",
            permission_mode="read_only",
        ),
    )


class AgentRunnerTests(unittest.TestCase):
    def test_returns_immediate_final_response(self):
        llm = FakeLLM(
            [
                ModelResponse(
                    content="Analysis complete",
                    tool_calls=[],
                    finish_reason="stop",
                )
            ]
        )

        result = make_runner(llm).run("Analyze the project")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.final_text, "Analysis complete")
        self.assertEqual(result.iterations, 1)

    def test_executes_tool_and_continues(self):
        llm = FakeLLM(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="echo",
                            arguments={"value": "hello"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    content="Found hello",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )

        result = make_runner(llm).run("Find a value")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.tool_calls_count, 1)
        second_request = llm.received_messages[1]
        self.assertEqual(second_request[-1]["role"], "tool")
        self.assertIn('"value": "hello"', second_request[-1]["content"])

    def test_stops_at_iteration_limit(self):
        responses = [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call-{index}",
                        name="echo",
                        arguments={"value": "again"},
                    )
                ],
                finish_reason="tool_calls",
            )
            for index in range(2)
        ]

        result = make_runner(
            FakeLLM(responses),
            max_iterations=2,
        ).run("Keep inspecting")

        self.assertEqual(result.status, AgentStatus.INCOMPLETE)
        self.assertEqual(result.error, "ITERATION_LIMIT_REACHED")


if __name__ == "__main__":
    unittest.main()
