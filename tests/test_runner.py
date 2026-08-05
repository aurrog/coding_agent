from collections.abc import Iterable
from io import StringIO
import json
import unittest

from agent.observability import ConsoleProgressReporter
from agent.runner import AgentRunner
from config import AgentContext, AgentSettings
from core.models import (
    AgentStatus,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from security.policy import ToolPolicy
from tools.base import BaseTool, ToolRisk
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self, responses: Iterable[ModelResponse]):
        self._responses = iter(responses)
        self.received_messages: list[list[dict]] = []
        self.received_tools: list[list[dict]] = []

    def complete(self, messages, tools):
        self.received_messages.append(list(messages))
        self.received_tools.append(list(tools))
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


def make_runner(
    llm,
    *,
    max_iterations=3,
    max_tool_calls=3,
    progress=None,
):
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
        progress=progress,
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

    def test_uses_tool_budget_then_finalizes_without_tools(self):
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
        responses.append(
            ModelResponse(
                content="Work completed before finalization.",
                tool_calls=[],
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=50,
                    completion_tokens=10,
                    total_tokens=60,
                    reported=True,
                ),
            )
        )
        llm = FakeLLM(responses)

        result = make_runner(
            llm,
            max_iterations=2,
        ).run("Keep inspecting")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.tool_calls_count, 2)
        self.assertEqual(result.usage.total_tokens, 60)
        self.assertEqual(llm.received_tools[-1], [])
        self.assertIn(
            "tool-use iteration budget is exhausted",
            llm.received_messages[-1][-1]["content"],
        )

    def test_compacts_large_tool_arguments_in_follow_up_history(self):
        large_content = "x" * 10_000
        llm = FakeLLM(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-large",
                            name="echo",
                            arguments={
                                "value": "hello",
                                "content": large_content,
                            },
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    content="Done",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )

        make_runner(llm).run("Compact history")

        serialized_history = str(llm.received_messages[1])
        self.assertNotIn(large_content, serialized_history)
        self.assertIn("compacted 10000 characters", serialized_history)

    def test_compacts_stale_read_content_after_file_change(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "read-1",
                "content": json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "path": "src/app.py",
                            "content": "large source code",
                            "sha256": "a" * 64,
                        },
                        "error": None,
                    }
                ),
            }
        ]

        count = AgentRunner._compact_stale_reads(
            messages,
            "src/app.py",
        )
        payload = json.loads(messages[0]["content"])

        self.assertEqual(count, 1)
        self.assertIn("compacted stale", payload["data"]["content"])
        self.assertEqual(payload["data"]["sha256"], "a" * 64)

    def test_aggregates_usage_and_reports_progress(self):
        output = StringIO()
        llm = FakeLLM(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call-usage",
                            name="echo",
                            arguments={"value": "hello"},
                        )
                    ],
                    finish_reason="tool_calls",
                    usage=TokenUsage(
                        prompt_tokens=100,
                        completion_tokens=20,
                        total_tokens=120,
                        cached_tokens=10,
                        reported=True,
                    ),
                ),
                ModelResponse(
                    content="Done",
                    tool_calls=[],
                    finish_reason="stop",
                    usage=TokenUsage(
                        prompt_tokens=140,
                        completion_tokens=10,
                        total_tokens=150,
                        reasoning_tokens=4,
                        reported=True,
                    ),
                ),
            ]
        )

        result = make_runner(
            llm,
            progress=ConsoleProgressReporter(output),
        ).run("Track usage")

        self.assertEqual(result.usage.prompt_tokens, 240)
        self.assertEqual(result.usage.completion_tokens, 30)
        self.assertEqual(result.usage.total_tokens, 270)
        self.assertEqual(result.usage.cached_tokens, 10)
        self.assertEqual(result.usage.reasoning_tokens, 4)
        progress_text = output.getvalue()
        self.assertIn("итерация 1/3", progress_text)
        self.assertIn("[tool] echo: запуск", progress_text)
        self.assertIn("всего 270", progress_text)


if __name__ == "__main__":
    unittest.main()
