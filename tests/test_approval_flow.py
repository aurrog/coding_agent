import unittest
from collections.abc import Iterable
from typing import ClassVar

from agent.runner import AgentRunner
from config import AgentContext, AgentSettings
from core.models import (
    AgentStatus,
    ApprovalDecision,
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
        del tools
        self.received_messages.append(list(messages))
        return next(self._responses)


class FakeApprovalProvider:
    def __init__(self, decision: ApprovalDecision):
        self.decision = decision
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        return self.decision


class CommandTool(BaseTool):
    name = "verify"
    description = "Test-only approved operation."
    risk = ToolRisk.COMMAND
    arguments_schema: ClassVar = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.executions = 0

    def execute(self, arguments):
        self.executions += 1
        return {"path": arguments["path"], "passed": True}


def make_runner(llm, tool, approval):
    registry = ToolRegistry(ToolPolicy("read_only"))
    registry.register(tool)
    return AgentRunner(
        llm=llm,
        tools=registry,
        settings=AgentSettings(
            max_iterations=4,
            max_tool_calls=4,
            max_context_characters=100_000,
        ),
        context=AgentContext(
            workspace_root="workspace",
            permission_mode="read_only",
        ),
        approval=approval,
    )


def tool_response(call_id: str) -> ModelResponse:
    return ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name="verify",
                arguments={"path": "."},
            )
        ],
        finish_reason="tool_calls",
    )


class ApprovalFlowTests(unittest.TestCase):
    def test_executes_command_tool_after_approval(self):
        llm = FakeLLM(
            [
                tool_response("call-approved"),
                ModelResponse("Done", [], "stop"),
            ]
        )
        tool = CommandTool()
        approval = FakeApprovalProvider(ApprovalDecision.APPROVED)

        result = make_runner(llm, tool, approval).run("Verify")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(tool.executions, 1)
        self.assertEqual(len(approval.requests), 1)
        self.assertIn(
            '"passed": true',
            llm.received_messages[1][-1]["content"],
        )

    def test_denial_prevents_execution_and_is_not_requested_twice(self):
        llm = FakeLLM(
            [
                tool_response("call-denied-1"),
                tool_response("call-denied-2"),
                ModelResponse("Not run", [], "stop"),
            ]
        )
        tool = CommandTool()
        approval = FakeApprovalProvider(ApprovalDecision.DENIED)

        result = make_runner(llm, tool, approval).run("Verify")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(tool.executions, 0)
        self.assertEqual(len(approval.requests), 1)
        second_history = str(llm.received_messages[2])
        self.assertEqual(second_history.count("APPROVAL_DENIED"), 2)


if __name__ == "__main__":
    unittest.main()
