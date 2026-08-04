from pathlib import Path
import tempfile
import unittest

from agent.runner import AgentRunner
from config import AgentContext, AgentSettings
from core.models import AgentStatus, ModelResponse, ToolCall
from security.policy import ToolPolicy
from security.workspace import Workspace
from tools.files import ListFilesTool, ReadFileTool, SearchTextTool
from tools.registry import ToolRegistry


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.second_request_messages = None

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="list-1",
                        name="list_files",
                        arguments={"path": ".", "max_depth": 2},
                    )
                ],
                finish_reason="tool_calls",
            )

        self.second_request_messages = list(messages)
        return ModelResponse(
            content="The workspace contains app.py.",
            tool_calls=[],
            finish_reason="stop",
        )


class MVPIntegrationTests(unittest.TestCase):
    def test_read_only_analysis_flow(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            (root / "app.py").write_text(
                "print('hello')\n",
                encoding="utf-8",
            )

            workspace = Workspace(root)
            registry = ToolRegistry(ToolPolicy("read_only"))
            registry.register(ListFilesTool(workspace))
            registry.register(ReadFileTool(workspace))
            registry.register(SearchTextTool(workspace))
            llm = FakeLLM()
            runner = AgentRunner(
                llm=llm,
                tools=registry,
                settings=AgentSettings(
                    max_iterations=3,
                    max_tool_calls=5,
                    max_context_characters=100_000,
                ),
                context=AgentContext(
                    workspace_root=root,
                    permission_mode="read_only",
                ),
            )

            result = runner.run("Analyze this project")

            self.assertEqual(result.status, AgentStatus.COMPLETED)
            self.assertEqual(result.tool_calls_count, 1)
            self.assertEqual(llm.calls, 2)
            self.assertEqual(
                llm.second_request_messages[-1]["role"],
                "tool",
            )
            self.assertIn(
                "app.py",
                llm.second_request_messages[-1]["content"],
            )


if __name__ == "__main__":
    unittest.main()
