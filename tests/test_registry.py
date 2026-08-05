import tempfile
from pathlib import Path
import unittest

from core.models import ToolCall
from security.policy import ToolPolicy
from security.workspace import Workspace
from tools.base import BaseTool, ToolRisk
from tools.files import (
    CreateFileTool,
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from tools.registry import ToolRegistry


class WriteTool(BaseTool):
    name = "write"
    description = "Test-only write tool."
    risk = ToolRisk.WORKSPACE_WRITE
    arguments_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return {"changed": True}


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.root = root
        (root / "main.py").write_text(
            "print('ok')\n",
            encoding="utf-8",
        )
        workspace = Workspace(root)
        self.registry = ToolRegistry(
            ToolPolicy("read_only")
        )
        self.registry.register(ListFilesTool(workspace))
        self.registry.register(ReadFileTool(workspace))
        self.registry.register(CreateFileTool(workspace))
        self.registry.register(EditFileTool(workspace))
        self.registry.register(WriteFileTool(workspace))
        self.registry.register(WriteTool())

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_executes_registered_tool(self):
        result = self.registry.execute(
            ToolCall(
                id="call-1",
                name="read_file",
                arguments={"path": "main.py"},
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data["path"], "main.py")

    def test_rejects_unknown_tool(self):
        result = self.registry.execute(
            ToolCall(
                id="call-2",
                name="delete_everything",
                arguments={},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "UNKNOWN_TOOL")

    def test_rejects_invalid_arguments(self):
        result = self.registry.execute(
            ToolCall(
                id="call-3",
                name="read_file",
                arguments={"path": 123},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "INVALID_ARGUMENTS")

    def test_rejects_malformed_model_arguments(self):
        result = self.registry.execute(
            ToolCall(
                id="call-4",
                name="read_file",
                arguments=None,
                error="Invalid JSON",
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "INVALID_ARGUMENTS")

    def test_read_only_policy_rejects_write_tool(self):
        result = self.registry.execute(
            ToolCall(
                id="call-5",
                name="write",
                arguments={},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "PERMISSION_DENIED")

    def test_read_only_policy_rejects_create_file(self):
        result = self.registry.execute(
            ToolCall(
                id="call-6",
                name="create_file",
                arguments={
                    "path": "created.py",
                    "content": "value = 1\n",
                },
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "PERMISSION_DENIED")
        self.assertFalse((self.root / "created.py").exists())

    def test_workspace_write_policy_allows_create_and_edit(self):
        workspace = Workspace(self.root)
        registry = ToolRegistry(ToolPolicy("workspace_write"))
        registry.register(ReadFileTool(workspace))
        registry.register(CreateFileTool(workspace))
        registry.register(EditFileTool(workspace))
        registry.register(WriteFileTool(workspace))

        created = registry.execute(
            ToolCall(
                id="call-7",
                name="create_file",
                arguments={
                    "path": "created.py",
                    "content": "value = 1\n",
                },
            )
        )
        edited = registry.execute(
            ToolCall(
                id="call-8",
                name="edit_file",
                arguments={
                    "path": "created.py",
                    "edits": [
                        {
                            "old_text": "value = 1",
                            "new_text": "value = 2",
                        }
                    ],
                    "expected_sha256": created.data["sha256"],
                },
            )
        )
        written = registry.execute(
            ToolCall(
                id="call-8-write",
                name="write_file",
                arguments={
                    "path": "created.py",
                    "content": "value = 3\n",
                    "expected_sha256": edited.data["sha256"],
                },
            )
        )

        self.assertTrue(created.ok)
        self.assertTrue(edited.ok)
        self.assertTrue(written.ok)
        self.assertEqual(
            (self.root / "created.py").read_text(encoding="utf-8"),
            "value = 3\n",
        )

    def test_edit_conflict_has_specific_error_code(self):
        workspace = Workspace(self.root)
        registry = ToolRegistry(ToolPolicy("workspace_write"))
        registry.register(EditFileTool(workspace))

        result = registry.execute(
            ToolCall(
                id="call-9",
                name="edit_file",
                arguments={
                    "path": "main.py",
                    "edits": [
                        {
                            "old_text": "print('ok')",
                            "new_text": "print('changed')",
                        }
                    ],
                    "expected_sha256": "0" * 64,
                },
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "WORKSPACE_CONFLICT")

    def test_rejects_invalid_nested_edit_before_execution(self):
        workspace = Workspace(self.root)
        registry = ToolRegistry(ToolPolicy("workspace_write"))
        registry.register(EditFileTool(workspace))
        current = workspace.read_text("main.py")

        result = registry.execute(
            ToolCall(
                id="call-10",
                name="edit_file",
                arguments={
                    "path": "main.py",
                    "edits": [{"old_text": "print('ok')"}],
                    "expected_sha256": current["sha256"],
                },
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "INVALID_ARGUMENTS")


if __name__ == "__main__":
    unittest.main()
