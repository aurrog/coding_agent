import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import VerificationSettings
from core.models import ToolCall, ToolResult
from execution.verification import VerificationRunner
from security.policy import ToolPolicy
from security.workspace import Workspace, WorkspaceViolation
from tools.registry import PreparedToolCall, ToolRegistry
from tools.verification import RunLinterTool, RunTestsTool


class VerificationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "src").mkdir()
        self.workspace = Workspace(self.root)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def settings(
        self,
        *,
        test_script: str = "print('tests passed')",
        lint_script: str = "print('lint passed')",
        max_output_characters: int = 12_000,
    ) -> VerificationSettings:
        return VerificationSettings(
            test_command=(sys.executable, "-c", test_script),
            lint_command=(sys.executable, "-c", lint_script),
            timeout_seconds=5,
            max_output_characters=max_output_characters,
        )

    def test_runs_fixed_test_and_linter_commands(self):
        runner = VerificationRunner(self.workspace, self.settings())

        tests = runner.run_tests()
        linter = runner.run_linter("src")

        self.assertTrue(tests["passed"])
        self.assertIn("tests passed", tests["stdout"])
        self.assertTrue(linter["passed"])
        self.assertEqual(linter["path"], "src")

    def test_nonzero_exit_is_a_normal_failed_check_result(self):
        runner = VerificationRunner(
            self.workspace,
            self.settings(test_script="import sys; sys.exit(7)"),
        )

        result = runner.run_tests()

        self.assertFalse(result["passed"])
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["timed_out"])

    def test_rejects_working_directory_outside_workspace(self):
        runner = VerificationRunner(self.workspace, self.settings())

        with self.assertRaises(WorkspaceViolation):
            runner.prepare_tests("..")

    def test_times_out_and_stops_process(self):
        runner = VerificationRunner(
            self.workspace,
            self.settings(
                test_script="import time; time.sleep(10)",
            ),
        )

        result = runner.run_tests(timeout_seconds=0.05)

        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"])

    def test_truncates_combined_output(self):
        runner = VerificationRunner(
            self.workspace,
            self.settings(
                test_script=(
                    "import sys; print('x' * 1000); print('y' * 1000, file=sys.stderr)"
                ),
                max_output_characters=53,
            ),
        )

        result = runner.run_tests()

        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(
            len(result["stdout"]) + len(result["stderr"]),
            53,
        )

    def test_does_not_pass_api_credentials_to_process(self):
        script = "import os; print(os.getenv('API_KEY', 'missing'))"
        runner = VerificationRunner(
            self.workspace,
            self.settings(test_script=script),
        )

        with patch.dict(os.environ, {"API_KEY": "very-secret"}):
            result = runner.run_tests()

        self.assertEqual(result["stdout"].strip(), "missing")


class VerificationToolTests(unittest.TestCase):
    def test_requires_approval_and_rejects_arbitrary_command_argument(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Workspace(temporary_directory)
            settings = VerificationSettings(
                test_command=(sys.executable, "-c", "print('ok')"),
                lint_command=(sys.executable, "-c", "print('ok')"),
            )
            runner = VerificationRunner(workspace, settings)
            registry = ToolRegistry(ToolPolicy("workspace_write"))
            registry.register(RunTestsTool(runner))
            registry.register(RunLinterTool(runner))

            invalid = registry.prepare(
                ToolCall(
                    id="invalid-command",
                    name="run_tests",
                    arguments={"command": ["rm", "-rf", "."]},
                )
            )
            prepared = registry.prepare(
                ToolCall(
                    id="valid-tests",
                    name="run_tests",
                    arguments={"path": "."},
                )
            )

            self.assertIsInstance(invalid, ToolResult)
            self.assertEqual(invalid.error.code, "INVALID_ARGUMENTS")
            self.assertIsInstance(prepared, PreparedToolCall)
            self.assertTrue(prepared.requires_approval)
            self.assertEqual(
                Path(prepared.approval_request.command[0]),
                Path(sys.executable),
            )


if __name__ == "__main__":
    unittest.main()
