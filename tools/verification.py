from typing import Any

from core.models import ApprovalRequest
from execution.verification import VerificationCommand, VerificationRunner
from tools.base import BaseTool, ToolRisk

VERIFICATION_ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4096,
            "description": (
                "Directory relative to the workspace in which to run the "
                "verification command. Defaults to the workspace root."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 300,
            "description": "Maximum execution time in seconds.",
        },
    },
    "additionalProperties": False,
}


class VerificationTool(BaseTool):
    risk = ToolRisk.COMMAND

    def __init__(self, runner: VerificationRunner):
        self._runner = runner

    def _parameters(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, int | None]:
        return (
            arguments.get("path", "."),
            arguments.get("timeout_seconds"),
        )

    def _approval(
        self,
        *,
        title: str,
        command: VerificationCommand,
        fingerprint: str,
    ) -> ApprovalRequest:
        return ApprovalRequest(
            tool_name=self.name,
            title=title,
            fingerprint=fingerprint,
            command=command.argv,
            working_directory=str(command.working_directory),
        )


class RunTestsTool(VerificationTool):
    name = "run_tests"
    description = (
        "Runs the application's fixed test command inside a selected "
        "workspace directory. The user must approve every run. The model "
        "cannot select or alter the executable."
    )
    arguments_schema = VERIFICATION_ARGUMENTS_SCHEMA

    def approval_request(
        self,
        arguments: dict[str, Any],
        fingerprint: str,
    ) -> ApprovalRequest:
        path, timeout = self._parameters(arguments)
        command = self._runner.prepare_tests(path, timeout)
        return self._approval(
            title="Запустить тесты?",
            command=command,
            fingerprint=fingerprint,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path, timeout = self._parameters(arguments)
        return self._runner.run_tests(path, timeout)


class RunLinterTool(VerificationTool):
    name = "run_linter"
    description = (
        "Runs the application's fixed Ruff check command inside a selected "
        "workspace directory. The user must approve every run. It checks "
        "code but never applies automatic fixes."
    )
    arguments_schema = VERIFICATION_ARGUMENTS_SCHEMA

    def approval_request(
        self,
        arguments: dict[str, Any],
        fingerprint: str,
    ) -> ApprovalRequest:
        path, timeout = self._parameters(arguments)
        command = self._runner.prepare_linter(path, timeout)
        return self._approval(
            title="Запустить линтер?",
            command=command,
            fingerprint=fingerprint,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path, timeout = self._parameters(arguments)
        return self._runner.run_linter(path, timeout)
