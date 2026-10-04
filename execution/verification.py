from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from config import VerificationSettings
from security.workspace import Workspace
from tools.base import ToolExecutionError

logger = logging.getLogger("coding_agent.execution.verification")


@dataclass(frozen=True)
class VerificationCommand:
    kind: str
    argv: tuple[str, ...]
    working_directory: Path
    relative_path: str
    timeout_seconds: float


class _BoundedOutputCollector:
    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._head_limit = round(max_bytes * 0.6)
        self._tail_limit = max_bytes - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        remaining = chunk
        head_space = self._head_limit - len(self._head)
        if head_space > 0:
            self._head.extend(remaining[:head_space])
            remaining = remaining[head_space:]
        if not remaining or self._tail_limit == 0:
            return
        self._tail.extend(remaining)
        if len(self._tail) > self._tail_limit:
            del self._tail[: -self._tail_limit]

    def text(self) -> tuple[str, bool]:
        truncated = self.total_bytes > self._max_bytes
        head = bytes(self._head).decode("utf-8", errors="replace")
        tail = bytes(self._tail).decode("utf-8", errors="replace")
        if not truncated:
            return head + tail, False
        return head + "\n...<output truncated>...\n" + tail, True


class VerificationRunner:
    def __init__(
        self,
        workspace: Workspace,
        settings: VerificationSettings,
    ):
        self._workspace = workspace
        self._settings = settings
        if not settings.test_command or not settings.lint_command:
            raise ValueError("Verification commands must not be empty")
        if settings.timeout_seconds <= 0:
            raise ValueError("Verification timeout must be positive")
        if settings.max_output_characters <= 0:
            raise ValueError("Verification output limit must be positive")

    def prepare_tests(
        self,
        path: str = ".",
        timeout_seconds: float | None = None,
    ) -> VerificationCommand:
        return self._prepare(
            kind="tests",
            command=self._settings.test_command,
            path=path,
            timeout_seconds=timeout_seconds,
        )

    def prepare_linter(
        self,
        path: str = ".",
        timeout_seconds: float | None = None,
    ) -> VerificationCommand:
        return self._prepare(
            kind="linter",
            command=self._settings.lint_command,
            path=path,
            timeout_seconds=timeout_seconds,
        )

    def run_tests(
        self,
        path: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict:
        return self._execute(self.prepare_tests(path, timeout_seconds))

    def run_linter(
        self,
        path: str = ".",
        timeout_seconds: float | None = None,
    ) -> dict:
        return self._execute(self.prepare_linter(path, timeout_seconds))

    def _prepare(
        self,
        *,
        kind: str,
        command: tuple[str, ...],
        path: str,
        timeout_seconds: float | None,
    ) -> VerificationCommand:
        working_directory = self._workspace.resolve_directory(path)
        timeout = (
            self._settings.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0:
            raise ToolExecutionError("Verification timeout must be positive")

        executable = shutil.which(
            command[0],
            path=self._sanitized_environment().get("PATH"),
        )
        if executable is None:
            raise ToolExecutionError(
                f"Configured executable is not available: {command[0]}"
            )

        argv = (executable, *command[1:])
        return VerificationCommand(
            kind=kind,
            argv=argv,
            working_directory=working_directory,
            relative_path=self._workspace.relative_path(working_directory),
            timeout_seconds=timeout,
        )

    def _execute(self, command: VerificationCommand) -> dict:
        started = perf_counter()
        timed_out = False
        try:
            process = subprocess.Popen(
                command.argv,
                cwd=command.working_directory,
                env=self._sanitized_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            raise ToolExecutionError(
                "Configured verification command could not be started"
            ) from exc

        stdout_collector = _BoundedOutputCollector(self._settings.max_output_characters)
        stderr_collector = _BoundedOutputCollector(self._settings.max_output_characters)
        output_threads = (
            self._start_output_reader(process.stdout, stdout_collector),
            self._start_output_reader(process.stderr, stderr_collector),
        )
        try:
            process.wait(timeout=command.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process(process)
            process.wait()
        for thread in output_threads:
            thread.join()

        duration = perf_counter() - started
        stdout, stdout_truncated = stdout_collector.text()
        stderr, stderr_truncated = stderr_collector.text()
        stdout, stderr, combined_truncated = _truncate_outputs(
            stdout,
            stderr,
            self._settings.max_output_characters,
        )
        output_truncated = stdout_truncated or stderr_truncated or combined_truncated
        executable_name = Path(command.argv[0]).name
        logger.info(
            "verification=%s executable=%s cwd=%s exit_code=%s "
            "timed_out=%s duration=%.3f stdout_bytes=%d stderr_bytes=%d "
            "output_truncated=%s",
            command.kind,
            executable_name,
            command.relative_path,
            process.returncode,
            timed_out,
            duration,
            stdout_collector.total_bytes,
            stderr_collector.total_bytes,
            output_truncated,
        )
        return {
            "kind": command.kind,
            "path": command.relative_path,
            "passed": process.returncode == 0 and not timed_out,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": output_truncated,
            "duration_seconds": round(duration, 3),
        }

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
        process.kill()

    @staticmethod
    def _start_output_reader(
        stream,
        collector: _BoundedOutputCollector,
    ) -> threading.Thread:
        if stream is None:
            raise ToolExecutionError("Verification output pipe is unavailable")

        def drain() -> None:
            try:
                while chunk := stream.read(8192):
                    collector.add(chunk)
            finally:
                stream.close()

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _sanitized_environment() -> dict[str, str]:
        allowed_names = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "TEMP",
            "TERM",
            "TMP",
            "TMPDIR",
            "VIRTUAL_ENV",
        }
        environment = {
            name: value for name, value in os.environ.items() if name in allowed_names
        }
        environment["UV_CACHE_DIR"] = str(
            Path(tempfile.gettempdir()) / "coding-agent-uv-cache"
        )
        return environment


def _truncate_outputs(
    stdout: str,
    stderr: str,
    max_characters: int,
) -> tuple[str, str, bool]:
    total_characters = len(stdout) + len(stderr)
    if total_characters <= max_characters:
        return stdout, stderr, False

    stdout_budget = round(max_characters * len(stdout) / total_characters)
    stderr_budget = max_characters - stdout_budget
    return (
        _truncate_text(stdout, stdout_budget),
        _truncate_text(stderr, stderr_budget),
        True,
    )


def _truncate_text(value: str, budget: int) -> str:
    if len(value) <= budget:
        return value
    if budget <= 0:
        return ""

    marker = "\n...<output truncated>...\n"
    if budget <= len(marker):
        return value[:budget]
    available = budget - len(marker)
    head_size = round(available * 0.6)
    tail_size = available - head_size
    tail = value[-tail_size:] if tail_size else ""
    return value[:head_size] + marker + tail
