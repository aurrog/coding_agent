from __future__ import annotations

import shlex
import sys
from typing import Protocol, TextIO

from core.models import ApprovalDecision, ApprovalRequest


class ApprovalProvider(Protocol):
    def request(self, request: ApprovalRequest) -> ApprovalDecision: ...


class DenyAllApprovalProvider:
    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        del request
        return ApprovalDecision.DENIED


class ConsoleApprovalProvider:
    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ):
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stderr

    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        print("\nАгент запрашивает разрешение:", file=self._output)
        print(f"  {request.title}", file=self._output)
        if request.command:
            print(
                f"  Команда: {shlex.join(request.command)}",
                file=self._output,
            )
        if request.working_directory:
            print(
                f"  Рабочая директория: {request.working_directory}",
                file=self._output,
            )
        elif request.target:
            print(f"  Цель: {request.target}", file=self._output)
        print("Разрешить? [y/N]: ", end="", file=self._output, flush=True)
        answer = self._input.readline()
        if answer.strip().casefold() in {"y", "yes", "д", "да"}:
            return ApprovalDecision.APPROVED
        return ApprovalDecision.DENIED
