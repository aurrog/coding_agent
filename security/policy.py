from dataclasses import dataclass
from enum import Enum
from typing import Any

from tools.base import Tool, ToolRisk


class PermissionMode(Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool = False
    reason: str | None = None


class ToolPolicy:
    def __init__(
        self,
        mode: PermissionMode | str,
    ):
        try:
            self._mode = (
                mode
                if isinstance(mode, PermissionMode)
                else PermissionMode(mode)
            )
        except ValueError as exc:
            raise ValueError(
                f"Unknown permission mode: {mode}"
            ) from exc

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def authorize(
        self,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> PolicyDecision:
        del arguments

        if tool.risk == ToolRisk.READ_ONLY:
            return PolicyDecision(allowed=True)

        if tool.risk == ToolRisk.WORKSPACE_WRITE:
            if self._mode == PermissionMode.WORKSPACE_WRITE:
                return PolicyDecision(allowed=True)
            return PolicyDecision(
                allowed=False,
                requires_approval=(
                    self._mode == PermissionMode.APPROVAL_REQUIRED
                ),
                reason="Workspace write permission is required",
            )

        if tool.risk in {
            ToolRisk.EXTERNAL,
            ToolRisk.DESTRUCTIVE,
        }:
            return PolicyDecision(
                allowed=False,
                requires_approval=True,
                reason="Explicit approval is required",
            )

        return PolicyDecision(
            allowed=False,
            reason="Tool risk is not covered by policy",
        )
