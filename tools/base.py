from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol

from core.models import ApprovalRequest


class ToolRisk(Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    COMMAND = "command"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class Tool(Protocol):
    name: str
    description: str
    risk: ToolRisk
    arguments_schema: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        ...

    def approval_request(
        self,
        arguments: dict[str, Any],
        fingerprint: str,
    ) -> ApprovalRequest:
        ...

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class BaseTool(ABC):
    name: str
    description: str
    risk: ToolRisk
    arguments_schema: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_schema,
            },
        }

    def approval_request(
        self,
        arguments: dict[str, Any],
        fingerprint: str,
    ) -> ApprovalRequest:
        target = arguments.get("path")
        return ApprovalRequest(
            tool_name=self.name,
            title=f"Allow tool {self.name}?",
            fingerprint=fingerprint,
            target=target if isinstance(target, str) else None,
        )

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class ToolValidationError(Exception):
    pass


class ToolExecutionError(Exception):
    pass
