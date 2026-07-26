from typing import Protocol
from enum import Enum


class Tool(Protocol):
    pass

    def execute():
        ...


class BaseTool:
    name: str
    description: str
    risk: ToolRisk
    arguments_schema: dict

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_schema,
            },
        }

    def execute(self):
        pass


class ToolRisk(Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class ToolValidationError(Exception):
    pass


class ToolExcecutionError(Exception):
    pass
