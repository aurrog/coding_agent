from typing import Any

from core.models import ToolCall, ToolResult
from security.policy import ToolPolicy
from security.workspace import WorkspaceConflict, WorkspaceViolation
from tools.base import Tool


JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def validate_arguments(
    arguments: dict[str, Any] | None,
    schema: dict[str, Any],
) -> str | None:
    if not isinstance(arguments, dict):
        return "Tool arguments must be an object"

    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    allow_additional = schema.get("additionalProperties", True)

    for argument_name in required:
        if argument_name not in arguments:
            errors.append(
                f"Missing required argument: {argument_name}"
            )

    for argument_name, value in arguments.items():
        argument_schema = properties.get(argument_name)
        if argument_schema is None:
            if not allow_additional:
                errors.append(
                    f"Unknown argument: {argument_name}"
                )
            continue

        expected_type_name = argument_schema.get("type")
        if expected_type_name is not None:
            expected_python_type = JSON_TYPES.get(expected_type_name)
            if expected_python_type is None:
                errors.append(
                    f"Unsupported schema type: {expected_type_name}"
                )
                continue
            if not _matches_json_type(
                value,
                expected_type_name,
                expected_python_type,
            ):
                errors.append(
                    f"Invalid type for '{argument_name}': "
                    f"expected {expected_type_name}, "
                    f"received {type(value).__name__}"
                )
                continue

        if "minimum" in argument_schema and value < argument_schema["minimum"]:
            errors.append(
                f"'{argument_name}' must be at least "
                f"{argument_schema['minimum']}"
            )
        if "maximum" in argument_schema and value > argument_schema["maximum"]:
            errors.append(
                f"'{argument_name}' must be at most "
                f"{argument_schema['maximum']}"
            )
        if "minLength" in argument_schema and len(value) < argument_schema["minLength"]:
            errors.append(
                f"'{argument_name}' must have length at least "
                f"{argument_schema['minLength']}"
            )
        if "maxLength" in argument_schema and len(value) > argument_schema["maxLength"]:
            errors.append(
                f"'{argument_name}' is too long"
            )
        if "enum" in argument_schema and value not in argument_schema["enum"]:
            errors.append(
                f"Invalid value for '{argument_name}'"
            )

    return "\n".join(errors) if errors else None


def _matches_json_type(
    value: Any,
    type_name: str,
    python_type: type | tuple[type, ...],
) -> bool:
    if type_name in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


class ToolRegistry:
    def __init__(self, policy: ToolPolicy):
        self._tools: dict[str, Tool] = {}
        self._policy = policy

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool name must not be empty")
        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            tool.schema()
            for tool in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if call.error:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="INVALID_ARGUMENTS",
                message=call.error,
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="UNKNOWN_TOOL",
                message=f"Unknown tool: {call.name}",
            )

        validation_error = validate_arguments(
            call.arguments,
            tool.arguments_schema,
        )
        if validation_error:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="INVALID_ARGUMENTS",
                message=validation_error,
            )

        arguments = call.arguments or {}
        decision = self._policy.authorize(
            tool=tool,
            arguments=arguments,
        )
        if not decision.allowed:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="PERMISSION_DENIED",
                message=decision.reason or "Operation is not allowed",
            )

        try:
            data = tool.execute(arguments)
        except WorkspaceConflict as exc:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="WORKSPACE_CONFLICT",
                message=str(exc),
            )
        except WorkspaceViolation as exc:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="WORKSPACE_VIOLATION",
                message=str(exc),
            )
        except Exception:
            return ToolResult.failure(
                tool_call_id=call.id,
                code="TOOL_EXECUTION_ERROR",
                message="Tool execution failed",
            )

        return ToolResult.success(
            tool_call_id=call.id,
            data=data,
        )
