import logging
from typing import Any

from core.models import ToolCall, ToolResult
from security.policy import ToolPolicy
from security.workspace import WorkspaceConflict, WorkspaceViolation
from tools.base import Tool


logger = logging.getLogger("coding_agent.tools.registry")


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
    errors: list[str] = []
    _validate_schema_value(arguments, schema, "arguments", errors)
    return "\n".join(errors) if errors else None


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    location: str,
    errors: list[str],
) -> None:
    expected_type_name = schema.get("type")
    if expected_type_name is not None:
        expected_python_type = JSON_TYPES.get(expected_type_name)
        if expected_python_type is None:
            errors.append(f"Unsupported schema type: {expected_type_name}")
            return
        if not _matches_json_type(
            value,
            expected_type_name,
            expected_python_type,
        ):
            errors.append(
                f"Invalid type for '{location}': expected "
                f"{expected_type_name}, received {type(value).__name__}"
            )
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"Invalid value for '{location}'")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(
                f"'{location}' must have length at least "
                f"{schema['minLength']}"
            )
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"'{location}' is too long")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(
                f"'{location}' must be at least {schema['minimum']}"
            )
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(
                f"'{location}' must be at most {schema['maximum']}"
            )

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(
                f"'{location}' must contain at least {schema['minItems']} items"
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(
                f"'{location}' must contain at most {schema['maxItems']} items"
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    item_schema,
                    f"{location}[{index}]",
                    errors,
                )

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for required_name in schema.get("required", []):
            if required_name not in value:
                errors.append(
                    f"Missing required argument: {location}.{required_name}"
                )
        for name, nested_value in value.items():
            nested_schema = properties.get(name)
            if nested_schema is None:
                if not schema.get("additionalProperties", True):
                    errors.append(f"Unknown argument: {location}.{name}")
                continue
            _validate_schema_value(
                nested_value,
                nested_schema,
                f"{location}.{name}",
                errors,
            )


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
            logger.warning(
                "tool=%s authorization_denied requires_approval=%s",
                tool.name,
                decision.requires_approval,
            )
            return ToolResult.failure(
                tool_call_id=call.id,
                code="PERMISSION_DENIED",
                message=decision.reason or "Operation is not allowed",
            )

        try:
            data = tool.execute(arguments)
        except WorkspaceConflict as exc:
            logger.info("tool=%s workspace_conflict", tool.name)
            return ToolResult.failure(
                tool_call_id=call.id,
                code="WORKSPACE_CONFLICT",
                message=str(exc),
            )
        except WorkspaceViolation as exc:
            logger.warning("tool=%s workspace_violation", tool.name)
            return ToolResult.failure(
                tool_call_id=call.id,
                code="WORKSPACE_VIOLATION",
                message=str(exc),
            )
        except Exception:
            logger.exception("tool=%s unexpected_execution_error", tool.name)
            return ToolResult.failure(
                tool_call_id=call.id,
                code="TOOL_EXECUTION_ERROR",
                message="Tool execution failed",
            )

        return ToolResult.success(
            tool_call_id=call.id,
            data=data,
        )
