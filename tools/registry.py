from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

from core.models import ApprovalRequest, ToolCall, ToolResult
from security.policy import ToolPolicy
from security.workspace import WorkspaceConflict, WorkspaceViolation
from tools.base import Tool, ToolExecutionError


logger = logging.getLogger("coding_agent.tools.registry")


@dataclass(frozen=True)
class PreparedToolCall:
    tool_call_id: str
    tool_name: str
    arguments_json: str
    fingerprint: str
    requires_approval: bool
    approval_request: ApprovalRequest | None = None

    def arguments(self) -> dict[str, Any]:
        value = json.loads(self.arguments_json)
        if not isinstance(value, dict):
            raise ValueError("Prepared tool arguments must be an object")
        return value


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

    def prepare(
        self,
        call: ToolCall,
    ) -> PreparedToolCall | ToolResult:
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
        if not decision.allowed and not decision.requires_approval:
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

        arguments_json = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(
            f"{tool.name}\0{arguments_json}".encode("utf-8")
        ).hexdigest()[:16]
        approval_request = None
        if decision.requires_approval:
            try:
                approval_request = tool.approval_request(
                    arguments,
                    fingerprint,
                )
            except WorkspaceViolation as exc:
                return ToolResult.failure(
                    tool_call_id=call.id,
                    code="WORKSPACE_VIOLATION",
                    message=str(exc),
                )
            except ToolExecutionError as exc:
                return ToolResult.failure(
                    tool_call_id=call.id,
                    code="TOOL_PREPARATION_ERROR",
                    message=str(exc),
                )
            except Exception:
                logger.exception(
                    "tool=%s approval_request_failed",
                    tool.name,
                )
                return ToolResult.failure(
                    tool_call_id=call.id,
                    code="TOOL_PREPARATION_ERROR",
                    message="Tool approval request could not be prepared",
                )

        return PreparedToolCall(
            tool_call_id=call.id,
            tool_name=tool.name,
            arguments_json=arguments_json,
            fingerprint=fingerprint,
            requires_approval=decision.requires_approval,
            approval_request=approval_request,
        )

    def execute(
        self,
        call: ToolCall,
    ) -> ToolResult:
        prepared = self.prepare(call)
        if isinstance(prepared, ToolResult):
            return prepared
        return self.execute_prepared(prepared)

    def execute_prepared(
        self,
        prepared: PreparedToolCall,
        *,
        approved: bool | None = None,
    ) -> ToolResult:
        if prepared.requires_approval and approved is None:
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="APPROVAL_REQUIRED",
                message="Explicit user approval is required",
            )
        if prepared.requires_approval and not approved:
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="APPROVAL_DENIED",
                message="User declined the operation",
            )

        tool = self._tools.get(prepared.tool_name)
        if tool is None:
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="UNKNOWN_TOOL",
                message=f"Unknown tool: {prepared.tool_name}",
            )

        try:
            arguments = prepared.arguments()
        except (json.JSONDecodeError, ValueError):
            logger.exception(
                "tool=%s invalid_prepared_arguments",
                prepared.tool_name,
            )
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="INVALID_ARGUMENTS",
                message="Prepared tool arguments are invalid",
            )

        try:
            data = tool.execute(arguments)
        except WorkspaceConflict as exc:
            logger.info("tool=%s workspace_conflict", tool.name)
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="WORKSPACE_CONFLICT",
                message=str(exc),
            )
        except WorkspaceViolation as exc:
            logger.warning("tool=%s workspace_violation", tool.name)
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="WORKSPACE_VIOLATION",
                message=str(exc),
            )
        except ToolExecutionError as exc:
            logger.warning("tool=%s execution_error", tool.name)
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="TOOL_EXECUTION_ERROR",
                message=str(exc),
            )
        except Exception:
            logger.exception("tool=%s unexpected_execution_error", tool.name)
            return ToolResult.failure(
                tool_call_id=prepared.tool_call_id,
                code="TOOL_EXECUTION_ERROR",
                message="Tool execution failed",
            )

        return ToolResult.success(
            tool_call_id=prepared.tool_call_id,
            data=data,
        )
