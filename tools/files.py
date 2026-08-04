from typing import Any

from security.workspace import Workspace
from tools.base import BaseTool, ToolRisk


class WorkspaceTool(BaseTool):
    def __init__(self, workspace: Workspace):
        self._workspace = workspace


class ListFilesTool(WorkspaceTool):
    name = "list_files"
    description = (
        "Lists files and directories inside the workspace. "
        "Use this first to understand the project structure. "
        "Forbidden and generated directories are excluded."
    )
    risk = ToolRisk.READ_ONLY
    arguments_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory relative to the workspace root. "
                    "Use '.' for the workspace root."
                ),
            },
            "max_depth": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "default": 3,
            },
            "max_entries": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5000,
                "default": 500,
            },
        },
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.list_entries(
            arguments.get("path", "."),
            max_depth=arguments.get("max_depth", 3),
            max_entries=arguments.get("max_entries", 500),
        )


class ReadFileTool(WorkspaceTool):
    name = "read_file"
    description = (
        "Reads a UTF-8 text file inside the workspace and returns "
        "the requested line range, total line count, and SHA-256 hash."
    )
    risk = ToolRisk.READ_ONLY
    arguments_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "File path relative to the workspace root.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.read_text(
            arguments["path"],
            start_line=arguments.get("start_line", 1),
            end_line=arguments.get("end_line"),
        )


class SearchTextTool(WorkspaceTool):
    name = "search_text"
    description = (
        "Searches for plain text inside workspace files. "
        "Use it to find symbols, imports, configuration, and references "
        "without reading every file."
    )
    risk = ToolRisk.READ_ONLY
    arguments_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Plain text to search for.",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "File or directory relative to the workspace.",
            },
            "glob": {
                "type": "string",
                "default": "*.py",
                "description": "File glob, for example '*.py' or '*.toml'.",
            },
            "case_sensitive": {
                "type": "boolean",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "default": 50,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.search_text(
            arguments["query"],
            user_path=arguments.get("path", "."),
            glob=arguments.get("glob", "*.py"),
            case_sensitive=arguments.get(
                "case_sensitive",
                False,
            ),
            max_results=arguments.get("max_results", 50),
        )


class CreateFileTool(WorkspaceTool):
    name = "create_file"
    description = (
        "Creates a new UTF-8 text file inside an existing workspace "
        "directory. The operation fails rather than overwriting an "
        "existing file."
    )
    risk = ToolRisk.WORKSPACE_WRITE
    arguments_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "description": "New file path relative to the workspace.",
            },
            "content": {
                "type": "string",
                "maxLength": 1_000_000,
                "description": "Complete UTF-8 content for the new file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.create_text(
            arguments["path"],
            arguments["content"],
        )


class EditFileTool(WorkspaceTool):
    name = "edit_file"
    description = (
        "Safely edits an existing UTF-8 text file by replacing an exact "
        "text fragment. Read the file immediately before editing and pass "
        "the SHA-256 returned by read_file. The edit fails if the file "
        "changed or the fragment count is unexpected."
    )
    risk = ToolRisk.WORKSPACE_WRITE
    arguments_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "description": "Existing file path relative to the workspace.",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1_000_000,
                "description": "Exact text fragment to replace.",
            },
            "new_text": {
                "type": "string",
                "maxLength": 1_000_000,
                "description": "Replacement text; may be empty to remove it.",
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "SHA-256 returned by the latest read_file call.",
            },
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1000,
                "default": 1,
                "description": "Required number of old_text occurrences.",
            },
        },
        "required": [
            "path",
            "old_text",
            "new_text",
            "expected_sha256",
        ],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.edit_text(
            arguments["path"],
            old_text=arguments["old_text"],
            new_text=arguments["new_text"],
            expected_sha256=arguments["expected_sha256"],
            expected_replacements=arguments.get(
                "expected_replacements",
                1,
            ),
        )
