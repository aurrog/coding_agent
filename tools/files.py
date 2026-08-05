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
        "Atomically applies one or more exact-text replacements to an "
        "existing UTF-8 file. Batch all planned replacements for the same "
        "file in one call. Read the file first and pass its SHA-256. No "
        "change is written if any replacement or hash check fails."
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
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": 100,
                "description": (
                    "Ordered replacements applied sequentially and atomically."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1_000_000,
                        },
                        "new_text": {
                            "type": "string",
                            "maxLength": 1_000_000,
                        },
                        "expected_replacements": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 1,
                        },
                    },
                    "required": ["old_text", "new_text"],
                    "additionalProperties": False,
                },
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "SHA-256 returned by the latest read_file call.",
            },
        },
        "required": [
            "path",
            "edits",
            "expected_sha256",
        ],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.edit_text_many(
            arguments["path"],
            edits=arguments["edits"],
            expected_sha256=arguments["expected_sha256"],
        )


class WriteFileTool(WorkspaceTool):
    name = "write_file"
    description = (
        "Atomically replaces the complete content of an existing UTF-8 "
        "file. Use this instead of many edits for a substantial rewrite. "
        "Read the file first and pass its SHA-256; the write fails if the "
        "file changed. It never creates a new file."
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
            "content": {
                "type": "string",
                "maxLength": 1_000_000,
                "description": "Complete replacement UTF-8 content.",
            },
            "expected_sha256": {
                "type": "string",
                "minLength": 64,
                "maxLength": 64,
                "description": "SHA-256 returned by the latest read_file call.",
            },
        },
        "required": ["path", "content", "expected_sha256"],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._workspace.write_text(
            arguments["path"],
            content=arguments["content"],
            expected_sha256=arguments["expected_sha256"],
        )
