from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv('API_KEY')
BASE_URL=os.getenv('BASE_URL')


IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Reads the full contents of a text file. "
                "Use this tool before editing a file or when you need to inspect "
                "the implementation of a specific module. "
                "The path must point to an existing text file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file that should be read. "
                            "Prefer a path relative to the project root, "
                            "for example: 'src/auth.py'."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Lists files and directories recursively starting from the specified "
                "directory. Use this tool to inspect the project structure and find "
                "files that may be relevant to the user's task. "
                "Ignored directories are excluded automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": (
                            "Directory from which the recursive listing should start. "
                            "Use '.' to inspect the project root."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": (
                            "Maximum recursion depth. Use a small value such as 2 or 3 "
                            "for an initial project overview. Increase it only when "
                            "deeper inspection is necessary."
                        ),
                        "minimum": 0,
                        "maximum": 10,
                        "default": 3,
                    },
                },
                "required": ["root"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replaces the entire contents of a text file with new content. "
                "Always read the file with read_file before using this tool. "
                "The content argument must contain the complete final contents of "
                "the file, not only the changed fragment. "
                "Use this tool only when an actual file modification is required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file that should be overwritten. "
                            "Prefer a path relative to the project root, "
                            "for example: 'src/auth.py'."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Complete new contents of the file. "
                            "This value fully replaces the existing file contents, "
                            "so all unchanged code must also be included."
                        ),
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = """
You are a coding agent working with a software project.

Use the provided tools to inspect and modify project files.

Rules:
1. Inspect the project structure before assuming where relevant code is located.
2. Never invent the contents of a file that you have not read.
3. Always use read_file before editing an existing file.
4. The edit_file tool replaces the entire file, so its content argument must include
   the complete final file contents.
5. Make only changes required for the user's task.
6. Do not claim that a file was modified until edit_file reports success.
7. If a tool returns an error, analyze the error and choose a safe next action.
8. When the task is complete, respond normally without calling another tool.
"""