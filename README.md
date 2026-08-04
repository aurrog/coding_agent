# Workspace coding agent

Minimal OpenAI-compatible agent for safe repository analysis and controlled
text-file changes. It can inspect a workspace, read and search UTF-8 source
files, create new files, and make conflict-checked exact-text edits. It cannot
run commands.

## Architecture

- `main.py` assembles the application and provides the CLI.
- `agent/runner.py` owns the model/tool loop and execution budgets.
- `agent/llm.py` adapts an OpenAI-compatible API to internal response models.
- `agent/prompts.py` contains the agent workflow and safety instructions.
- `core/models.py` defines tool calls, results, and agent results.
- `tools/registry.py` validates, authorizes, and dispatches tool calls.
- `tools/files.py` contains the file tools exposed to the model.
- `security/workspace.py` confines all file access to one workspace.
- `security/policy.py` enforces tool risk permissions.

## Configuration

Create a local `.env` file:

```dotenv
API_KEY=your-api-key
BASE_URL=https://your-openai-compatible-endpoint/v1
MODEL=your-model-name

TIMEOUT_SECONDS=60
MAX_RETRIES=2
MAX_ITERATIONS=8
MAX_TOOL_CALLS=20
MAX_CONTEXT_CHARACTERS=200000
PERMISSION_MODE=read_only
```

`TIMEOUT_SETTINGS` is accepted as a compatibility alias for
`TIMEOUT_SECONDS`.

## Run

```bash
uv run python main.py
```

The program first asks for the workspace directory and then for the analysis
request. The model is read from `MODEL` in `.env`.

## Current tools

- `list_files`: inspect the workspace tree with depth and result limits.
- `read_file`: read a line range and receive file metadata and SHA-256.
- `search_text`: search plain text using a file glob.
- `create_file`: create a new UTF-8 file without overwriting an existing path.
- `edit_file`: replace exact text after verifying the file's current SHA-256.

The workspace layer blocks path traversal, access outside the selected root,
known secret/generated directories, oversized files, and non-UTF-8 reads.
Edits are written atomically and fail if the file changed after it was read.

`PERMISSION_MODE=read_only` is the safe default and rejects both write tools.
Set `PERMISSION_MODE=workspace_write` to allow creating and editing files inside
the selected workspace. Existing files are never overwritten by `create_file`.

## Tests

```bash
uv run python -m unittest discover -v
```

Tests use fake model responses and temporary workspaces. They do not call a
real model API.
