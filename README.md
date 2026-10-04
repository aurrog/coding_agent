# Workspace coding agent

Minimal OpenAI-compatible agent for safe repository analysis and controlled
text-file changes. It can inspect a workspace, read and search UTF-8 source
files, create new files, make conflict-checked exact-text edits, and run fixed
test and linter commands after explicit user approval. It cannot run arbitrary
commands.

## Architecture

- `main.py` assembles the application and provides the CLI.
- `agent/runner.py` owns the model/tool loop and execution budgets.
- `agent/llm.py` adapts an OpenAI-compatible API to internal response models.
- `agent/observability.py` provides console progress and rotating system logs.
- `agent/approval.py` isolates interactive approval from the agent loop.
- `agent/prompts.py` contains the agent workflow and safety instructions.
- `core/models.py` defines tool calls, results, and agent results.
- `tools/registry.py` validates, authorizes, and dispatches tool calls.
- `tools/files.py` contains the file tools exposed to the model.
- `tools/verification.py` exposes fixed test and linter operations.
- `execution/verification.py` safely manages verification subprocesses.
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
VERIFICATION_TIMEOUT_SECONDS=120
MAX_COMMAND_OUTPUT_CHARACTERS=12000

SHOW_PROGRESS=true
LOG_LEVEL=INFO
LOG_FILE=agent.log
```

`TIMEOUT_SETTINGS` is accepted as a compatibility alias for
`TIMEOUT_SECONDS`.

## Run

```bash
uv run python main.py
```

The program first asks for the workspace directory and then for the analysis
request. The model is read from `MODEL` in `.env`.

The CLI writes short progress events to stderr: model iterations, tool names,
safe file targets, tool outcomes, durations, and token usage. It never prints
or logs hidden chain-of-thought. `SHOW_PROGRESS=false` disables these messages.

Technical events are written to a rotating log configured by `LOG_FILE`. Each
file is limited to 5 MB and three backups are retained. Logs contain operation
metadata and error codes, but not user prompts, file contents, or tool content
arguments. Token totals come from the provider's response usage; if an
OpenAI-compatible provider omits usage, the CLI explicitly reports that the
statistics are unavailable.

`MAX_ITERATIONS` limits tool-capable model rounds. If every round is consumed
by tool calls, the runner makes one additional finalization request without
tools so the user still receives a final answer. Large generated arguments
such as file contents and edit fragments are compacted in subsequent history.

Test and linter tools always require a separate interactive confirmation,
regardless of `PERMISSION_MODE`. A denied operation is returned to the model as
`APPROVAL_DENIED`, and an identical denied request is not shown again during
the same agent run.

## Current tools

- `list_files`: inspect the workspace tree with depth and result limits.
- `read_file`: read a line range and receive file metadata and SHA-256.
- `search_text`: search plain text using a file glob.
- `create_file`: create a new UTF-8 file without overwriting an existing path.
- `edit_file`: replace exact text after verifying the file's current SHA-256.
- `write_file`: atomically replace a complete existing file after verifying
  its current SHA-256.
- `run_tests`: run the fixed `uv run python -m unittest discover -v` command.
- `run_linter`: run the fixed `ruff check .` command without automatic fixes.

`edit_file` accepts an ordered `edits` array and applies all replacements in a
single atomic operation. Use it to batch small changes to one file; use
`write_file` for a substantial rewrite.

The workspace layer blocks path traversal, access outside the selected root,
known secret/generated directories, oversized files, and non-UTF-8 reads.
Edits are written atomically and fail if the file changed after it was read.

`PERMISSION_MODE=read_only` is the safe default and rejects both write tools.
Set `PERMISSION_MODE=workspace_write` to allow creating and editing files inside
the selected workspace. Existing files are never overwritten by `create_file`.

The model can select only a workspace directory and a bounded timeout for
verification. It cannot provide an executable, shell expression, environment,
or arbitrary command arguments. Commands use `shell=False`, receive a reduced
environment without the agent API key, run with no interactive stdin, and are
terminated as a process group on timeout. Their combined output is truncated
before it is returned to the model to limit context and token growth.

## Tests

```bash
uv run python -m unittest discover -v
```

Tests use fake model responses and temporary workspaces. They do not call a
real model API.
