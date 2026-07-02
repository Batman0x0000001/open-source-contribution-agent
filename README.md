# Open Source Contribution Agent Harness

This project is a staged Python CLI implementation of an open source contribution agent harness.

Current stage: **s01 - minimal agent loop + bash**.

## What s01 includes

- `osc-agent --repo <path>` interactive CLI.
- `.env` loading for Anthropic-compatible settings.
- A minimal Anthropic Messages-style agent loop.
- A single `bash` tool executed with the target repo as its working directory.
- Command output truncation to 50,000 characters, with a 200-character preview printed during execution.

Later stages will add file tools, permissions, hooks, todo planning, context compaction, memory, task systems, and worktree isolation.

## S02 update: repo file tools and handler map

S02 extends the s01 loop without changing its core shape. The main loop still calls the model, appends the assistant message, executes requested tools, appends `tool_result`, and continues until the model stops.

New in s02:

- Tool schema registration through `TOOLS`.
- Tool dispatch through `build_tool_handlers(repo_root)`.
- Repo file tools:
  - `read_file(path, limit, offset)`
  - `write_file(path, content)`
  - `edit_file(path, old_text, new_text)`
  - `glob(pattern)`
- Read-only git tools:
  - `git_status()`
  - `git_diff()`
  - `git_log(limit)`
- A lightweight `inspect_repo()` tool for `README*`, `CONTRIBUTING*`, `pyproject.toml`, `package.json`, and test directories.

## S03 update: permission boundaries

S03 adds a small permission layer while keeping tool execution local and reviewable. It does not add hooks or trace yet; those belong to S04.

New in s03:

- `harness/permissions.py` defines structured permission decisions:
  - `allow`
  - `deny`
  - `ask`
- `safe_repo_path(repo_root, path)` resolves file paths and blocks path escape.
- File tools now use `safe_repo_path` before reading, writing, or editing.
- `write_file` and `edit_file` check write size before changing files.
- `bash` checks commands before execution:
  - dangerous commands such as `git push`, `gh pr create`, `sudo`, `shutdown`, `reboot`, `mkfs`, and `dd if=` are denied.
  - suspicious commands such as `pip install`, `npm install`, `git commit`, and deletion commands require explicit confirmation.
- Because there is no approval interaction in s03, `ask` returns `Permission required: ...` and blocks execution for now.

## Project layout

```text
osc_agent/
  cli.py              # CLI entry point
  agent_loop.py       # Anthropic-style loop and handler-map dispatch
  config.py           # .env and Anthropic client setup
  harness/
    permissions.py    # path, shell, and write permission decisions
  tools/
    shell.py          # bash tool
    files.py          # read/write/edit/glob file tools
    git.py            # read-only git tools
    repo.py           # lightweight repo inspection
tests/
  test_agent_loop.py
  test_file_tools.py
  test_permissions.py
  test_shell.py
```

## Configuration

Copy `.env.example` to `.env` and fill in your provider values:

```env
ANTHROPIC_API_KEY=
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
MODEL_ID=deepseek-v4-pro
FALLBACK_MODEL_ID=
```

## Run

With conda:

```sh
conda activate osc-agent
python -m pip install -r requirements.txt
```

```sh
osc-agent --repo /path/to/repo
```

For local development:

```sh
python -m osc_agent.cli --repo /path/to/repo
```

## Verify

```sh
python -m pytest tests/test_permissions.py tests/test_file_tools.py
python -m pytest tests
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py
```

On Windows, if pytest temporary directories are locked by the shell, use a project-local temp directory:

```sh
python -m pytest tests --basetemp .pytest-local -p no:cacheprovider
```
