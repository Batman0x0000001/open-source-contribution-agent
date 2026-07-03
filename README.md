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

## S04 update: hooks and trace

S04 introduces hook points around tool execution and records an audit trail. The agent loop still keeps the same Anthropic-style shape, but tool calls now pass through hook events.

New in s04:

- `harness/hooks.py` defines a minimal hook registry with:
  - `UserPromptSubmit`
  - `PreToolUse`
  - `PostToolUse`
  - `Stop`
- `PreToolUse` runs permission checks before a handler executes.
- A blocked `PreToolUse` returns a normal `tool_result`, so the model can see why execution stopped.
- `PostToolUse` records tool name, arguments, output preview, latency, and error status.
- `Stop` writes a summary with tool count, failure count, modified files, and stop reason.
- `harness/trace.py` appends audit events to `.osc_agent/traces/session.jsonl`.

## S04 refinement: ask confirmation

The `ask` policy is now completed at the hook layer.

- S03 still decides whether an action is `allow`, `deny`, or `ask`.
- S04 `PreToolUse` handles `ask` by calling a confirmation callback.
- The CLI uses `typer.confirm(..., default=False)`, so pressing Enter is treated as `n`.
- If the user answers `y`, the tool handler runs.
- If the user answers `n`, the tool returns `Permission required: ...` and does not run.
- Confirmation decisions are written to trace as `permission_confirmation` events.
- Direct tool calls still enforce permissions by default; the agent loop disables duplicate tool-level checks because `PreToolUse` has already made the decision.

## S05 update: contribution TodoWrite

S05 adds a planning tool so the agent can maintain an explicit contribution checklist before editing files.

New in s05:

- `harness/todo.py` defines the in-process TODO state and `todo_write(todos)`.
- `todo_write` accepts:
  - a Python list
  - a JSON array string
  - a Python list literal string parsed with `ast.literal_eval`, not `eval`
- Each todo contains `content`, `status`, and optional `evidence`.
- Supported statuses are `pending`, `in_progress`, and `completed`.
- At most one todo can be `in_progress`.
- TODO updates are appended to `.osc_agent/traces/session.jsonl` as `todo_write` events.
- The system prompt now asks the agent to create a contribution plan before modifying files.

Suggested reading order for S05:

1. `learn-claude-code/coding_plan.md` section `s05`
2. `learn-claude-code/s05_todo_write/README.md`
3. `osc_agent/harness/todo.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_todo.py`

S05 verification:

```sh
python -m pytest tests/test_todo.py
python -m pytest tests
```

## S06 update: read-only subagents

S06 adds a `task(description, role)` tool so the parent agent can delegate local analysis to a child agent with fresh context.

New in s06:

- `task` launches a synchronous subagent and returns only the final structured summary.
- Supported roles are:
  - `issue_analyzer`
  - `repo_mapper`
  - `test_analyzer`
  - `doc_reviewer`
- The subagent starts with a fresh `messages` list, so intermediate tool chatter does not enter the parent context.
- The subagent cannot call `task`, `write_file`, or `edit_file`.
- The subagent tool set is limited to:
  - `bash`
  - `read_file`
  - `glob`
  - `git_status`
  - `inspect_repo`
- Subagent bash commands are restricted to read-only inspection prefixes such as `git status`, `git diff`, `git log`, `rg`, `grep`, `dir`, `ls`, `type`, and `pwd`.
- Trace now records `subagent_start`, `subagent_tool_use`, and `subagent_stop` events with `agent=subagent`.

Suggested reading order for S06:

1. `learn-claude-code/coding_plan.md` section `s06`
2. `learn-claude-code/s06_subagent/README.md`
3. `osc_agent/tools/task.py`
4. `osc_agent/agent_loop.py`
5. `osc_agent/harness/trace.py`
6. `osc_agent/tools/repo.py`
7. `tests/test_subagent.py`

S06 verification:

```sh
python -m pytest tests/test_subagent.py
python -m pytest tests
```

## S07 update: on-demand skill loading

S07 adds a two-level skill loading system: the system prompt includes only a compact skill catalog, and the model can call `load_skill(name)` to fetch full instructions when needed.

New in s07:

- `osc_agent/skills/` contains five built-in skills:
  - `python`
  - `javascript`
  - `docs`
  - `tests`
  - `open-source`
- Each skill is a directory with `SKILL.md` frontmatter containing `name` and `description`.
- `osc_agent/skills/registry.py` scans skills and implements `load_skill(name)`.
- `osc_agent/harness/prompt.py` assembles the runtime system prompt with:
  - repo path
  - contribution planning instruction
  - skill catalog
  - suggested relevant skills
- The prompt does not include full skill bodies.
- `load_skill(name)` returns the full `SKILL.md` content through a tool result.
- Adding a new skill directory does not require changing the agent loop.
- `inspect_repo()` now reports suggested skills based on repository markers.

Suggested reading order for S07:

1. `learn-claude-code/coding_plan.md` section `s07`
2. `learn-claude-code/s07_skill_loading/README.md`
3. `osc_agent/skills/registry.py`
4. `osc_agent/skills/*/SKILL.md`
5. `osc_agent/harness/prompt.py`
6. `osc_agent/agent_loop.py`
7. `osc_agent/tools/repo.py`
8. `tests/test_skills.py`

S07 verification:

```sh
python -m pytest tests/test_skills.py
python -m pytest tests
```

## S08 update: context compaction

S08 adds a lightweight context compaction pipeline so long-running sessions do not keep every old message and large tool result in the active prompt.

New in s08:

- `osc_agent/harness/compact.py` implements:
  - `estimate_size(messages)`
  - `tool_result_budget(messages, repo_root)`
  - `snip_compact(messages)`
  - `micro_compact(messages)`
  - `compact_history(messages, repo_root)`
  - `reactive_compact(messages, repo_root)`
- Every agent loop iteration now runs the cheap compaction pipeline before calling the model:
  - persist large tool results
  - snip old middle messages
  - replace old tool results with previews
  - compact history when estimated size remains too large
- Large tool outputs are persisted under `.osc_agent/tool-results/`.
- Full transcripts are written under `.osc_agent/transcripts/` before history compaction.
- A `compact(reason)` tool lets the model request manual history compaction.
- If the model API raises a prompt-too-long style error, the loop runs `reactive_compact` and retries once.

Suggested reading order for S08:

1. `learn-claude-code/coding_plan.md` section `s08`
2. `learn-claude-code/s08_context_compact/README.md`
3. `osc_agent/harness/compact.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_compact.py`

S08 verification:

```sh
python -m pytest tests/test_compact.py
python -m pytest tests
```

## S09 update: contribution memory

S09 adds a small persistent memory layer so stable project facts can survive context compaction and new sessions.

New in s09:

- `osc_agent/harness/memory.py` manages `.osc_agent/memory/`.
- `ensure_memory_store(repo_root)` creates `.osc_agent/memory/MEMORY.md`.
- Memory files are readable Markdown with YAML-style frontmatter:
  - `name`
  - `description`
  - `type`
- Supported memory types are `user`, `feedback`, `project`, and `reference`.
- `extract_repo_memories(repo_root)` records reusable project facts such as:
  - Python test command hints from `pyproject.toml`
  - JavaScript workflow hints from `package.json`
  - contribution guide references from `CONTRIBUTING*`
  - PR template references from `.github/pull_request_template*`
- `memory_prompt(repo_root, query, limit_chars)` injects a bounded memory index plus relevant details.
- Sensitive terms such as secrets, tokens, passwords, API keys, and private absolute user paths are rejected.
- `assemble_system_prompt()` now includes bounded persistent memory content.

Suggested reading order for S09:

1. `learn-claude-code/coding_plan.md` section `s09`
2. `learn-claude-code/s09_memory/README.md`
3. `osc_agent/harness/memory.py`
4. `osc_agent/harness/prompt.py`
5. `tests/test_memory.py`

S09 verification:

```sh
python -m pytest tests/test_memory.py
python -m pytest tests
```

## Project layout

```text
osc_agent/
  cli.py              # CLI entry point
  agent_loop.py       # Anthropic-style loop and handler-map dispatch
  config.py           # .env and Anthropic client setup
  harness/
    hooks.py          # UserPromptSubmit/PreToolUse/PostToolUse/Stop hooks
    permissions.py    # path, shell, and write permission decisions
    trace.py          # JSONL audit trace writer
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
python -m pytest tests/test_permissions.py
python -m pytest tests/test_permissions.py tests/test_file_tools.py
python -m pytest tests
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py
```

On Windows, if pytest temporary directories are locked by the shell, use a project-local temp directory:

```sh
python -m pytest tests --basetemp .pytest-local -p no:cacheprovider
```
