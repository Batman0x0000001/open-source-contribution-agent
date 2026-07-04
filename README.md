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

S06 adds a `subagent(description, role)` tool so the parent agent can delegate local analysis to a child agent with fresh context.

New in s06:

- `subagent` launches a synchronous child agent and returns only the final structured summary.
- Supported roles are:
  - `issue_analyzer`
  - `repo_mapper`
  - `test_analyzer`
  - `doc_reviewer`
- The subagent starts with a fresh `messages` list, so intermediate tool chatter does not enter the parent context.
- The subagent cannot call `subagent`, `write_file`, or `edit_file`.
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
3. `osc_agent/harness/subagent.py`
4. `osc_agent/agent_loop.py`
5. `osc_agent/harness/trace.py`
6. `osc_agent/tools/repo.py`
7. `tests/test_subagent.py`

S06 verification:

```sh
python -m pytest tests/test_subagent.py
python -m pytest tests
```

## Naming clarification: subagent, tasks, and todo

The harness now uses distinct names for three different planning layers:

- `subagent`: a tool-backed harness mechanism for delegating isolated read-only analysis to a child agent.
- `tasks`: persistent contribution tasks stored as JSON under `.osc_agent/tasks/`.
- `todo`: an in-process checklist for the current execution step.

This keeps S06 subagent delegation separate from S12 persistent task graph management.

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

## S10 update: runtime system prompt assembly

S10 turns the system prompt into runtime sections built from real harness state instead of one hard-coded string.

New in s10:

- `osc_agent/harness/prompt.py` now defines:
  - `PromptContext`
  - `PROMPT_SECTIONS`
  - `update_context(...)`
  - `assemble_system_prompt(context)`
  - `get_system_prompt(context)`
- Prompt sections include:
  - identity
  - repo
  - task
  - tools
  - permissions
  - skills
  - memory
  - current todos
  - git state
- `update_context()` gathers real state from:
  - target repo path
  - current messages
  - enabled tools
  - repo inspection
  - permission summary
  - skill catalog
  - persistent memory
  - current todos
  - git status
- `get_system_prompt()` caches assembled prompt text using a stable JSON context key.
- `agent_loop()` now rebuilds prompt context before each model call.
- The prompt explicitly says the goal is a reviewable contribution and that the agent must not automatically `git push` or open PRs.

Suggested reading order for S10:

1. `learn-claude-code/coding_plan.md` section `s10`
2. `learn-claude-code/s10_system_prompt/README.md`
3. `osc_agent/harness/prompt.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_prompt.py`

S10 verification:

```sh
python -m pytest tests/test_prompt.py
python -m pytest tests
```

## S11 update: error recovery

S11 adds recovery paths for common model and command failures so the CLI does not stop at the first transient error.

New in s11:

- `osc_agent/harness/recovery.py` defines:
  - `RecoveryState`
  - `with_retry(...)`
  - `retry_delay(...)`
  - `is_prompt_too_long_error(...)`
  - `is_rate_limit_error(...)`
  - `is_overloaded_error(...)`
- LLM calls now retry 429 and 529 style transient errors with exponential backoff.
- Repeated 529 overloaded errors can switch to `FALLBACK_MODEL_ID` when configured.
- Prompt-too-long errors trigger `reactive_compact` and retry once.
- `max_tokens` responses first retry with `64000` max tokens.
- If output is still truncated after escalation, the loop appends a continuation prompt.
- Shell timeout, OS, and non-zero exit errors now return structured `Error: {...}` text.
- Failed test commands include recovery guidance asking the agent to read the failure, locate files, update todos, and rerun focused tests.
- Recovery events are written to trace.

Suggested reading order for S11:

1. `learn-claude-code/coding_plan.md` section `s11`
2. `learn-claude-code/s11_error_recovery/README.md`
3. `osc_agent/harness/recovery.py`
4. `osc_agent/agent_loop.py`
5. `osc_agent/tools/shell.py`
6. `tests/test_recovery.py`

S11 verification:

```sh
python -m pytest tests/test_recovery.py tests/test_shell.py
python -m pytest tests
```

## S12 update: persistent contribution task graph

S12 adds a persistent task graph for contribution work that needs dependencies, ownership, and recovery across CLI restarts.

New in s12:

- `osc_agent/harness/tasks.py` defines `ContributionTask`.
- Task files are stored as inspectable JSON under `.osc_agent/tasks/`.
- Task fields include:
  - `id`
  - `subject`
  - `description`
  - `status`
  - `owner`
  - `blockedBy`
  - `files`
  - `evidence`
  - `worktree`
- New task tools:
  - `create_task`
  - `list_tasks`
  - `get_task`
  - `claim_task`
  - `complete_task`
- `claim_task` refuses blocked tasks until every `blockedBy` task is completed.
- `complete_task` reports newly unblocked downstream tasks.
- `create_default_task_graph()` creates the standard contribution flow:
  - repo scan
  - plan
  - edit
  - test
  - summarize
  - draft PR

Suggested reading order for S12:

1. `learn-claude-code/coding_plan.md` section `s12`
2. `learn-claude-code/s12_task_system/README.md`
3. `osc_agent/harness/tasks.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_tasks.py`

S12 verification:

```sh
python -m pytest tests/test_tasks.py
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

## S13 Background Tasks

S13 adds background execution for slow bash commands. The bash tool now accepts `run_in_background`; when it is true, the main loop starts a daemon background task, immediately returns a placeholder `tool_result`, and later injects a `<task_notification>` message after the task finishes. Output is written under `.osc_agent/background/`.

The implementation deliberately does not auto-background every slow-looking command. `pytest`, `npm test`, `cargo test`, and build commands are recognized as slow candidates, but background execution still requires an explicit `run_in_background=true` request from the model.

S13 reading order:

1. `learn-claude-code/coding_plan.md` section `s13`
2. `learn-claude-code/s13_background_tasks/README.md`
3. `osc_agent/tools/shell.py`
4. `osc_agent/harness/background.py`
5. `osc_agent/agent_loop.py`
6. `tests/test_background.py`
7. `tests/test_agent_loop.py`

S13 operation steps:

1. Add `run_in_background` to the bash tool schema.
2. Add background task lifecycle state in `osc_agent/harness/background.py`.
3. Write command output to `.osc_agent/background/<task_id>.out`.
4. Return a placeholder while the task is running.
5. Inject completed background notifications at the start of later loop rounds.
6. Keep completion delivery automatic through `collect_background_results()`, without exposing a separate background-query tool to the model.

S13 verification:

```sh
python -m pytest tests/test_background.py tests/test_agent_loop.py tests/test_shell.py --basetemp .pytest-s13-focused -p no:cacheprovider
python -m pytest tests --basetemp .pytest-s13-all -p no:cacheprovider
```

## S14 Cron Scheduler

S14 adds a minimal persisted scheduler for recurring contribution checks. The agent exposes `schedule_check(cron, prompt, enabled)`, `list_schedules()`, and `cancel_schedule(schedule_id)`. Schedules are written to `.osc_agent/scheduled_tasks.json`; each agent-loop round checks enabled schedules, injects due reminders as `<task_notification>` blocks, and records the last fired minute so the same schedule does not fire twice in one minute.

This v1 follows `coding_plan.md` naming (`schedule_check`, `list_schedules`, `cancel_schedule`) while using the S14 tutorial's cron semantics. It supports five-field cron expressions with `*`, `*/N`, `N`, `N-M`, `N-M/S`, and comma-separated values. Day-of-month and day-of-week use OR semantics when both are constrained.

S14 reading order:

1. `learn-claude-code/coding_plan.md` section `s14`
2. `learn-claude-code/s14_cron_scheduler/README.md`
3. `osc_agent/harness/cron.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_cron.py`

S14 operation steps:

1. Define persisted cron schedule records with `id`, `cron`, `prompt`, `enabled`, `created_at`, and `last_fired_at`.
2. Validate five-field cron expressions before saving schedules.
3. Register `schedule_check`, `list_schedules`, and `cancel_schedule` in the agent tool pool.
4. At the start of each agent loop round, collect due cron notifications and inject them with background-task notifications.
5. Keep canceled schedules disabled for auditability instead of deleting them.

S14 verification:

```sh
python -m pytest tests/test_cron.py tests/test_agent_loop.py --basetemp .pytest-s14-focused -p no:cacheprovider
python -m pytest tests --basetemp .pytest-s14-all -p no:cacheprovider
```

## S15 Agent Teams

S15 adds a lightweight team harness for longer-lived teammate agents. The core is a file-backed `MessageBus`: each agent has a JSONL inbox under `.osc_agent/mailboxes/`, sending appends one message, and reading consumes the inbox. The Lead agent gets three tools: `spawn_teammate(name, role, prompt, allow_write)`, `send_message(to_agent, content, message_type, metadata)`, and `check_inbox()`.

Initial teammate roles are `reviewer`, `tester`, and `doc_writer`. Teammates run in daemon threads with their own messages and limited tools. By default they can inspect the repo and send messages, but they do not receive `write_file`; `allow_write=true` must be explicit for documentation or edit assignments.

S15 reading order:

1. `learn-claude-code/coding_plan.md` section `s15`
2. `learn-claude-code/s15_agent_teams/README.md`
3. `osc_agent/harness/teams.py`
4. `osc_agent/agent_loop.py`
5. `tests/test_teams.py`

S15 operation steps:

1. Create file-backed mailboxes in `.osc_agent/mailboxes/*.jsonl`.
2. Implement `MessageBus.send(...)` and consuming `read_inbox(agent)`.
3. Add `spawn_teammate(...)` to start a teammate thread with its own context.
4. Add `send_message(...)` and `check_inbox()` for Lead/team communication.
5. Automatically inject Lead inbox messages at the start of each agent-loop round.
6. Keep teammate tools limited, with file writes disabled unless `allow_write=true`.

S15 verification:

```sh
python -m pytest tests/test_teams.py tests/test_agent_loop.py --basetemp .pytest-s15-focused -p no:cacheprovider
python -m pytest tests --basetemp .pytest-s15-all -p no:cacheprovider
```
