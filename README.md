# Agent/LLM Python Open Source Contribution Agent

A Python 3.10+ CLI harness that constrains an Anthropic-compatible coding agent into an auditable contribution workflow for Agent and LLM Python repositories.

The project focuses on local analysis, scoped implementation, deterministic validation, and local PR-draft generation. GitHub access remains read-only: the CLI never pushes, comments, assigns issues, or creates a remote pull request.

## Installation

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
osc-agent --help
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. `GITHUB_TOKEN` is optional and is only used for read-only Issue, Timeline, linked-PR, and recent-commit queries.

## Contribution workflow

```text
clean repository + base commit
  -> discover -> discover gate
  -> choose direction -> design -> design gate
  -> isolated worktree
  -> understanding -> READY_TO_EDIT
  -> edit -> deterministic scope validation
  -> execute verification commands
  -> implementation gate
  -> local PR draft
```

The four stages are:

1. `discover`: verify that the target is an Agent/LLM Python repository, collect Issue and Python AST evidence, detect linked open PRs, and rank candidates with explainable `HIGH`, `MEDIUM`, `LOW`, or `REJECT` levels.
2. `design`: persist the approved files, allowed new directories, forbidden paths, symbols, source hashes, acceptance checks, assumptions, impact area, and change budgets as structured JSON.
3. `implement`: create a worktree at the saved base commit, enforce the understanding/edit/verification sequence, compare the real diff with the design, and execute verification commands while recording exit codes.
4. `draft-pr`: generate a local Markdown draft only after scope, repository consistency, and verification gates succeed.

Common commands:

```sh
osc-agent contribute discover --repo <local_path> --repo-url <github_url>
osc-agent contribute discover --repo <local_path> --repo-url <github_url> --issues-file issues.json
osc-agent contribute design --repo <local_path> --run-id <id> --direction "<direction>"
osc-agent contribute update-design --repo <local_path> --run-id <id> --allow-file osc_agent/example.py --test-command "python -m pytest tests/test_example.py"
osc-agent contribute implement --repo <local_path> --run-id <id>
osc-agent contribute draft-pr --repo <local_path> --run-id <id>
osc-agent contribute run --repo <local_path> --repo-url <github_url>
```

The source repository must have at least one commit and no uncommitted project changes. Runtime files under `.osc_agent/` are ignored by this cleanliness check. Existing runs use Schema v2; older run files are intentionally rejected instead of migrated.

`--no-llm` selects deterministic generation for discovery, design, and PR drafting. Implementation still uses the Anthropic-compatible agent loop, so it is not a fully offline end-to-end mode.

## Runtime budgets and gates

Default limits are:

| Limit | Default | Environment variable | CLI override |
|---|---:|---|---|
| Model rounds | 30 | `OSC_AGENT_MAX_ROUNDS` | `--max-rounds` |
| Input + output tokens | 200,000 | `OSC_AGENT_MAX_TOKENS` | `--max-tokens` |
| Agent deadline | 1,800 seconds | `OSC_AGENT_DEADLINE_SECONDS` | `--deadline-seconds` |
| Repeated action threshold | 3 | `OSC_AGENT_REPEAT_ACTION_LIMIT` | — |
| Consecutive tool failures | 3 | `OSC_AGENT_FAILURE_LIMIT` | — |
| No-progress threshold | 6 | `OSC_AGENT_NO_PROGRESS_LIMIT` | — |
| Changed files | 5 | `OSC_AGENT_MAX_CHANGED_FILES` | `--max-files` |
| Added + deleted lines | 400 | `OSC_AGENT_MAX_DIFF_LINES` | `--max-diff-lines` |

Agent runs end with one of these deterministic statuses:

```text
SUCCESS
FAILED_VALIDATION
FAILED_BUDGET
FAILED_TOOL
BLOCKED_NEEDS_USER
OUT_OF_SCOPE
STALE_RUN
```

The implementation gate blocks PR drafting when the base commit or saved artifacts drift, files fall outside the approved scope, forbidden paths change, change budgets are exceeded, tests fail, or no verification command was executed. A task with no applicable automated test may continue only with an audited waiver:

```sh
osc-agent contribute implement \
  --repo <local_path> \
  --run-id <id> \
  --test-waiver-reason "Documentation-only change; links checked manually"
```

Failures preserve the worktree, diff, tool trace, and diagnostic artifacts for inspection; the CLI does not automatically roll them back.

## Artifacts and recovery

Each run is stored under `.osc_agent/contribution_runs/<run_id>/`:

```text
run.json
01_discover.json
01_discover.md
01_discover_agent_prompt.md
02_design.json
02_design.md
02_design_agent_prompt.md
03_implementation.json
03_implementation_report.md
04_pr_draft.md
metrics.json
metrics.md
```

JSON is the source of truth and Markdown is rendered from it. State writes use a temporary file plus atomic replacement. The run state records the Schema version, base commit, configuration snapshot, Issue snapshot time, stage states, input/output hashes, critical file hashes, metrics, and final status. Resume checks reject changed commits, evidence files, or stage artifacts as `STALE_RUN`.

An additional `.osc_agent/runtime_state.json` keeps the current goal, user constraints, allowed and forbidden scope, verified facts, plan, modified files, test results, failed strategies, and unresolved questions outside the compactable message history. This state is injected into every model round.

## Tool and Skill contracts

Every model-visible built-in tool result is serialized with the same protocol: success state, typed error code, retryability, side-effect marker, summary, artifact reference, metadata, call ID, action fingerprint, and latency. The loop uses these fields for failure counting and progress detection rather than parsing free-form error prefixes.

Built-in Markdown Skills use validated YAML Frontmatter with a Schema version, Skill version, applicability, required tools, permissions, and input/output contracts. Skill permissions can only reduce available capabilities; they never override global path, command, workflow, or human-approval checks.

Multi-Agent support is limited to optional parallel read-only analysis. Multiple agents are not allowed to edit the same worktree concurrently.

## Metrics and verification

Each run reports stage duration, model calls, input/output tokens, retries, tool calls and failures, changed files, diff lines, verification commands and exit codes, human confirmations, and the final status.

Run the offline suite and compilation checks with:

```sh
python -m pytest tests
python -m py_compile osc_agent/*.py osc_agent/harness/*.py osc_agent/tools/*.py osc_agent/skills/*.py
```

Two real-model contract checks are excluded by default. Run them manually when API usage is intended:

```sh
set OSC_AGENT_RUN_LIVE_TESTS=1
python -m pytest tests/test_live_model.py -m live_model
```

The automated suite validates software behavior; it is not a benchmark of contribution quality. Real-repository benchmark results, remote PR creation, OS-level sandboxing, general multi-provider support, and effectiveness claims remain explicitly out of scope for this version.
