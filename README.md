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
  -> reuse an existing failure, or create a test-only regression reproduction
  -> freeze generated regression test hashes
  -> edit -> deterministic scope validation
  -> rerun requirement-linked verification commands
  -> on test failure: repair from logs + current diff -> rerun verification (bounded)
  -> implementation gate
  -> local PR draft
```

The four stages are:

1. `discover`: verify that the target is an Agent/LLM Python repository, collect Issue and Python AST evidence, detect linked open PRs, and rank candidates with explainable `HIGH`, `MEDIUM`, `LOW`, or `REJECT` levels.
2. `design`: persist Issue-derived requirements, approved files, symbol-level source evidence, pre-change failure checks, requirement-linked acceptance checks, assumptions, impact area, and change budgets as structured JSON.
3. `implement`: create a worktree at the saved base commit, reproduce the expected failure before production editing, enforce the understanding/reproduce/edit/verification sequence, freeze generated regression tests, repair failed edits from controlled logs and the current diff, compare the real diff with the design, and prove that every requirement's verification command passes after the change.
4. `draft-pr`: generate a local Markdown draft only after scope, repository consistency, and verification gates succeed.

Common commands:

```sh
osc-agent contribute discover --repo <local_path> --repo-url <github_url>
osc-agent contribute discover --repo <local_path> --repo-url <github_url> --issues-file issues.json
osc-agent contribute design --repo <local_path> --run-id <id> --direction "<direction>"
osc-agent contribute update-design --repo <local_path> --run-id <id> --allow-file osc_agent/example.py --test-command "python -m pytest tests/test_example.py"
osc-agent contribute update-design --repo <local_path> --run-id <id> --task-type behavior --baseline-command "python -m pytest tests/test_example.py" --baseline-output "AssertionError"
osc-agent contribute update-design --repo <local_path> --run-id <id> --task-type behavior --test-command "python -m pytest tests/test_example.py" --allow-new-dir tests --reproduction-test-file tests/test_example.py
osc-agent contribute implement --repo <local_path> --run-id <id>
osc-agent contribute draft-pr --repo <local_path> --run-id <id>
osc-agent contribute run --repo <local_path> --repo-url <github_url>
```

The source repository must have at least one commit and no uncommitted project changes. Runtime files under `.osc_agent/` are ignored by this cleanliness check. Existing runs use Schema v3; older run files are intentionally rejected instead of migrated.

The contribution workflow always uses the configured LLM for discovery, design, implementation, and PR drafting. Set `ANTHROPIC_API_KEY` in `.env` before running it.

## Runtime budgets and gates

Default limits are:

| Limit | Default | Environment variable | CLI override |
|---|---:|---|---|
| Model rounds | 30 | `OSC_AGENT_MAX_ROUNDS` | `--max-rounds` |
| Input + output tokens | 200,000 | `OSC_AGENT_MAX_TOKENS` | `--max-tokens` |
| Agent deadline | 1,800 seconds | `OSC_AGENT_DEADLINE_SECONDS` | `--deadline-seconds` |
| Repeated action threshold | 3 | `OSC_AGENT_REPEAT_ACTION_LIMIT` | — |
| Consecutive tool failures / verification repairs | 3 | `OSC_AGENT_FAILURE_LIMIT` | — |
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

For behavior changes, the design must identify symbol-level evidence and choose one reproduction mode. `existing` runs a known narrow command and matches its non-zero exit code plus stable failure fragment before editing. `generated_test` restricts the reproduction agent to declared Python test files, accepts only a pytest assertion-failure exit, records the failure log, and freezes the test hashes before production editing. The generated test must contain a supported assertion, call at least one approved target symbol, and appear in the controlled pytest failure output; these semantic-binding facts are stored with the Issue requirement IDs. Any later change or deletion of those tests fails validation. After editing, the exact reproduction command is rerun and mapped back to Issue-derived requirement IDs; every requirement must pass before PR drafting.

The implementation gate also blocks PR drafting when the base commit or saved artifacts drift, files fall outside the approved scope, forbidden paths change, change budgets are exceeded, tests fail, or requirement coverage is incomplete. Only documentation and configuration tasks may continue without an automated command, using an audited waiver:

When post-edit verification fails with a normal non-zero exit code, the workflow marks the edit checkpoint as `NEEDS_REPAIR`, gives the repair agent the exact controlled results, captured logs, and current diff, then reruns the same verification commands. Frozen generated tests remain immutable. Repair history is persisted, and the run stops with `FAILED_VALIDATION` after the configured consecutive-failure limit; resume cannot silently reset that limit.

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

Generated-test reproduction evidence is stored inside `03_implementation.json`, including the controlled failing command, execution-log path, declared test files, frozen SHA-256 hashes, and checkpoint state. A resumed run reuses this evidence only when every frozen test still matches.

Additional `.osc_agent/sessions/<session_id>/runtime_state.json` files keep each session's goal, user constraints, allowed and forbidden scope, verified facts, plan, modified files, test results, failed strategies, and unresolved questions outside the compactable message history. Contribution sessions bind this state to an explicit run ID instead of guessing the latest run by file time.

## Tool and Skill contracts

Every model-visible built-in tool result is serialized with the same protocol: success state, typed error code, retryability, side-effect marker, summary, artifact reference, metadata, call ID, action fingerprint, and latency. The loop uses these fields for failure counting and progress detection rather than parsing free-form error prefixes.

Contribution stages use a hard capability policy in addition to prompts. Understanding is read-only; reproduction can write only declared regression-test files; edit and repair can write only approved production files and cannot change frozen tests. Shell verification, worktree lifecycle, and PR operations remain deterministic orchestrator responsibilities.

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
