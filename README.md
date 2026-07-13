# Open Source Contribution Agent Harness

This project is a staged Python CLI implementation of an open source contribution agent harness.


## OpenSourcePR 1-4 Workflow

This upgrade turns the four OpenSourcePR documents into a resumable contribution workflow:

1. `discover`: find contribution entry points from repository structure, issues, and architecture dimensions.
2. `design`: turn a selected direction into a scoped technical plan.
3. `implement`: create an isolated worktree, prepare todo/task state, and run controlled understanding, editing, and verification substeps.
4. `draft-pr`: generate a full PR draft with Problem, Solution, Changes, Testing, and Notes for Reviewer.

The workflow stores artifacts under `.osc_agent/contribution_runs/<run_id>/`:

```text
run.json
01_discover.md
01_discover.json
01_discover_agent_prompt.md
02_design.md
02_design.json
02_design_agent_prompt.md
03_implementation_report.md
04_pr_draft.md
```

New CLI commands:

```sh
osc-agent contribute discover --repo <local_path> --repo-url <github_url>
osc-agent contribute discover --repo <local_path> --repo-url <github_url> --issues-file issues.json
osc-agent contribute design --repo <local_path> --run-id <id> --direction "<direction>"
osc-agent contribute implement --repo <local_path> --run-id <id>
osc-agent contribute draft-pr --repo <local_path> --run-id <id>
osc-agent contribute run --repo <local_path> --repo-url <github_url>
```

LLM analysis is enabled by default and requires `ANTHROPIC_API_KEY` in the environment or `.env`.
Use `--no-llm` on `discover`, `design`, `draft-pr`, or `run` to select deterministic local generation for those
analysis stages. The `implement` stage always runs the Anthropic-backed agent loop, so `contribute implement` and a
complete `contribute run --no-llm` still require valid Anthropic configuration. `--no-llm` is therefore not a fully
offline end-to-end mode.

The GitHub issue reader is read-only. It can fetch open issues with the standard library and `GITHUB_TOKEN` when present, but it never comments, assigns, pushes, or opens pull requests. If network access fails, use `--issues-file` with a JSON object containing `issues` and optional `comments_by_issue`.

OpenSourcePR workflow operation steps:

1. Parse or fetch candidate issues with read-only GitHub APIs.
2. Build a depth=3 repository tree, detect entry points, and locate functions/classes for the seven architecture dimensions.
3. Persist `discover` artifacts and Top 3 contribution directions.
4. Resume by `run_id`, select a direction, and persist the technical design artifact.
5. Validate discover and design artifacts when using the end-to-end `run` command.
6. Before implementation, inspect local git status, ask before continuing with existing changes, and create an isolated worktree for the run.
7. Prepare todo state, a persistent task graph, and the initial implementation report.
8. Run the implementation controller in three ordered substeps:
   - `understanding`: inspect the approved scope without editing; execution continues only when the agent returns `READY_TO_EDIT`.
   - `edit`: make the scoped code changes using the confirmed understanding and saved design.
   - `verification`: run focused checks and report exact commands and results without committing, pushing, or opening a PR.
9. Save all three outputs in `03_implementation_report.md` and apply the implementation quality gate.
10. Generate a workflow-aware PR draft from the saved artifacts and current diff.

The end-to-end control flow is:

```text
discover -> discover gate -> choose direction -> design -> design gate
         -> confirm implementation -> isolated worktree
         -> understanding -> READY_TO_EDIT checkpoint -> edit -> verification
         -> implementation gate -> draft-pr
```

Standalone `discover`, `design`, and `implement` commands generate their stage output but do not enforce every gate in
the same way as `contribute run`. Use `contribute run` when you want the complete gated workflow.

Verification:

```sh
python -m pytest tests/test_github_tools.py tests/test_repo_analysis.py tests/test_contribution_workflow.py tests/test_contribute_cli.py tests/test_cli.py tests/test_pr_draft.py --basetemp .pytest-opensourcepr-focused
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py osc_agent/tools/github.py osc_agent/tools/repo.py osc_agent/tools/pr.py osc_agent/harness/contribution_workflow.py
python -m pytest tests --basetemp .pytest-opensourcepr-all
```

## OpenSourcePR Workflow Quality Upgrade

This follow-up makes the four-step workflow closer to how a careful open source contributor works:

1. `discover` writes an `01_discover_agent_prompt.md` evidence prompt and uses LLM analysis by default, with an explicit `--no-llm` fallback.
2. `design` writes an `02_design_agent_prompt.md` prompt and uses the selected direction to generate a focused technical design.
3. `implement` prepares todo and persistent task state, then enforces the `understanding -> edit -> verification` sequence through the workflow layer. Editing cannot begin until the understanding output contains `READY_TO_EDIT`.
4. `draft-pr` now extracts Testing evidence from the implementation report and includes reviewer notes tied to the saved workflow artifacts.

Use LLM analysis when you want the workflow to perform deeper issue and design review:

```sh
osc-agent contribute discover --repo <local_path> --repo-url <github_url> --llm
osc-agent contribute design --repo <local_path> --run-id <id> --direction "<direction>" --llm
osc-agent contribute run --repo <local_path> --repo-url <github_url> --llm
```

Quality-upgrade reading order:

1. `osc_agent/harness/contribution_workflow.py`
2. `osc_agent/cli.py`
3. `osc_agent/tools/pr.py`
4. `tests/test_contribution_workflow.py`
5. `tests/test_pr_draft.py`

Quality-upgrade verification:

```sh
python -m pytest tests/test_contribution_workflow.py tests/test_contribute_cli.py tests/test_pr_draft.py tests/test_cli.py --basetemp .pytest-opensourcepr-upgrade-focused
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py osc_agent/tools/github.py osc_agent/tools/repo.py osc_agent/tools/pr.py osc_agent/harness/contribution_workflow.py
python -m pytest tests --basetemp .pytest-opensourcepr-upgrade-all
```
