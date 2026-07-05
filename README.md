# Open Source Contribution Agent Harness

This project is a staged Python CLI implementation of an open source contribution agent harness.


## OpenSourcePR 1-4 Workflow

This upgrade turns the four OpenSourcePR documents into a resumable contribution workflow:

1. `discover`: find contribution entry points from repository structure, issues, and architecture dimensions.
2. `design`: turn a selected direction into a scoped technical plan.
3. `implement`: prepare the implementation prompt, task graph, todo state, and run the normal agent loop.
4. `draft-pr`: generate a full PR draft with Problem, Solution, Changes, Testing, and Notes for Reviewer.

The workflow stores artifacts under `.osc_agent/contribution_runs/<run_id>/`:

```text
run.json
01_discover.md
01_discover.json
02_design.md
02_design.json
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

The GitHub issue reader is read-only. It can fetch open issues with the standard library and `GITHUB_TOKEN` when present, but it never comments, assigns, pushes, or opens pull requests. If network access fails, use `--issues-file` with a JSON object containing `issues` and optional `comments_by_issue`.

OpenSourcePR workflow operation steps:

1. Parse or fetch candidate issues with read-only GitHub APIs.
2. Build a depth=3 repository tree, detect entry points, and locate functions/classes for the seven architecture dimensions.
3. Persist `discover` artifacts and Top 3 contribution directions.
4. Resume by `run_id`, select a direction, and persist the technical design artifact.
5. Before implementation, check local git status and ask before continuing with existing local changes.
6. Use the existing agent loop for implementation, then save the implementation report.
7. Generate a workflow-aware PR draft from artifacts plus current diff.

Verification:

```sh
python -m pytest tests/test_github_tools.py tests/test_repo_analysis.py tests/test_contribution_workflow.py tests/test_contribute_cli.py tests/test_cli.py tests/test_pr_draft.py --basetemp .pytest-opensourcepr-focused -p no:cacheprovider
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py osc_agent/tools/github.py osc_agent/tools/repo.py osc_agent/tools/pr.py osc_agent/harness/contribution_workflow.py
python -m pytest tests --basetemp .pytest-opensourcepr-all -p no:cacheprovider
```

## OpenSourcePR Workflow Quality Upgrade

This follow-up makes the four-step workflow closer to how a careful open source contributor works:

1. `discover` now writes an `01_discover_agent_prompt.md` evidence prompt, so an optional agent review can deepen the issue and architecture analysis instead of relying only on keyword heuristics.
2. `design` now writes an `02_design_agent_prompt.md` prompt and can attach an agent-generated concrete design back into `02_design.md/json`.
3. `implement` now prepares todo state, persistent tasks, and the implementation prompt before calling the agent loop. The implementation report is updated after the agent finishes, so the workflow order matches the third OpenSourcePR document.
4. `draft-pr` now extracts Testing evidence from the implementation report and includes reviewer notes tied to the saved workflow artifacts.

Use deeper agent review when you want the workflow to behave more like a real contribution assistant:

```sh
osc-agent contribute discover --repo <local_path> --repo-url <github_url> --agent-review
osc-agent contribute design --repo <local_path> --run-id <id> --direction "<direction>" --agent-review
osc-agent contribute run --repo <local_path> --repo-url <github_url> --agent-review
```

Quality-upgrade reading order:

1. `osc_agent/harness/contribution_workflow.py`
2. `osc_agent/cli.py`
3. `osc_agent/tools/pr.py`
4. `tests/test_contribution_workflow.py`
5. `tests/test_pr_draft.py`

Quality-upgrade verification:

```sh
python -m pytest tests/test_contribution_workflow.py tests/test_contribute_cli.py tests/test_pr_draft.py tests/test_cli.py --basetemp .pytest-opensourcepr-upgrade-focused -p no:cacheprovider
python -m py_compile osc_agent/cli.py osc_agent/agent_loop.py osc_agent/tools/github.py osc_agent/tools/repo.py osc_agent/tools/pr.py osc_agent/harness/contribution_workflow.py
python -m pytest tests --basetemp .pytest-opensourcepr-upgrade-all -p no:cacheprovider
```
