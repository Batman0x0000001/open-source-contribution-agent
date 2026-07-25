---
name: open-source-contribution
version: 1
description: Analyze, design, implement, verify, and draft an open-source contribution through one adaptive agent loop.
when_to_use: Use when a user wants to find and complete a contribution to an open-source repository.
allowed_tools:
  - read_file
  - glob
  - powershell
  - git_status
  - git_diff
  - git_log
  - github_list_issues
  - github_get_issue
  - ask_user_question
  - enter_plan_mode
  - write_plan
  - read_plan
  - exit_plan_mode
  - read_skill_resource
  - enter_worktree
  - exit_worktree
  - write_file
  - edit_file
  - read_tool_result
context: inline
user_invocable: true
disable_model_invocation: false
resources:
  - discover.md
  - design.md
  - implement.md
  - pr-draft.md
input_schema:
  type: object
  properties:
    repo_url: {type: string}
    goal: {type: [string, 'null']}
  required: [repo_url]
  additionalProperties: false
output_schema:
  type: object
  properties: {}
  additionalProperties: true
---
You are running the open-source contribution method for the supplied repository.

Progress dynamically from evidence; do not create or maintain a fixed phase-state object. The Session transcript, approved plan, repository files, tests, and git diff are the sources of truth.

1. Read `discover.md` with `read_skill_resource`, inspect the repository, and ask the user to choose among evidence-backed candidates.
2. Read `design.md`, enter Plan Mode, write the plan, and request approval through ExitPlanMode.
3. After approval, enter a Git worktree, read `implement.md`, implement and verify. Return failures to the agent loop and adapt from evidence.
4. After verification, read `pr-draft.md` and create a local PR draft. Commit, push, and remote PR creation require separate permission.

Never claim success without command output or repository evidence. Never discard a dirty worktree without explicit permission.
