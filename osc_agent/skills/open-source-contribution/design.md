# Design

## Input and prerequisites

Use the selected direction from the transcript and re-read every affected file. Enter Plan Mode before producing the implementation plan.

## Execution

1. Define the problem, non-goals, compatibility constraints, and measurable success criteria.
2. Compare only materially different viable designs; prefer the smallest design consistent with the host project.
3. Specify file-level changes, tests, failure cases, and validation commands.
4. Write the plan using `write_plan`, then call `exit_plan_mode` for user approval.

## Success evidence and checkpoint

The typed Session state must reference the current session's fixed plan in the user-state directory, and ExitPlanMode must be approved. Do not modify production code while in Plan Mode or treat silence as approval.
