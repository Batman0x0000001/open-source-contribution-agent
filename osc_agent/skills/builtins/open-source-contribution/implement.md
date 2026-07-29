# Implement

## Input and prerequisites

Require an approved saved plan. Enter a real Git worktree before editing and use its cwd for all file and shell tools.

## Execution

1. Establish the relevant baseline test or reproduction.
2. Implement the smallest plan-compliant change in the repository's existing style.
3. Run focused tests after each coherent change. Feed failures back into the agent loop, inspect evidence, and repair adaptively.
4. Run the agreed broader validation.
5. After the final successful primary test, call the `verify` Agent with the original goal, an honest implementation summary, and focused risks. It must inspect the actual diff, execute checks, and include an adversarial probe.
6. Repair a `FAIL`, rerun the primary test, and invoke Verify again. A `PARTIAL` may proceed only through the dedicated waiver bound to that child Session.
7. After Verify `PASS` or an approved `PARTIAL` waiver, call `git_diff` to capture the final complete change snapshot.

## Success evidence and checkpoint

The transcript must contain a successful primary test after the final edit, later independent verification evidence, and a still later complete Git snapshot. If primary tests cannot run, request the existing test waiver. If Verify is `PARTIAL`, use only the independent-verification waiver; `FAIL` cannot be waived. Do not weaken tests, silently broaden scope, commit, push, discard changes, or claim success after failing validation.
