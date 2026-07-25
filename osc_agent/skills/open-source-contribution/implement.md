# Implement

## Input and prerequisites

Require an approved saved plan. Enter a real Git worktree before editing and use its cwd for all file and shell tools.

## Execution

1. Establish the relevant baseline test or reproduction.
2. Implement the smallest plan-compliant change in the repository's existing style.
3. Run focused tests after each coherent change. Feed failures back into the agent loop, inspect evidence, and repair adaptively.
4. Run the agreed broader validation and inspect `git status`/`git diff`.

## Success evidence and checkpoint

The transcript must contain validation output and the worktree must contain the reviewable diff. Do not weaken tests, silently broaden scope, commit, push, discard changes, or claim success after failing validation.
