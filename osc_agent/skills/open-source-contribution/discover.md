# Discover

## Input and prerequisites

- Use the invocation `repo_url`, current Session transcript, and actual repository cwd.
- Read README, contribution guidance, changelog, manifests, entry points, tests, and a depth-limited tree. Missing files are not errors.

## Execution

1. Explain the core runtime path and project conventions using file/function evidence.
2. Inspect current open issues when tools permit, but do not invent remote state.
3. Derive 1–3 contribution candidates from issues or concrete code gaps. For each give scope, files, verification, risk, and maintainer-acceptance likelihood.
4. Use `ask_user_question` to let the user select one candidate or provide a focused alternative.

## Success evidence and checkpoint

The transcript must contain repository evidence, ranked candidates, and the user's explicit selection. Do not edit code, create a worktree, or choose on the user's behalf.
