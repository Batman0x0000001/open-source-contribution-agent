---
name: open-source
description: Open source contribution workflow, maintainer empathy, and PR preparation guidance.
schema_version: 1
version: 1.0.0
applies_to: [open-source]
required_tools: [read_file, glob, git_status, git_diff]
permissions: [read]
input_contract: Repository state, issue evidence, and approved contribution scope.
output_contract: Auditable contribution workflow and maintainer-facing guidance.
---

# Open Source Skill

Use this skill for contribution-oriented work across unfamiliar repositories.

- Read README and contribution guidance before editing.
- Keep the change scoped to the reported issue or requested stage.
- Avoid unrelated cleanup.
- Summarize modified files, tests run, residual risks, and a concise PR title/body draft.
