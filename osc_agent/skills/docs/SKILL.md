---
name: docs
description: Documentation, README, changelog, and contribution guide updates.
schema_version: 1
version: 1.0.0
applies_to: [documentation]
required_tools: [read_file, glob]
permissions: [read]
input_contract: Documentation files and the approved contribution scope.
output_contract: Reviewable documentation guidance with verification steps.
---

# Docs Skill

Use this skill when the task touches README files, documentation directories, examples, changelogs, or contributor-facing text.

- Preserve existing document tone and structure.
- Prefer additive updates unless the user asks for replacement.
- Keep commands and file paths accurate.
- Verify examples still match the current code.
