---
name: tests
description: Test discovery, focused regression coverage, and failure analysis guidance.
schema_version: 1
version: 1.0.0
applies_to: [testing]
required_tools: [read_file, glob, bash]
permissions: [read, shell]
input_contract: Changed files, available test commands, and acceptance criteria.
output_contract: Executable verification plan and failure diagnosis guidance.
---

# Tests Skill

Use this skill when adding, fixing, or diagnosing tests.

- Reproduce failures with the smallest command that exercises the behavior.
- Add regression tests near related existing tests.
- Keep fixtures local and readable unless existing shared fixtures already fit.
- Report commands run and their results.
