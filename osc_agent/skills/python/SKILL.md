---
name: python
description: Python packaging, testing, typing, and idiomatic contribution guidance.
schema_version: 1
version: 1.0.0
applies_to: [python]
required_tools: [read_file, glob, bash]
permissions: [read, shell]
input_contract: Repository metadata, relevant Python source, and task constraints.
output_contract: Scoped Python implementation and verification guidance.
---

# Python Skill

Use this skill when the target repository contains Python files, `pyproject.toml`, `requirements.txt`, or pytest-style tests.

- Prefer small, focused changes.
- Read `pyproject.toml`, `requirements.txt`, and test configuration before changing behavior.
- Add or update focused pytest coverage for bug fixes.
- Run the narrowest relevant test first, then broader tests when shared behavior changes.
