---
name: javascript
description: JavaScript and TypeScript package, test, and frontend contribution guidance.
schema_version: 1
version: 1.0.0
applies_to: [javascript, typescript]
required_tools: [read_file, glob, bash]
permissions: [read, shell]
input_contract: JavaScript project metadata, source evidence, and task constraints.
output_contract: Scoped JavaScript implementation and test guidance.
---

# JavaScript Skill

Use this skill when the repository contains `package.json`, `tsconfig.json`, JavaScript, TypeScript, or frontend source files.

- Inspect package scripts before choosing commands.
- Respect existing formatter, linter, and framework conventions.
- Prefer targeted tests or type checks before full project commands.
- Keep UI changes consistent with existing components and styling.
