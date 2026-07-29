"""验证架构依赖边界的契约、边界条件与回归行为。"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "osc_agent" / "runtime"
FORBIDDEN_RUNTIME_IMPORTS = (
    "osc_agent.application",
    "osc_agent.bot",
    "osc_agent.completion.evidence",
    "osc_agent.skills",
    "osc_agent.subagents",
    "osc_agent.tools",
    "osc_agent.workflows",
    "osc_agent.providers.anthropic",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_runtime_depends_only_on_core_contracts_and_shared_capabilities() -> None:
    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        for module in _imports(path):
            if module.startswith(FORBIDDEN_RUNTIME_IMPORTS):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")

    assert not violations, "Runtime dependency violations:\n" + "\n".join(violations)


def test_runtime_does_not_depend_on_subagents() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {module}"
        for path in sorted(RUNTIME_ROOT.rglob("*.py"))
        for module in _imports(path)
        if module.startswith("osc_agent.subagents")
    ]

    assert not violations, "Runtime must not depend on subagents:\n" + "\n".join(violations)


def test_runtime_legacy_aggregate_modules_are_absent() -> None:
    for name in (
        "models.py",
        "completion.py",
        "instructions.py",
        "session_summary.py",
        "state_paths.py",
    ):
        assert not (RUNTIME_ROOT / name).exists()


def test_agent_runtime_query_is_the_async_generator_entrypoint() -> None:
    from osc_agent.runtime.query import AgentRuntime

    assert inspect.isasyncgenfunction(AgentRuntime.query)


def test_subagent_runner_delegates_to_agent_runtime_query() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "subagents" / "runner.py").read_text(encoding="utf-8")

    assert "self.runtime.query(" in source
    assert "model_gateway.stream(" not in source


def test_skill_tool_delegates_to_skill_preparer() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "skills" / "invocation_tool.py").read_text(encoding="utf-8")

    assert "self.preparer.prepare(" in source
    assert "AgentRuntime(" not in source
    assert "model_gateway.stream(" not in source


def test_skills_do_not_depend_on_subagents() -> None:
    skill_root = PROJECT_ROOT / "osc_agent" / "skills"
    violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {module}"
        for path in sorted(skill_root.rglob("*.py"))
        for module in _imports(path)
        if module.startswith("osc_agent.subagents")
    ]

    assert not violations, "Skills must not depend on subagents:\n" + "\n".join(violations)


def test_shared_and_control_packages_do_not_import_runtime_context() -> None:
    roots = (
        PROJECT_ROOT / "osc_agent" / "processes",
        PROJECT_ROOT / "osc_agent" / "workspaces",
        PROJECT_ROOT / "osc_agent" / "bot" / "domain",
        PROJECT_ROOT / "osc_agent" / "bot" / "control",
    )
    violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {module}"
        for root in roots
        for path in root.rglob("*.py")
        for module in _imports(path)
        if module.startswith(("osc_agent.runtime.state", "osc_agent.runtime.hooks"))
    ]
    assert not violations
