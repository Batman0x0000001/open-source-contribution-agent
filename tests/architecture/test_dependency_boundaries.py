"""验证架构依赖边界的契约、边界条件与回归行为。"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "osc_agent" / "runtime"
FORBIDDEN_RUNTIME_IMPORTS = (
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


def test_runtime_does_not_depend_on_workflows_or_provider_implementation() -> None:
    violations: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        for module in _imports(path):
            if module.startswith(FORBIDDEN_RUNTIME_IMPORTS):
                violations.append(f"{path.relative_to(PROJECT_ROOT)} -> {module}")

    assert not violations, "Runtime dependency violations:\n" + "\n".join(violations)


def test_agent_runtime_query_is_the_async_generator_entrypoint() -> None:
    from osc_agent.runtime.query import AgentRuntime

    assert inspect.isasyncgenfunction(AgentRuntime.query)


def test_agent_runner_delegates_to_agent_runtime_query() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "agents" / "runner.py").read_text(encoding="utf-8")

    assert "self.runtime.query(" in source
    assert "model_gateway.stream(" not in source


def test_skill_tool_delegates_to_skill_executor() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "skills" / "tool.py").read_text(encoding="utf-8")

    assert "self.executor.execute(" in source
    assert "AgentRuntime(" not in source
    assert "model_gateway.stream(" not in source
