"""验证 Tool adapter、共享能力和显式注册之间的架构边界。"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.contracts.registry_factory import build_test_tool_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "osc_agent"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_core_registry_has_one_authoritative_tool_set() -> None:
    registry = build_test_tool_registry()

    assert registry.names() == sorted(
        {
            "read_file", "write_file", "edit_file", "glob", "grep", "bash",
            "git_status", "git_diff", "git_log", "github_list_issues",
            "github_get_issue", "ask_user_question", "enter_plan_mode",
            "write_plan", "read_plan", "exit_plan_mode", "enter_worktree",
            "exit_worktree", "read_tool_result",
        }
    )


def test_core_registry_injects_question_handler_into_question_tool() -> None:
    async def answer(_questions):
        return {}

    registry = build_test_tool_registry(question_handler=answer)

    assert registry.get("ask_user_question").question_handler is answer


def test_runtime_and_shared_capabilities_do_not_import_tool_implementations() -> None:
    roots = [
        PACKAGE_ROOT / "runtime",
        PACKAGE_ROOT / "workspaces",
        PACKAGE_ROOT / "processes",
        PACKAGE_ROOT / "subagents",
        PACKAGE_ROOT / "bot",
    ]
    violations = [
        f"{path.relative_to(PROJECT_ROOT)} -> {module}"
        for root in roots
        for path in root.rglob("*.py")
        for module in _imports(path)
        if module.startswith("osc_agent.tools")
    ]

    assert not violations, "shared layer imports Tool adapters:\n" + "\n".join(violations)


def test_legacy_tool_helper_modules_are_deleted() -> None:
    old_paths = [
        "filesystem_operations.py",
        "filesystem_tools.py",
        "interaction.py",
        "path_policy.py",
        "process_runner.py",
        "state_tools.py",
    ]

    assert not [path for name in old_paths if (path := PACKAGE_ROOT / "tools" / name).exists()]
