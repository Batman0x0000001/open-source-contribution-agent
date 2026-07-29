"""验证最终架构不变量的契约、边界条件与回归行为。"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "osc_agent"


def _class_definitions(name: str) -> list[Path]:
    matches: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(isinstance(node, ast.ClassDef) and node.name == name for node in ast.walk(tree)):
            matches.append(path)
    return matches


def test_old_execution_architecture_is_absent() -> None:
    forbidden = [
        PACKAGE_ROOT / "agent_loop.py",
        PACKAGE_ROOT / "harness",
        PACKAGE_ROOT / "skills" / "registry.py",
        PACKAGE_ROOT / "skills" / "runner.py",
        PACKAGE_ROOT / "skills" / "executor.py",
        PACKAGE_ROOT / "skills" / "tool.py",
        PACKAGE_ROOT / "agent_service.py",
        PACKAGE_ROOT / "composition.py",
        PACKAGE_ROOT / "cli.py",
        PACKAGE_ROOT / "cli_session.py",
        PACKAGE_ROOT / "config.py",
        PACKAGE_ROOT / "runtime_config.py",
        PACKAGE_ROOT / "bot" / "service_runner.py",
        PACKAGE_ROOT / "bot" / "models.py",
        PACKAGE_ROOT / "bot" / "store.py",
        PACKAGE_ROOT / "bot" / "state_machine.py",
        PACKAGE_ROOT / "bot" / "control.py",
        PACKAGE_ROOT / "bot" / "worker.py",
        PACKAGE_ROOT / "bot" / "doctor.py",
        PACKAGE_ROOT / "bot" / "sandbox.py",
        PACKAGE_ROOT / "bot" / "policy.py",
        PACKAGE_ROOT / "bot" / "application_factory.py",
        PACKAGE_ROOT / "bot" / "agent_inputs.py",
        PACKAGE_ROOT / "bot" / "artifact_tools.py",
        PACKAGE_ROOT / "bot" / "control_service.py",
        PACKAGE_ROOT / "bot" / "worker_service.py",
        PACKAGE_ROOT / "bot" / "github_app.py",
        PACKAGE_ROOT / "bot" / "job_workspace.py",
        PACKAGE_ROOT / "bot" / "outbox.py",
        PACKAGE_ROOT / "bot" / "publisher.py",
        PACKAGE_ROOT / "bot" / "webhook.py",
        PACKAGE_ROOT / "bot" / "cleanup.py",
        PACKAGE_ROOT / "bot" / "operations.py",
        PACKAGE_ROOT / "agents",
        PACKAGE_ROOT / "runtime" / "models.py",
        PACKAGE_ROOT / "runtime" / "completion.py",
        PACKAGE_ROOT / "runtime" / "instructions.py",
        PACKAGE_ROOT / "runtime" / "session_summary.py",
        PACKAGE_ROOT / "runtime" / "state_paths.py",
        PACKAGE_ROOT / "workflows" / "contribution" / "agents.py",
        PACKAGE_ROOT / "workflows" / "contribution" / "design.py",
        PACKAGE_ROOT / "workflows" / "contribution" / "discover.py",
        PACKAGE_ROOT / "workflows" / "contribution" / "implementation.py",
        PACKAGE_ROOT / "workflows" / "contribution" / "pr_draft.py",
    ]
    assert not [path for path in forbidden if path.exists()]
    assert not (PROJECT_ROOT / "tests_v2").exists()


def test_query_and_tool_execution_have_one_authoritative_definition() -> None:
    assert _class_definitions("AgentRuntime") == [PACKAGE_ROOT / "runtime" / "query.py"]
    assert _class_definitions("ToolExecutor") == [PACKAGE_ROOT / "runtime" / "tool_execution.py"]
    assert _class_definitions("SkillPreparer") == [PACKAGE_ROOT / "skills" / "preparer.py"]
    assert not _class_definitions("SkillExecutor")
    assert _class_definitions("BotStore") == [
        PACKAGE_ROOT / "bot" / "persistence" / "store.py"
    ]
    assert _class_definitions("SqliteSessionStore") == [
        PACKAGE_ROOT / "bot" / "persistence" / "session_store.py"
    ]


def test_workspace_capabilities_have_explicit_names() -> None:
    expected = [
        PACKAGE_ROOT / "workspaces" / "git_worktree.py",
        PACKAGE_ROOT / "tools" / "worktree.py",
        PACKAGE_ROOT / "bot" / "control" / "job_workspace.py",
    ]
    forbidden = [
        PACKAGE_ROOT / "isolation" / "worktree.py",
        PACKAGE_ROOT / "tools" / "worktree_tools.py",
        PACKAGE_ROOT / "bot" / "workspace.py",
    ]

    assert all(path.is_file() for path in expected)
    assert not any(path.exists() for path in forbidden)
    assert _class_definitions("GitWorktreeManager") == [expected[0]]
    assert _class_definitions("BotJobWorkspacePreparer") == [expected[2]]


def test_only_provider_calls_anthropic_sdk() -> None:
    importers: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if module == "anthropic" or "anthropic" in names:
                importers.append(path)
                break
    assert importers == [PACKAGE_ROOT / "providers" / "anthropic.py"]


def test_no_generic_workflow_dsl_was_added() -> None:
    names = {path.stem.casefold() for path in PACKAGE_ROOT.rglob("*.py")}
    assert not names & {"dag", "graph", "workflow_dsl", "node_engine"}


def test_removed_minimum_version_capabilities_are_absent() -> None:
    assert not (PACKAGE_ROOT / "tools" / "repo.py").exists()
    assert not (PACKAGE_ROOT / "tools" / "pr.py").exists()
    for name in ("docs", "python", "javascript", "tests"):
        assert not (PACKAGE_ROOT / "skills" / name / "SKILL.md").exists()
    subagent_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PACKAGE_ROOT / "subagents").rglob("*.py")
    )
    assert "background" not in subagent_sources


def test_subagent_registry_has_no_file_or_plugin_loading_path() -> None:
    subagent_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PACKAGE_ROOT / "subagents").rglob("*.py")
    )
    for forbidden in (
        ".osc_agent/agents",
        ".claude/agents",
        "load_agent",
        "plugin_agent",
        "AgentLoader",
    ):
        assert forbidden not in subagent_sources
