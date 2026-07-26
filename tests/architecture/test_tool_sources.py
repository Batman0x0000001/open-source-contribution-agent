from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_new_file_tools_do_not_import_legacy_schema_registry() -> None:
    path = PROJECT_ROOT / "osc_agent" / "tools" / "file_tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "osc_agent.tools.files"
        for alias in node.names
    }

    assert imported_names == {"edit_file", "glob_files", "write_file"}
    assert "FILE_TOOLS" not in imported_names


def test_core_registry_registers_each_tool_once() -> None:
    source = (PROJECT_ROOT / "osc_agent" / "tools" / "core.py").read_text(encoding="utf-8")

    assert source.count("ReadFileTool(instructions)") == 1
    assert source.count("WriteFileTool(instructions)") == 1
    assert source.count("EditFileTool(instructions)") == 1
    assert source.count("GlobTool()") == 1
    assert source.count("GrepTool(instructions)") == 1
    assert source.count("ShellTool(environment_allowlist=subprocess_env_allowlist)") == 1
    assert source.count("GitStatusTool()") == 1
    assert source.count("GitDiffTool(tool_result_store)") == 1
    assert source.count("GitLogTool()") == 1
    assert source.count("GitHubListIssuesTool()") == 1
    assert source.count("GitHubGetIssueTool()") == 1
    assert source.count("AskUserQuestionTool()") == 1
    assert source.count("EnterPlanModeTool()") == 1
    assert source.count("WritePlanTool()") == 1
    assert source.count("ReadPlanTool()") == 1
    assert source.count("ExitPlanModeTool()") == 1
    assert source.count("EnterWorktreeTool(worktree_manager, instructions)") == 1
    assert source.count("ExitWorktreeTool(worktree_manager, instructions)") == 1
    assert source.count("ReadToolResultTool(tool_result_store)") == 1


def test_legacy_runtime_and_registration_sources_are_deleted() -> None:
    legacy_paths = [
        PROJECT_ROOT / "osc_agent" / "agent_loop.py",
        PROJECT_ROOT / "osc_agent" / "harness" / "subagent.py",
        PROJECT_ROOT / "osc_agent" / "harness" / "teams.py",
        PROJECT_ROOT / "osc_agent" / "harness" / "tool_registry.py",
        PROJECT_ROOT / "osc_agent" / "skills" / "registry.py",
        PROJECT_ROOT / "osc_agent" / "support",
        PROJECT_ROOT / "osc_agent" / "workflows",
        PROJECT_ROOT / "osc_agent" / "tools" / "todo_tools.py",
        PROJECT_ROOT / "osc_agent" / "tools" / "task_tools.py",
        PROJECT_ROOT / "osc_agent" / "tools" / "mcp_tools.py",
    ]

    assert not [path for path in legacy_paths if path.exists()]


def test_redundant_repo_and_pr_tools_are_deleted() -> None:
    assert not (PROJECT_ROOT / "osc_agent" / "tools" / "repo.py").exists()
    assert not (PROJECT_ROOT / "osc_agent" / "tools" / "pr.py").exists()
