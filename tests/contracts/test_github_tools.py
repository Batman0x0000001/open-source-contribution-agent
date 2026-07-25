from __future__ import annotations

import asyncio
from pathlib import Path

import osc_agent.tools.github as github_module
from osc_agent.runtime.models import ToolUseBlock, ToolUseContext
from osc_agent.runtime.tool_execution import ToolExecutor
from tests.contracts.registry_factory import build_test_tool_registry as build_core_tool_registry
from osc_agent.tools.github import GitHubGetIssueInput, GitHubListIssuesInput


ISSUE = {
    "number": 7,
    "title": "Fix parser",
    "body": "The parser should preserve escaped values.",
    "state": "open",
    "html_url": "https://github.com/acme/project/issues/7",
    "labels": [{"name": "Bug"}, {"name": "help wanted"}],
    "updated_at": "2026-07-22T00:00:00Z",
}


def context(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(session_id="session-1", working_directory=str(tmp_path), repository_root=str(tmp_path), state_directory=str(tmp_path / "state"))


def test_github_tools_are_read_only_and_concurrency_safe() -> None:
    registry = build_core_tool_registry()
    list_tool = registry.get("github_list_issues")
    get_tool = registry.get("github_get_issue")

    list_input = GitHubListIssuesInput(repo_url="https://github.com/acme/project")
    get_input = GitHubGetIssueInput(repo_url="https://github.com/acme/project", issue_number=7)
    assert list_tool.is_read_only(list_input) is True
    assert list_tool.is_concurrency_safe(list_input) is True
    assert get_tool.is_read_only(get_input) is True
    assert get_tool.is_concurrency_safe(get_input) is True


def test_github_external_json_is_normalized_before_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        github_module,
        "fetch_issues",
        lambda repo_url, labels, updated_within_days: {"ok": True, "issues": [ISSUE]},
    )
    monkeypatch.setattr(
        github_module,
        "fetch_issue",
        lambda repo_url, issue_number: {"ok": True, "issue": ISSUE},
    )
    executor = ToolExecutor(build_core_tool_registry())

    async def run():
        listed = await executor.execute(
            ToolUseBlock(
                id="list-1",
                name="github_list_issues",
                input={"repo_url": "https://github.com/acme/project"},
            ),
            context(tmp_path),
        )
        fetched = await executor.execute(
            ToolUseBlock(
                id="get-1",
                name="github_get_issue",
                input={"repo_url": "https://github.com/acme/project", "issue_number": 7},
            ),
            context(tmp_path),
        )
        return listed, fetched

    listed, fetched = asyncio.run(run())

    expected = {
        "number": 7,
        "title": "Fix parser",
        "body": "The parser should preserve escaped values.",
        "state": "open",
        "url": "https://github.com/acme/project/issues/7",
        "labels": ["bug", "help wanted"],
        "updated_at": "2026-07-22T00:00:00Z",
    }
    assert listed.data == {"issues": [expected]}
    assert fetched.data == {"issue": expected}


def test_github_tool_rejects_non_github_url_before_network(monkeypatch, tmp_path: Path) -> None:
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return {"ok": True, "issues": []}

    monkeypatch.setattr(github_module, "fetch_issues", unexpected)
    result = asyncio.run(
        ToolExecutor(build_core_tool_registry()).execute(
            ToolUseBlock(
                id="list-1",
                name="github_list_issues",
                input={"repo_url": "https://example.com/acme/project"},
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "TOOL_VALIDATION_FAILED"
    assert called is False


def test_github_read_failure_is_structured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        github_module,
        "fetch_issue",
        lambda repo_url, issue_number: {"ok": False, "error": "rate limited", "issue": {}},
    )
    result = asyncio.run(
        ToolExecutor(build_core_tool_registry()).execute(
            ToolUseBlock(
                id="get-1",
                name="github_get_issue",
                input={"repo_url": "https://github.com/acme/project", "issue_number": 7},
            ),
            context(tmp_path),
        )
    )

    assert result.error and result.error.code == "GITHUB_READ_FAILED"
    assert result.error.retryable is True
