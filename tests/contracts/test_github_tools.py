from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess

import osc_agent.tools.github as github_module
from osc_agent.runtime.models import RuntimeMessage, ToolResultBlock, ToolUseBlock, ToolUseContext
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
        lambda repo_url, issue_number, max_comments: {"ok": True, "issue": ISSUE},
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
        "comments": [],
    }
    trust_marker = {
        "content_source": "github",
        "trust": "untrusted_external",
    }
    assert listed.data == {**trust_marker, "issues": [expected]}
    assert fetched.data == {**trust_marker, "issue": expected}


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
        lambda repo_url, issue_number, max_comments: {"ok": False, "error": "rate limited", "issue": {}},
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


def test_github_issue_comments_are_bounded_and_normalized(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url: str, token=None):
        calls.append(url)
        if url.endswith("/issues/7"):
            return {"ok": True, "data": dict(ISSUE)}
        return {
            "ok": True,
            "data": [
                {
                    "user": {"login": "maintainer"},
                    "body": "Please add a regression test.",
                    "created_at": "2026-07-23T00:00:00Z",
                    "html_url": "https://github.com/acme/project/issues/7#issuecomment-1",
                }
            ],
        }

    monkeypatch.setattr(github_module, "_github_get_json", fake_get)
    fetched = github_module.fetch_issue(
        "https://github.com/acme/project",
        7,
        max_comments=1,
    )
    normalized = github_module._normalize_issue(fetched["issue"])

    assert normalized["comments"][0]["author"] == "maintainer"
    assert "per_page=1" in calls[-1]


def test_remote_mismatch_requires_structured_user_confirmation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/local/project.git"],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(
        github_module,
        "fetch_issue",
        lambda repo_url, issue_number, max_comments: {"ok": True, "issue": ISSUE},
    )
    executor = ToolExecutor(build_core_tool_registry())
    call = ToolUseBlock(
        id="get",
        name="github_get_issue",
        input={"repo_url": "https://github.com/acme/project", "issue_number": 7},
    )
    denied = asyncio.run(executor.execute(call, context(tmp_path)))
    approved_context = context(tmp_path).model_copy(
        update={
            "transcript_messages": [
                RuntimeMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="question",
                            content={
                                "data": {
                                    "questions": [
                                        {
                                            "id": "remote",
                                            "purpose": "remote_mismatch",
                                            "remote_mismatch_reference": {
                                                "local_origin": "local/project",
                                                "requested_repository": "acme/project",
                                            },
                                        }
                                    ],
                                    "answers": [
                                        {
                                            "question_id": "remote",
                                            "selected_option_id": "proceed",
                                            "custom_text": None,
                                        }
                                    ],
                                }
                            },
                        )
                    ],
                )
            ]
        }
    )
    approved = asyncio.run(executor.execute(call, approved_context))
    different = asyncio.run(
        executor.execute(
            ToolUseBlock(
                id="different",
                name="github_get_issue",
                input={
                    "repo_url": "https://github.com/another/project",
                    "issue_number": 7,
                },
            ),
            approved_context,
        )
    )

    assert denied.error and denied.error.code == "TOOL_VALIDATION_FAILED"
    assert approved.error is None
    assert different.error and different.error.code == "TOOL_VALIDATION_FAILED"
