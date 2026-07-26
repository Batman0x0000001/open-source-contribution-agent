from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess

from osc_agent.application import build_application
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import (
    ApprovalResponse,
    ResumeQueryParams,
    RunCompleted,
    RuntimeMessage,
    TextBlock,
    ToolUseBlock,
)
from osc_agent.runtime.state_paths import ApplicationStatePaths


class ScriptedGateway:
    def __init__(self, messages: list[RuntimeMessage]) -> None:
        self.messages = list(messages)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        message = self.messages.pop(0)
        stop_reason = "tool_use" if any(isinstance(block, ToolUseBlock) for block in message.content) else "end_turn"
        yield ModelCompleted(message=message, stop_reason=stop_reason)


def tool(call_id: str, name: str, payload: dict) -> RuntimeMessage:
    return RuntimeMessage(
        role="assistant",
        content=[ToolUseBlock(id=call_id, name=name, input=payload)],
    )


def git(cwd: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True)


def test_contribution_skill_runs_through_plan_worktree_draft_and_resume(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "tests@example.com")
    git(repo, "config", "user.name", "Tests")
    (repo / "README.md").write_text("# Fixture", encoding="utf-8")
    (repo / "AGENTS.md").write_text("Run focused tests before drafting.", encoding="utf-8")
    git(repo, "add", "README.md", "AGENTS.md")
    git(repo, "commit", "-m", "base")
    state_root = tmp_path / "user-state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state_root))

    gateway = ScriptedGateway(
        [
            RuntimeMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="explore-runtime",
                        name="agent",
                        input={
                            "agent": "explore",
                            "task": "Trace the runtime and worktree path flow.",
                            "arguments": {
                                "scope_paths": ["."],
                                "context_summary": "Investigate a cross-module Windows resume issue.",
                            },
                        },
                    ),
                    ToolUseBlock(
                        id="explore-tests",
                        name="agent",
                        input={
                            "agent": "explore",
                            "task": "Find repository instruction expectations and tests.",
                            "arguments": {
                                "scope_paths": ["."],
                                "context_summary": "Investigate a cross-module Windows resume issue.",
                            },
                        },
                    ),
                ],
            ),
            RuntimeMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            '{"summary":"runtime path traced","findings":[{"claim":"The fixture '
                            'contains repository instructions","confidence":"high","evidence":'
                            '[{"path":"AGENTS.md","line_start":1,"observation":"Root instruction"}]}],'
                            '"relevant_files":["AGENTS.md"],"likely_change_locations":[],'
                            '"recommended_tests":["resume contract"],"unresolved_questions":[]}'
                        )
                    )
                ],
            ),
            RuntimeMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            '{"summary":"test evidence found","findings":[{"claim":"The repository '
                            'has a fixture entry point","confidence":"high","evidence":[{"path":'
                            '"README.md","line_start":1,"observation":"Fixture repository"}]}],'
                            '"relevant_files":["README.md"],"likely_change_locations":[],'
                            '"recommended_tests":["worktree contract"],"unresolved_questions":[]}'
                        )
                    )
                ],
            ),
            tool("discover-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "discover.md"}),
            tool("choose", "ask_user_question", {"questions": [{"id": "scope", "header": "Scope", "question": "Choose contribution?", "options": [{"id": "small_fix", "label": "Small fix", "description": "Bounded"}, {"id": "stop", "label": "Stop", "description": "Do nothing"}]}]}),
            tool("enter-plan", "enter_plan_mode", {}),
            tool("design-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "design.md"}),
            tool("write-plan", "write_plan", {"content": "# Plan\n\nCreate a local contribution fixture and verify it."}),
            tool("exit-plan", "exit_plan_mode", {}),
            tool("enter-worktree", "enter_worktree", {"name": "contribution-e2e"}),
            tool("implement-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "implement.md"}),
            tool("implementation", "write_file", {"path": "CONTRIBUTION.txt", "content": "implemented\n"}),
            tool("verification", "powershell", {"command": "python -m unittest"}),
            tool(
                "independent-verification",
                "agent",
                {
                    "agent": "verify",
                    "task": "Independently verify the contribution fixture.",
                    "arguments": {
                        "original_goal": "Create a small contribution fixture.",
                        "implementation_summary": "Added CONTRIBUTION.txt in the approved worktree.",
                        "focus_areas": ["empty test discovery"],
                    },
                },
            ),
            tool(
                "verify-probe",
                "powershell",
                {"command": "python -c \"assert 0 == 0\""},
            ),
            RuntimeMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            '{"evidence_type":"independent_verification","verdict":"PASS",'
                            '"summary":"fixture independently verified","checks":[{"name":'
                            '"empty boundary probe","command":"python -c \\"assert 0 == 0\\"",'
                            '"exit_code":0,"output_excerpt":"","result":"pass","adversarial":true}],'
                            '"risks":[],"unverified":[]}'
                        )
                    )
                ],
            ),
            tool("snapshot", "git_diff", {}),
            tool("draft-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "pr-draft.md"}),
            RuntimeMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            "# PR Draft\n\nContribution fixture with successful unittest verification.\n\n"
                            "## Independent Verification\n\nPASS with an empty-input adversarial probe."
                        )
                    )
                ],
            ),
        ]
    )

    async def approve(_decision) -> ApprovalResponse:
        return ApprovalResponse(choice="allow_once")

    async def answer(questions):
        return {questions[0]["id"]: "small_fix"}

    services = build_application(
        settings=Settings(model_id="test-model"),
        repo_root=repo,
        approval_handler=approve,
        question_handler=answer,
        model_gateway=gateway,
    )
    session_id = "contribution-session"

    async def run_contribution():
        return [
            event
            async for event in services.skill_command_runner.run(
                name="open-source-contribution",
                arguments={"repo_url": "https://github.com/example/project", "goal": "small fixture"},
                session_id=session_id,
                working_directory=str(repo),
            )
        ]

    events = asyncio.run(run_contribution())
    assert isinstance(events[-1], RunCompleted)
    assert "Run focused tests before drafting." in gateway.requests[0].system_prompt
    assert "- explore:" in gateway.requests[0].system_prompt
    assert "- verify:" in gateway.requests[0].system_prompt
    assert "agent" in {schema["name"] for schema in gateway.requests[0].tools}
    explore_requests = [
        request
        for request in gateway.requests
        if request.system_prompt.startswith("Explore the repository")
    ]
    assert len(explore_requests) == 2
    assert all(
        {schema["name"] for schema in request.tools}
        == {"read_file", "glob", "grep", "git_status", "git_diff", "git_log"}
        for request in explore_requests
    )
    assert all(
        "Progress dynamically from evidence" not in request.messages[0].content[0].text
        for request in explore_requests
    )
    verify_requests = [
        request
        for request in gateway.requests
        if request.system_prompt.startswith("You are an independent verification specialist")
    ]
    assert len(verify_requests) == 2
    assert {
        schema["name"]
        for schema in verify_requests[0].tools
    } == {
        "read_file",
        "glob",
        "grep",
        "git_status",
        "git_diff",
        "git_log",
        "powershell",
    }
    assert "Progress dynamically from evidence" not in verify_requests[0].messages[0].content[0].text
    paths = ApplicationStatePaths.for_repository(repo, state_root=state_root)
    worktree = paths.worktrees / "contribution-e2e"
    assert (worktree / "CONTRIBUTION.txt").read_text(encoding="utf-8") == "implemented\n"
    assert not (worktree / "PR_DRAFT.md").exists()
    assert (paths.plans / f"{session_id}.md").is_file()
    assert (paths.sessions / f"{session_id}.jsonl").is_file()
    assert not (repo / ".osc_agent").exists()

    resumed_gateway = ScriptedGateway(
        [RuntimeMessage(role="assistant", content=[TextBlock(text="Resumed successfully.")])]
    )
    resumed_services = build_application(
        settings=Settings(model_id="different-current-model"),
        repo_root=repo,
        model_gateway=resumed_gateway,
    )

    async def resume():
        return [
            event
            async for event in resumed_services.runtime.query(
                ResumeQueryParams(session_id=session_id, repository_root=str(repo))
            )
        ]

    resumed_events = asyncio.run(resume())
    assert isinstance(resumed_events[-1], RunCompleted)
    assert resumed_gateway.requests[0].model == "test-model"
    assert "Run focused tests before drafting." in resumed_gateway.requests[0].system_prompt
