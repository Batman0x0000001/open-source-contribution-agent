from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess

from osc_agent.application import build_application
from osc_agent.config import Settings
from osc_agent.runtime.gateway import ModelCompleted, ModelEvent, ModelRequest
from osc_agent.runtime.models import (
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
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "base")
    state_root = tmp_path / "user-state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state_root))

    gateway = ScriptedGateway(
        [
            tool("discover-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "discover.md"}),
            tool("choose", "ask_user_question", {"questions": [{"header": "Scope", "question": "Choose contribution?", "options": [{"label": "Small fix", "description": "Bounded"}, {"label": "Stop", "description": "Do nothing"}]}]}),
            tool("enter-plan", "enter_plan_mode", {}),
            tool("design-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "design.md"}),
            tool("write-plan", "write_plan", {"content": "# Plan\n\nCreate a local contribution fixture and verify it."}),
            tool("exit-plan", "exit_plan_mode", {}),
            tool("enter-worktree", "enter_worktree", {"name": "contribution-e2e"}),
            tool("implement-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "implement.md"}),
            tool("implementation", "write_file", {"path": "CONTRIBUTION.txt", "content": "implemented\n"}),
            tool("draft-resource", "read_skill_resource", {"skill": "open-source-contribution", "path": "pr-draft.md"}),
            tool("draft", "write_file", {"path": "PR_DRAFT.md", "content": "# Summary\n\nLocal contribution fixture.\n"}),
            tool("keep-worktree", "exit_worktree", {"action": "keep"}),
            RuntimeMessage(role="assistant", content=[TextBlock(text="Contribution draft is ready.")]),
        ]
    )

    async def approve(_decision) -> bool:
        return True

    async def answer(questions):
        return {questions[0]["question"]: "Small fix"}

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
    paths = ApplicationStatePaths.for_repository(repo, state_root=state_root)
    worktree = paths.worktrees / "contribution-e2e"
    assert (worktree / "CONTRIBUTION.txt").read_text(encoding="utf-8") == "implemented\n"
    assert (worktree / "PR_DRAFT.md").is_file()
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
