"""验证命令行交互体验的契约、边界条件与回归行为。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from typer.testing import CliRunner

from osc_agent.cli.app import app
from osc_agent.cli.agent import run_conversation
from osc_agent.runtime.events import Complete, RunCompleted, RuntimeEvent
from osc_agent.runtime.messages import RuntimeMessage, TextBlock
from osc_agent.runtime.session import SessionMetadata
from osc_agent.runtime.session_store import FileSessionStore
from tests.runtime_factories import agent_run_state


def _store_with_session(root: Path, *, session_id: str = "session-1") -> FileSessionStore:
    store = FileSessionStore(root)
    store.create(
        SessionMetadata(
            schema_version=6,
            session_id=session_id,
            workspace_root=str(root),
            model="test-model",
        ),
        agent_run_state(str(root), status="completed"),
    )
    store.append_message(
        session_id,
        RuntimeMessage(role="user", content=[TextBlock(text="private goal")]),
    )
    return store


def test_session_cli_hides_messages_by_default(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state))
    from osc_agent.application.state_paths import ApplicationStatePaths

    store = FileSessionStore(ApplicationStatePaths.for_repository(tmp_path).sessions)
    store.create(
        SessionMetadata(
            schema_version=6,
            session_id="session-1",
            workspace_root=str(tmp_path),
            model="test-model",
        ),
        agent_run_state(str(tmp_path)),
    )
    store.append_message(
        "session-1",
        RuntimeMessage(role="user", content=[TextBlock(text="private goal")]),
    )

    runner = CliRunner()
    listed = runner.invoke(app, ["session", "list", "--repo", str(tmp_path)])
    shown = runner.invoke(
        app,
        ["session", "show", "session-1", "--repo", str(tmp_path)],
    )
    with_message = runner.invoke(
        app,
        [
            "session",
            "show",
            "session-1",
            "--repo",
            str(tmp_path),
            "--messages",
            "1",
        ],
    )

    assert listed.exit_code == 0
    assert "session-1" in listed.output
    assert shown.exit_code == 0
    assert "private goal" not in shown.output
    assert "private goal" in with_message.output


def test_non_tty_run_requires_explicit_task(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["run", "--repo", str(tmp_path), "--once"])

    assert result.exit_code != 0
    assert "task is required when stdin is not a TTY" in result.output


def test_local_cli_has_no_bot_deploy_or_architecture_commands() -> None:
    runner = CliRunner()

    for command in ("bot", "deploy", "architecture"):
        result = runner.invoke(app, [command])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_conversation_driver_reuses_session_for_follow_up(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store = _store_with_session(tmp_path / "sessions")
    prompts = iter(["follow up", "/exit"])
    monkeypatch.setattr("osc_agent.cli.agent.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "osc_agent.cli.agent.typer.prompt",
        lambda *_args, **_kwargs: next(prompts),
    )
    resumed = []

    class Conversation:
        def resume(self, input=None):
            resumed.append(input)

            async def events() -> AsyncIterator[RuntimeEvent]:
                yield RunCompleted(transition=Complete(reason="end_turn"))

            return events()

        def snapshot(self):
            return store.load("session-1")

    conversation = Conversation()
    run_conversation(
        conversation=conversation,  # type: ignore[arg-type]
        repository_root=tmp_path,
        initial_events=conversation.resume(),
        once=False,
        quiet=True,
    )

    assert len(resumed) == 2
    assert resumed[0] is None
    assert resumed[1].text == "follow up"


def test_resume_latest_resolves_repository_scoped_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state))
    from osc_agent.cli.sessions import session_store

    store = session_store(tmp_path)
    store.create(
        SessionMetadata(
            schema_version=6,
            session_id="latest-session",
            workspace_root=str(tmp_path),
            model="test",
        ),
        agent_run_state(str(tmp_path)),
    )
    captured = {}

    class AgentApplication:
        def open_session(self, session_id):
            captured["opened_session_id"] = session_id
            return Conversation()

    class Conversation:
        def resume(self, input=None):
            captured["resume_input"] = input
            return _completed_events()

    async def _completed_events():
        yield RunCompleted(transition=Complete(reason="done"))

    monkeypatch.setattr(
        "osc_agent.cli.app.build_cli_application",
        lambda *_args, **_kwargs: AgentApplication(),
    )
    monkeypatch.setattr(
        "osc_agent.cli.app.run_conversation",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        ["resume", "--repo", str(tmp_path), "--latest", "--once"],
    )

    assert result.exit_code == 0
    assert captured["opened_session_id"] == "latest-session"
    assert captured["resume_input"] is None
    assert "initial_events" in captured


def test_explicit_resume_never_starts_an_unknown_session(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class Conversation:
        def start(self, _input):
            calls.append("start")
            raise AssertionError("resume command must not start a Session")

        def resume(self, _input=None):
            calls.append("resume")

            async def events():
                raise ValueError("unknown session: missing")
                yield  # pragma: no cover

            return events()

        def snapshot(self):
            return None

    class AgentApplication:
        def open_session(self, _session_id):
            return Conversation()

    monkeypatch.setattr(
        "osc_agent.cli.app.build_cli_application",
        lambda *_args, **_kwargs: AgentApplication(),
    )

    result = CliRunner().invoke(
        app,
        ["resume", "--repo", str(tmp_path), "missing", "--prompt", "continue", "--once"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "unknown session" in str(result.exception)
    assert calls == ["resume"]
