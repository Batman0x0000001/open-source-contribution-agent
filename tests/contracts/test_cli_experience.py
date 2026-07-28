"""验证命令行交互体验的契约、边界条件与回归行为。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from osc_agent.cli import app
from osc_agent.cli_session import run_conversation
from osc_agent.runtime.models import (
    Complete,
    QueryConfig,
    RunCompleted,
    RuntimeEvent,
    RuntimeMessage,
    SessionMetadata,
    SessionRuntimeState,
    TextBlock,
)
from osc_agent.runtime.session_store import FileSessionStore


def _store_with_session(root: Path, *, session_id: str = "session-1") -> FileSessionStore:
    store = FileSessionStore(root)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id=session_id,
            repository_root=str(root),
            initial_working_directory=str(root),
            model="test-model",
        )
    )
    store.append_message(
        session_id,
        RuntimeMessage(role="user", content=[TextBlock(text="private goal")]),
    )
    store.save_state(
        session_id,
        SessionRuntimeState(last_status="completed", last_reason="end_turn"),
    )
    return store


def test_session_cli_hides_messages_by_default(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state))
    from osc_agent.runtime.state_paths import ApplicationStatePaths

    store = FileSessionStore(ApplicationStatePaths.for_repository(tmp_path).sessions)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="session-1",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="test-model",
        )
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


def test_conversation_driver_reuses_session_for_follow_up(
    monkeypatch,
    tmp_path: Path,
) -> None:
    store = _store_with_session(tmp_path / "sessions")
    prompts = iter(["follow up", "/exit"])
    monkeypatch.setattr("osc_agent.cli_session.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "osc_agent.cli_session.typer.prompt",
        lambda *_args, **_kwargs: next(prompts),
    )
    resumed = []

    class AgentApplication:
        def run(self, spec):
            resumed.append(spec)
            async def events() -> AsyncIterator[RuntimeEvent]:
                yield RunCompleted(transition=Complete(reason="end_turn"))
            return events()

    async def initial() -> AsyncIterator[RuntimeEvent]:
        yield RunCompleted(transition=Complete(reason="end_turn"))

    services = SimpleNamespace(
        session_store=store,
        query_config=QueryConfig(),
    )

    run_conversation(
        services=services,
        session_id="session-1",
        repository_root=tmp_path,
        initial_events=initial(),
        once=False,
        quiet=True,
        agent_application=AgentApplication(),  # type: ignore[arg-type]
    )

    assert len(resumed) == 1
    assert resumed[0].session_id == "session-1"
    assert resumed[0].inbound_message.text == "follow up"


def test_resume_latest_resolves_repository_scoped_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("OSC_AGENT_STATE_DIR", str(state))
    from osc_agent.composition import build_session_store

    store = build_session_store(tmp_path)
    store.create(
        SessionMetadata(
            schema_version=4,
            session_id="latest-session",
            repository_root=str(tmp_path),
            initial_working_directory=str(tmp_path),
            model="test",
        )
    )
    captured = {}

    class AgentApplication:
        services = SimpleNamespace(query_config=QueryConfig())

        def run(self, spec):
            captured["spec"] = spec
            async def events():
                if False:
                    yield None
            return events()

    monkeypatch.setattr(
        "osc_agent.cli.build_agent_application", lambda **_kwargs: AgentApplication()
    )
    monkeypatch.setattr(
        "osc_agent.cli.run_conversation",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        ["resume", "--repo", str(tmp_path), "--latest", "--once"],
    )

    assert result.exit_code == 0
    assert captured["session_id"] == "latest-session"
