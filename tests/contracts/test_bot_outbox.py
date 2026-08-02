"""验证Bot Outbox 投递的契约、边界条件与回归行为。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from osc_agent.bot.control.outbox import (
    OutboxProcessor,
    RetryableOutboxError,
    TerminalOutboxError,
)
from osc_agent.bot.domain.events import OutboxEvent
from osc_agent.bot.domain.repositories import RepositoryBotCatalog, RepositoryBotConfig


def _event() -> OutboxEvent:
    return OutboxEvent(
        event_id="event", job_id="1" * 36, kind="prepare",
        idempotency_key="prepare:test", payload={"phase": "plan"},
    )


class Store:
    def __init__(self) -> None:
        self.event = _event()
        self.completed = False
        self.failed = False
        self.dead = False

    def next_outbox(self):
        event, self.event = self.event, None
        return event

    def get_job(self, _job_id):
        return SimpleNamespace(
            job_id="1" * 36,
            status="ready_to_publish",
            pull_request_url=None,
            installation_id=1,
            repository_full_name="owner/repo",
            issue_number=2,
        )

    def complete_outbox(self, _event_id):
        self.completed = True

    def fail_outbox(self, _event_id, _error):
        self.failed = True

    def dead_letter_outbox(self, _event_id, _error):
        self.dead = True


def _processor(
    store: Store,
    preparer,
    *,
    github=object(),
    publisher=object(),
    catalog: RepositoryBotCatalog | None = None,
) -> OutboxProcessor:
    default_catalog = RepositoryBotCatalog(
        repositories={
            "owner/repo": RepositoryBotConfig(
                image="sha256:" + "a" * 64,
                validation_commands=("python -m pytest",),
            )
        }
    )
    return OutboxProcessor(
        store=store,
        github=github,
        preparer=preparer,
        publisher=publisher,
        catalog=catalog or default_catalog,
    )


def test_retryable_outbox_error_is_contained() -> None:
    class Preparer:
        async def prepare(self, _job, _phase):
            raise RetryableOutboxError("temporary")

    store = Store()
    assert asyncio.run(_processor(store, Preparer()).run_once()) is True
    assert store.failed is True and store.dead is False


def test_terminal_outbox_error_is_dead_lettered() -> None:
    class Preparer:
        async def prepare(self, _job, _phase):
            raise TerminalOutboxError("permanent")

    store = Store()
    assert asyncio.run(_processor(store, Preparer()).run_once()) is True
    assert store.dead is True and store.failed is False


@pytest.mark.parametrize("kind", ["prepare", "publish"])
def test_disabled_repository_revokes_workspace_and_publish_events(kind: str) -> None:
    class SideEffect:
        async def prepare(self, _job, _phase):
            raise AssertionError("disabled repository must not prepare a workspace")

        async def publish(self, _job):
            raise AssertionError("disabled repository must not publish")

    store = Store()
    store.event = _event().model_copy(update={"kind": kind})
    catalog = RepositoryBotCatalog(
        repositories={
            "owner/repo": RepositoryBotConfig(
                enabled=False,
                image="sha256:" + "a" * 64,
                validation_commands=("python -m pytest",),
            )
        }
    )

    assert asyncio.run(
        _processor(
            store,
            SideEffect(),
            publisher=SideEffect(),
            catalog=catalog,
        ).run_once()
    ) is True
    assert store.dead is True and store.completed is False


def test_unclassified_handler_and_failure_record_errors_propagate() -> None:
    class Preparer:
        async def prepare(self, _job, _phase):
            raise RuntimeError("unexpected")

    store = Store()
    with pytest.raises(RuntimeError, match="unexpected"):
        asyncio.run(_processor(store, Preparer()).run_once())
    assert store.failed is True

    class BrokenStore(Store):
        def fail_outbox(self, _event_id, _error):
            raise OSError("fail_outbox unavailable")

    with pytest.raises(OSError, match="fail_outbox unavailable"):
        asyncio.run(_processor(BrokenStore(), Preparer()).run_once())


def test_complete_outbox_error_propagates() -> None:
    class Preparer:
        async def prepare(self, _job, _phase):
            return None

    class BrokenStore(Store):
        def complete_outbox(self, _event_id):
            raise OSError("complete_outbox unavailable")

    with pytest.raises(OSError, match="complete_outbox unavailable"):
        asyncio.run(_processor(BrokenStore(), Preparer()).run_once())


def test_publish_outbox_is_explicitly_routed() -> None:
    class Publisher:
        called = False

        async def publish(self, _job):
            self.called = True

    store = Store()
    store.event = _event().model_copy(update={"kind": "publish", "payload": {}})
    publisher = Publisher()
    catalog = RepositoryBotCatalog(
        repositories={
            "owner/repo": RepositoryBotConfig(
                image="sha256:" + "a" * 64,
                validation_commands=("python -m pytest",),
            )
        }
    )

    assert asyncio.run(
        _processor(store, object(), publisher=publisher, catalog=catalog).run_once()
    ) is True
    assert publisher.called is True
    assert store.completed is True


def test_issue_comment_outbox_is_explicitly_routed() -> None:
    class GitHub:
        body: str | None = None

        async def find_issue_comment(self, *_args):
            return None

        async def create_issue_comment(self, *_args):
            self.body = _args[-1]

    store = Store()
    store.event = _event().model_copy(
        update={"kind": "issue_comment", "payload": {"body": "status"}}
    )
    github = GitHub()

    assert asyncio.run(_processor(store, object(), github=github).run_once()) is True
    assert github.body is not None and "<!-- osa-outbox:prepare:test -->" in github.body
    assert store.completed is True


def test_unknown_outbox_kind_is_dead_lettered_without_publishing() -> None:
    class Publisher:
        async def publish(self, _job):
            raise AssertionError("unknown outbox kind must not publish")

    store = Store()
    store.event = _event().model_copy(update={"kind": "future"})

    assert asyncio.run(_processor(store, object(), publisher=Publisher()).run_once()) is True
    assert store.dead is True
    assert store.completed is False
