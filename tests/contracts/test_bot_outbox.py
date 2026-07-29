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
from osc_agent.bot.domain.repositories import RepositoryBotCatalog


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
        return SimpleNamespace(job_id="1" * 36)

    def complete_outbox(self, _event_id):
        self.completed = True

    def fail_outbox(self, _event_id, _error):
        self.failed = True

    def dead_letter_outbox(self, _event_id, _error):
        self.dead = True


def _processor(store: Store, preparer) -> OutboxProcessor:
    return OutboxProcessor(
        store=store, github=object(), preparer=preparer, publisher=object(),
        catalog=RepositoryBotCatalog(repositories={}),
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
