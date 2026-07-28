from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic

from osc_agent.bot.github_app import GitHubApiError, GitHubControlClient
from osc_agent.bot.models import RepositoryBotCatalog
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.store import BotStore
from osc_agent.bot.workspace import WorkspacePreparer
from osc_agent.bot.observability import OUTBOX_HEARTBEAT, OUTBOX_ITERATIONS, OUTBOX_UP


class RetryableOutboxError(Exception):
    pass


class TerminalOutboxError(Exception):
    pass


@dataclass
class DispatcherHealth:
    running: bool = False
    last_heartbeat: float = 0.0
    last_error: str | None = None

    @property
    def ready(self) -> bool:
        age = monotonic() - self.last_heartbeat if self.last_heartbeat else float("inf")
        if OUTBOX_HEARTBEAT is not None:
            OUTBOX_HEARTBEAT.set(age)
        return self.running and age < 10


class OutboxProcessor:
    def __init__(self, *, store: BotStore, github: GitHubControlClient, preparer: WorkspacePreparer, publisher: TrustedPublisher, catalog: RepositoryBotCatalog) -> None:
        self.store = store
        self.github = github
        self.preparer = preparer
        self.publisher = publisher
        self.catalog = catalog

    async def run_once(self) -> bool:
        event = self.store.next_outbox()
        if event is None:
            return False
        job = self.store.get_job(event.job_id)
        if job is None:
            self.store.fail_outbox(event.event_id, "job no longer exists")
            return True
        try:
            if event.kind == "prepare":
                phase = event.payload.get("phase")
                if phase not in {"plan", "implementation"}:
                    raise TerminalOutboxError("prepare outbox has invalid phase")
                await self.preparer.prepare(job, phase)
            elif event.kind == "issue_comment":
                body = event.payload.get("body")
                if not isinstance(body, str):
                    raise TerminalOutboxError("issue comment outbox has no body")
                marker = f"<!-- osa-outbox:{event.idempotency_key} -->"
                rendered = body if marker in body else f"{body}\n\n{marker}"
                existing = await self.github.find_issue_comment(
                    job.installation_id,
                    job.repository_full_name,
                    job.issue_number,
                    marker,
                )
                if existing is None:
                    await self.github.create_issue_comment(
                        job.installation_id,
                        job.repository_full_name,
                        job.issue_number,
                        rendered,
                    )
            else:
                if job.status == "completed" and job.pull_request_url is not None:
                    self.store.complete_outbox(event.event_id)
                    return True
                config = self.catalog.repositories.get(job.repository_full_name)
                if config is None:
                    raise TerminalOutboxError("repository configuration disappeared before publish")
                await self.publisher.publish(job, config)
        except TerminalOutboxError as exc:
            self.store.dead_letter_outbox(event.event_id, str(exc))
            return True
        except RetryableOutboxError as exc:
            self.store.fail_outbox(event.event_id, str(exc))
            return True
        except GitHubApiError as exc:
            if exc.retryable:
                self.store.fail_outbox(event.event_id, str(exc))
            else:
                self.store.dead_letter_outbox(event.event_id, str(exc))
            return True
        except Exception as exc:
            try:
                self.store.fail_outbox(event.event_id, str(exc))
            finally:
                raise
        self.store.complete_outbox(event.event_id)
        return True


class OutboxDispatcher:
    def __init__(self, processor: OutboxProcessor, health: DispatcherHealth | None = None) -> None:
        self.processor = processor
        self.health = health or DispatcherHealth()

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        self.health.running = True
        if OUTBOX_UP is not None:
            OUTBOX_UP.set(1)
        self.health.last_error = None
        try:
            while not shutdown.is_set():
                self.health.last_heartbeat = monotonic()
                if OUTBOX_HEARTBEAT is not None:
                    OUTBOX_HEARTBEAT.set(0)
                worked = await self.processor.run_once()
                if OUTBOX_ITERATIONS is not None:
                    OUTBOX_ITERATIONS.labels(outcome="worked" if worked else "idle").inc()
                self.health.last_heartbeat = monotonic()
                if not worked:
                    try:
                        await asyncio.wait_for(shutdown.wait(), timeout=1)
                    except TimeoutError:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.health.last_error = str(exc)[:500]
            raise
        finally:
            self.health.running = False
            if OUTBOX_UP is not None:
                OUTBOX_UP.set(0)
