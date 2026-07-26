from __future__ import annotations

from osc_agent.bot.github_app import GitHubControlClient
from osc_agent.bot.models import RepositoryBotCatalog
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.store import BotStore
from osc_agent.bot.workspace import WorkspacePreparer


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
                    raise ValueError("prepare outbox has invalid phase")
                await self.preparer.prepare(job, phase)
            elif event.kind == "issue_comment":
                body = event.payload.get("body")
                if not isinstance(body, str):
                    raise ValueError("issue comment outbox has no body")
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
                    raise ValueError("repository configuration disappeared before publish")
                await self.publisher.publish(job, config)
        except Exception as exc:
            self.store.fail_outbox(event.event_id, str(exc))
            return True
        self.store.complete_outbox(event.event_id)
        return True
