from __future__ import annotations

import asyncio

from osc_agent.bot.config import BotSettings, BotWorkerSettings, load_repository_catalog
from osc_agent.bot.control import BotControlService
from osc_agent.bot.github_app import GitHubAppClient
from osc_agent.bot.outbox import OutboxProcessor
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.sandbox import resolve_image_id
from osc_agent.bot.store import BotStore
from osc_agent.bot.webhook import create_webhook_app
from osc_agent.bot.worker import BotWorker
from osc_agent.bot.workspace import WorkspacePreparer
from osc_agent.config import load_settings


def build_control_app(settings: BotSettings):
    catalog = load_repository_catalog(settings.repositories_config)
    store = BotStore(settings.database_path)
    store.initialize()
    github = GitHubAppClient(settings)
    control = BotControlService(
        settings=settings,
        catalog=catalog,
        store=store,
        github=github,
        image_resolver=resolve_image_id,
    )
    preparer = WorkspacePreparer(root=settings.workspace_root, store=store, github=github)
    publisher = TrustedPublisher(settings=settings, store=store, github=github)
    outbox = OutboxProcessor(
        store=store,
        github=github,
        preparer=preparer,
        publisher=publisher,
        catalog=catalog,
    )
    return create_webhook_app(
        secret=settings.github_webhook_secret,
        store=store,
        control=control,
        outbox_processor=outbox,
    )


async def run_worker_forever(settings: BotWorkerSettings) -> None:
    store = BotStore(settings.database_path)
    store.initialize()
    worker = BotWorker(
        settings=load_settings(),
        bot_settings=settings,
        catalog=load_repository_catalog(settings.repositories_config),
        store=store,
    )
    active: set[asyncio.Task[bool]] = set()
    while True:
        while len(active) < settings.max_concurrent_jobs:
            active.add(asyncio.create_task(worker.run_once()))
        done, active = await asyncio.wait(
            active,
            return_when=asyncio.FIRST_COMPLETED,
        )
        worked = [task.result() for task in done]
        if worked and not any(worked) and not active:
            await asyncio.sleep(1)
