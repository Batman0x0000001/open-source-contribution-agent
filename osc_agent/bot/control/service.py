"""启动并监管 Bot Control 与 Outbox 常驻服务。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from osc_agent.bot.config import BotControlSettings, load_repository_catalog
from osc_agent.bot.control.github import GitHubAppClient
from osc_agent.bot.control.handler import BotControlService
from osc_agent.bot.control.job_workspace import BotJobWorkspacePreparer
from osc_agent.bot.control.outbox import DispatcherHealth, OutboxDispatcher, OutboxProcessor
from osc_agent.bot.control.publisher import TrustedPublisher
from osc_agent.bot.control.webhook import create_webhook_app
from osc_agent.bot.persistence.store import BotStore


@dataclass(frozen=True)
class ControlComponents:
    app: Any
    dispatcher: OutboxDispatcher


def build_control_components(settings: BotControlSettings) -> ControlComponents:
    catalog = load_repository_catalog(settings.repositories_config)
    store = BotStore(settings.database_path)
    store.initialize()
    github = GitHubAppClient(settings)
    control = BotControlService(
        settings=settings,
        catalog=catalog,
        store=store,
        github=github,
    )
    preparer = BotJobWorkspacePreparer(
        root=settings.workspace_root,
        store=store,
        github=github,
    )
    publisher = TrustedPublisher(settings=settings, store=store, github=github)
    processor = OutboxProcessor(
        store=store,
        github=github,
        preparer=preparer,
        publisher=publisher,
        catalog=catalog,
    )
    health = DispatcherHealth()
    dispatcher = OutboxDispatcher(processor, health)
    app = create_webhook_app(
        secret=settings.github_webhook_secret,
        store=store,
        control=control,
        dispatcher_health=health,
    )
    return ControlComponents(app=app, dispatcher=dispatcher)


@dataclass
class ControlSupervisor:
    settings: BotControlSettings

    async def run(self) -> None:
        import uvicorn

        components = build_control_components(self.settings)
        shutdown = asyncio.Event()
        server = uvicorn.Server(uvicorn.Config(
            components.app,
            host=self.settings.bind_host,
            port=self.settings.bind_port,
            log_config=None,
        ))
        web_task = asyncio.create_task(server.serve(), name="control-http")
        dispatcher_task = asyncio.create_task(
            components.dispatcher.run_forever(shutdown), name="control-outbox"
        )
        tasks = {web_task, dispatcher_task}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            normal_shutdown = web_task in done and server.should_exit
            shutdown.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
            if not normal_shutdown:
                raise RuntimeError("critical Control task exited unexpectedly")
        finally:
            shutdown.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def run_control_forever(settings: BotControlSettings) -> None:
    await ControlSupervisor(settings).run()
