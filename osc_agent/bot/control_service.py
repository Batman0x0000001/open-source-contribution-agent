"""启动并监管 Bot Control 与 Outbox 常驻服务。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from osc_agent.bot.config import BotSettings, load_repository_catalog
from osc_agent.bot.control import BotControlService
from osc_agent.bot.github_app import GitHubAppClient
from osc_agent.bot.outbox import DispatcherHealth, OutboxDispatcher, OutboxProcessor
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.store import BotStore
from osc_agent.bot.webhook import create_webhook_app
from osc_agent.bot.job_workspace import BotJobWorkspacePreparer


def build_control_components(settings: BotSettings):
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
    return app, dispatcher


def build_control_app(settings: BotSettings):
    """Compatibility helper for ASGI inspection; production uses run_control_forever."""
    return build_control_components(settings)[0]


@dataclass
class ControlSupervisor:
    settings: BotSettings

    async def run(self) -> None:
        import uvicorn

        app, dispatcher = build_control_components(self.settings)
        shutdown = asyncio.Event()
        server = uvicorn.Server(uvicorn.Config(
            app, host=self.settings.bind_host, port=self.settings.bind_port, log_config=None
        ))
        web_task = asyncio.create_task(server.serve(), name="control-http")
        dispatcher_task = asyncio.create_task(
            dispatcher.run_forever(shutdown), name="control-outbox"
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


async def run_control_forever(settings: BotSettings) -> None:
    await ControlSupervisor(settings).run()

