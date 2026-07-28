from __future__ import annotations

import asyncio
from dataclasses import dataclass
import signal
from typing import Literal

from osc_agent.bot.config import BotSettings, BotWorkerSettings, load_repository_catalog
from osc_agent.bot.control import BotControlService
from osc_agent.bot.github_app import GitHubAppClient
from osc_agent.bot.outbox import DispatcherHealth, OutboxDispatcher, OutboxProcessor
from osc_agent.bot.publisher import TrustedPublisher
from osc_agent.bot.store import BotStore
from osc_agent.bot.webhook import create_webhook_app
from osc_agent.bot.worker import BotWorker
from osc_agent.bot.workspace import WorkspacePreparer
from osc_agent.config import load_settings


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
    preparer = WorkspacePreparer(root=settings.workspace_root, store=store, github=github)
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


async def _worker_slot(
    worker: BotWorker,
    phase: Literal["plan", "implementation"],
    shutdown: asyncio.Event,
) -> None:
    while not shutdown.is_set():
        worked = await worker.run_once(phase)
        if not worked:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=1)
            except TimeoutError:
                pass


async def run_worker_forever(
    settings: BotWorkerSettings,
    shutdown: asyncio.Event | None = None,
) -> None:
    store = BotStore(settings.database_path)
    store.initialize()
    worker = BotWorker(
        settings=load_settings(),
        bot_settings=settings,
        catalog=load_repository_catalog(settings.repositories_config),
        store=store,
    )
    own_shutdown = shutdown is None
    shutdown = shutdown or asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    if own_shutdown:
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, shutdown.set)
                installed_signals.append(signum)
            except (NotImplementedError, RuntimeError):
                # 生产环境是 Linux；显式 shutdown 参数仍让非 POSIX 契约测试可控。
                pass
    slots = [
        *(
            asyncio.create_task(
                _worker_slot(worker, "plan", shutdown),
                name=f"worker-plan-{index}",
            )
            for index in range(settings.max_concurrent_plans)
        ),
        *(
            asyncio.create_task(
                _worker_slot(worker, "implementation", shutdown),
                name=f"worker-implementation-{index}",
            )
            for index in range(settings.max_concurrent_implementations)
        ),
    ]
    shutdown_task = asyncio.create_task(shutdown.wait(), name="worker-shutdown")
    try:
        done, _pending = await asyncio.wait(
            {*slots, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if shutdown_task not in done:
            failed = next((task for task in done if task.exception() is not None), None)
            if failed is not None:
                raise failed.exception()  # type: ignore[misc]
            raise RuntimeError("worker slot exited unexpectedly")
    finally:
        shutdown.set()
        shutdown_task.cancel()
        await asyncio.gather(shutdown_task, return_exceptions=True)
        try:
            await asyncio.wait_for(
                asyncio.gather(*slots, return_exceptions=True),
                timeout=settings.shutdown_timeout_seconds,
            )
        except TimeoutError:
            for task in slots:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*slots, return_exceptions=True)
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
