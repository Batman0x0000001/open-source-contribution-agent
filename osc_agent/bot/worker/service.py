"""启动并监管 Bot Worker 的 Plan 与 Implementation 槽位。"""

from __future__ import annotations

import asyncio
import signal
from typing import Literal

from osc_agent.bot.config import BotWorkerSettings, load_repository_catalog
from osc_agent.bot.persistence.store import BotStore
from osc_agent.bot.worker.coordinator import BotWorker
from osc_agent.configuration import load_agent_settings


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
        settings=load_agent_settings(),
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
