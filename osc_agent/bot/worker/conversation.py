"""消费 Agent Runtime 事件并维护 Worker Lease 与脱敏进度。"""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Literal

from osc_agent.bot.persistence.store import BotStore
from osc_agent.runtime.events import RunCompleted, RunStopped


async def consume_runtime_events(
    events,
    *,
    store: BotStore,
    worker_id: str,
    job_id: str,
    phase: Literal["plan", "implementation"],
) -> bool:
    completed = False
    last_heartbeat = monotonic()
    last_progress = 0.0
    iterator = events.__aiter__()
    pending = asyncio.create_task(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=5)
            current = store.get_job(job_id)
            if current is None or current.status == "cancelled":
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                raise ValueError("bot job was cancelled while the Agent was running")
            if monotonic() - last_heartbeat >= 20:
                store.heartbeat(job_id, worker_id)
                last_heartbeat = monotonic()
            if not done:
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            event_type = event.type
            now = monotonic()
            if last_progress == 0.0 or now - last_progress >= 5 or isinstance(event, (RunCompleted, RunStopped)):
                store.record_progress(job_id, f"{phase}:{event_type}")
                last_progress = now
            if isinstance(event, RunCompleted):
                completed = True
            elif isinstance(event, RunStopped):
                completed = False
            pending = asyncio.create_task(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    return completed
