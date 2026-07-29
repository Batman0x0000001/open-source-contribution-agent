"""清理过期 Bot 作业、工作区和相关运行状态。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import shutil

from osc_agent.bot.config import BotSettings
from osc_agent.bot.store import BotStore
from osc_agent.application.state_paths import ApplicationStatePaths


def cleanup_bot_state(settings: BotSettings) -> tuple[int, int]:
    """Remove expired terminal workspaces, then prune their durable records."""

    store = BotStore(settings.database_path)
    store.initialize()
    now = datetime.now(timezone.utc)
    workspace_cutoff = now - timedelta(hours=settings.completed_workspace_hours)
    audit_cutoff = now - timedelta(days=settings.audit_retention_days)
    root = settings.workspace_root.resolve()
    removed_workspaces = 0
    for job in store.terminal_jobs_before(workspace_cutoff):
        job_root = (root / job.job_id).resolve()
        if not job_root.is_relative_to(root) or job_root == root:
            raise ValueError("bot cleanup target escapes the configured workspace root")
        if job_root.exists():
            if job_root.is_symlink():
                raise ValueError("bot cleanup refuses a symlinked job workspace")
            shutil.rmtree(job_root)
            removed_workspaces += 1
        runtime_root = root.parent / "runtime-state"
        for phase in ("plan", "implementation"):
            state = ApplicationStatePaths.for_repository(
                job_root / phase,
                state_root=runtime_root,
            ).repository
            if state.exists():
                if not state.resolve().is_relative_to(runtime_root.resolve()) or state.is_symlink():
                    raise ValueError("bot cleanup refuses an unsafe runtime-state path")
                shutil.rmtree(state)
    removed_records = 0
    for job in store.terminal_jobs_before(audit_cutoff):
        store.delete_terminal_job(job.job_id, expected_version=job.version)
        removed_records += 1
    store.delete_deliveries_before(audit_cutoff)
    return removed_workspaces, removed_records
