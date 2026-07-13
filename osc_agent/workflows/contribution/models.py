from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osc_agent.harness.contracts import StageStatus

STAGES = {"discover", "design", "implement", "draft_pr"}
STATE_SCHEMA_VERSION = 2
DEFAULT_FORBIDDEN_PATHS = [
    ".git/**",
    ".github/**",
    ".env*",
    "*lock*",
    "**/security/**",
]


@dataclass
class ContributionRun:
    run_id: str
    repo_root: str
    repo_url: str
    stage: str
    selected_direction: str | None
    artifacts_dir: str
    schema_version: int = STATE_SCHEMA_VERSION
    base_commit_sha: str = ""
    issue_snapshot_at: str = ""
    config_snapshot: dict[str, Any] | None = None
    stage_status: dict[str, str] | None = None
    stage_hashes: dict[str, str] | None = None
    critical_file_hashes: dict[str, str] | None = None
    final_status: str | None = None
    metrics: dict[str, Any] | None = None
    worktree_root: str | None = None
    implementation_checkpoint: dict[str, Any] | None = None
    last_transition: dict[str, Any] | None = None
    recovery_stage: str | None = None
