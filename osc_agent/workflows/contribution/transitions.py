from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.workflows.contribution.gates import GateResult, gate_design, gate_discover, gate_implementation
from osc_agent.workflows.contribution.models import ContributionRun, STAGES
from osc_agent.workflows.contribution.state import save_run

def transition_run(
    run: ContributionRun,
    stage: str,
    *,
    repo_root: Path | None = None,
    success: bool | None = None,
) -> None:
    """唯一的阶段状态转换入口；开始阶段时在这里执行前置 Gate。"""
    if stage not in STAGES:
        raise ValueError(f"unknown contribution stage: {stage}")
    if run.stage_status is None or run.metrics is None:
        raise ValueError("run state is missing required schema fields")

    if success is not None:
        if run.stage != stage or run.stage_status.get(stage) != StageStatus.RUNNING.value:
            raise ValueError(f"cannot complete stage {stage!r} because it is not running")
        run.stage_status[stage] = StageStatus.SUCCEEDED.value if success else StageStatus.FAILED.value
        record = run.metrics.setdefault("stages", {}).setdefault(stage, {})
        finished = datetime.now(timezone.utc)
        record["finished_at"] = finished.isoformat()
        try:
            started = datetime.fromisoformat(str(record["started_at"]))
            record["duration_ms"] = int((finished - started).total_seconds() * 1000)
        except (KeyError, ValueError):
            record["duration_ms"] = 0
        run.recovery_stage = None if success else stage
        run.last_transition = {
            "stage": stage,
            "status": run.stage_status[stage],
            "at": finished.isoformat(),
        }
        return

    current_status = run.stage_status.get(run.stage, StageStatus.PENDING.value)
    allowed_retry = stage == run.stage and current_status in {
        StageStatus.PENDING.value,
        StageStatus.FAILED.value,
        StageStatus.RUNNING.value,
    }
    ordered = ["discover", "design", "implement", "draft_pr"]
    allowed_forward = ordered.index(stage) == ordered.index(run.stage) + 1
    if not (allowed_retry or allowed_forward):
        raise ValueError(f"illegal contribution transition: {run.stage} -> {stage}")

    active_repo = (repo_root or Path(run.worktree_root or run.repo_root)).resolve()
    gate = _transition_gate(run, stage, active_repo)
    if not gate.passed:
        run.final_status = (gate.status or RunStatus.FAILED_VALIDATION).value
        run.recovery_stage = run.stage if allowed_forward else stage
        run.last_transition = {
            "from": run.stage,
            "to": stage,
            "status": "BLOCKED",
            "reason": gate.reason,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        save_run(run)
        raise ValueError(f"{run.final_status}: transition to {stage} blocked: {gate.reason}")

    run.stage = stage
    run.stage_status[stage] = StageStatus.RUNNING.value
    run.final_status = None
    run.recovery_stage = stage
    stages = run.metrics.setdefault("stages", {})
    stages[stage] = {"started_at": datetime.now(timezone.utc).isoformat()}
    run.last_transition = {
        "to": stage,
        "status": StageStatus.RUNNING.value,
        "at": stages[stage]["started_at"],
    }
    save_run(run)


def _begin_stage(run: ContributionRun, stage: str, repo_root: Path | None = None) -> None:
    transition_run(run, stage, repo_root=repo_root)


def _complete_stage(run: ContributionRun, stage: str, *, success: bool) -> None:
    transition_run(run, stage, success=success)


def _transition_gate(run: ContributionRun, target: str, repo_root: Path) -> GateResult:
    artifacts = Path(run.artifacts_dir)
    if target == "discover":
        return GateResult(True, "initial stage")
    if target == "design":
        return gate_discover(artifacts)
    if target == "implement":
        design_gate = gate_design(artifacts)
        if not design_gate.passed:
            return design_gate
        from osc_agent.workflows.contribution.discover import revalidate_selected_issue

        return revalidate_selected_issue(run)
    return gate_implementation(artifacts, repo_root)

