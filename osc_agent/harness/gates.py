"""
阶段产物生成完成
    ↓
读取 artifact JSON / MD
    ↓
检查结构完整性和内容质量
    ↓
返回 GateResult（pass / fail + 原因）
    ↓
调用方根据结果决定是否继续
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from osc_agent.harness.contracts import RunStatus
from osc_agent.tools.git import git_head


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quality gate check."""

    passed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    status: RunStatus | None = None


# ---------------------------------------------------------------------------
# Gate: discover → design
# ---------------------------------------------------------------------------


def gate_discover(artifacts_dir: Path) -> GateResult:
    """Validate that the discover stage produced usable output."""
    artifact = artifacts_dir / "01_discover.json"
    if not artifact.exists():
        return GateResult(passed=False, reason="01_discover.json not found")

    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(passed=False, reason=f"Failed to parse 01_discover.json: {exc}")

    directions = data.get("top_directions", [])
    if not directions:
        return GateResult(passed=False, reason="top_directions is empty or missing")

    for idx, d in enumerate(directions):
        for key in ("name", "description", "source"):
            if not d.get(key):
                return GateResult(
                    passed=False,
                    reason=f"top_directions[{idx}] has empty or missing '{key}'",
                )

    profile = data.get("repository_profile") or {}
    if not profile.get("supported"):
        return GateResult(
            passed=False,
            reason="repository is not recognized as an Agent/LLM Python project",
            status=RunStatus.OUT_OF_SCOPE,
        )

    warnings: list[str] = []
    candidate_issues = data.get("candidate_issues", [])
    arch_dims = data.get("architecture_dimensions", {})
    locations = arch_dims.get("locations", []) if isinstance(arch_dims, dict) else []
    if not candidate_issues and not locations:
        warnings.append(
            "No candidate_issues and architecture_dimensions found no locations"
        )

    return GateResult(passed=True, reason="discover artifacts valid", warnings=warnings)


# ---------------------------------------------------------------------------
# Gate: design → implementation
# ---------------------------------------------------------------------------


def gate_design(artifacts_dir: Path) -> GateResult:
    """Validate that the design stage produced usable output."""
    artifact = artifacts_dir / "02_design.json"
    if not artifact.exists():
        return GateResult(passed=False, reason="02_design.json not found")

    try:
        data = json.loads(artifact.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(passed=False, reason=f"Failed to parse 02_design.json: {exc}")

    options = data.get("options", [])
    if not options:
        return GateResult(passed=False, reason="options is empty or missing")

    if not data.get("recommended"):
        return GateResult(passed=False, reason="recommended is empty or missing")

    if not data.get("selected_direction"):
        return GateResult(passed=False, reason="selected_direction is empty or missing")

    required_scope = (
        "allowed_files",
        "forbidden_paths",
        "source_evidence",
        "acceptance_checks",
        "max_changed_files",
        "max_diff_lines",
    )
    missing_scope = [key for key in required_scope if not data.get(key)]
    if missing_scope:
        return GateResult(
            passed=False,
            reason=f"design scope contract is incomplete: {', '.join(missing_scope)}",
            status=RunStatus.FAILED_VALIDATION,
        )
    for index, evidence in enumerate(data.get("source_evidence") or []):
        path = evidence.get("file")
        if not path or not evidence.get("content_hash") or not evidence.get("line_range"):
            return GateResult(
                passed=False,
                reason=f"source_evidence[{index}] is incomplete",
                status=RunStatus.FAILED_VALIDATION,
            )
    validation = data.get("validation") or {}
    if not validation.get("ok", False):
        return GateResult(
            passed=False,
            reason=f"design references missing files: {validation.get('missing_files', [])}",
            status=RunStatus.FAILED_VALIDATION,
        )
    for index, check in enumerate(data.get("acceptance_checks") or []):
        if not check.get("criterion") or (not check.get("command") and not check.get("manual_check")):
            return GateResult(
                passed=False,
                reason=f"acceptance_checks[{index}] has no executable command or manual check",
                status=RunStatus.FAILED_VALIDATION,
            )

    warnings: list[str] = []
    if not data.get("agent_design"):
        warnings.append("No agent_design present in design artifact")

    return GateResult(passed=True, reason="design artifacts valid", warnings=warnings)


# ---------------------------------------------------------------------------
# Gate: implementation → PR draft
# ---------------------------------------------------------------------------


def gate_implementation(artifacts_dir: Path, repo_root: Path) -> GateResult:
    """Validate that the implementation stage produced real changes."""
    report = artifacts_dir / "03_implementation.json"
    if not report.exists():
        return GateResult(passed=False, reason="03_implementation.json not found", status=RunStatus.FAILED_VALIDATION)

    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(passed=False, reason=f"invalid implementation artifact: {exc}", status=RunStatus.FAILED_VALIDATION)

    run_path = artifacts_dir / "run.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(passed=False, reason=f"invalid run state: {exc}", status=RunStatus.STALE_RUN)
    current_head = git_head(repo_root=repo_root).strip()
    if current_head != run.get("base_commit_sha"):
        return GateResult(
            passed=False,
            reason="repository HEAD does not match the saved base commit",
            status=RunStatus.STALE_RUN,
        )

    scope = data.get("scope_validation") or {}
    if not scope.get("ok"):
        return GateResult(
            passed=False,
            reason="; ".join(scope.get("violations") or ["implementation scope validation failed"]),
            status=RunStatus.OUT_OF_SCOPE if scope.get("outside_scope") or scope.get("forbidden_changes") else RunStatus.FAILED_VALIDATION,
        )

    verification = data.get("verification_results") or []
    failed = [item for item in verification if item.get("exit_code") != 0]
    if failed:
        return GateResult(
            passed=False,
            reason=f"verification command failed: {failed[0].get('command')}",
            status=RunStatus.FAILED_VALIDATION,
        )
    if not verification and not data.get("test_waiver"):
        return GateResult(
            passed=False,
            reason="no verification command executed and no audited test waiver recorded",
            status=RunStatus.BLOCKED_NEEDS_USER,
        )

    return GateResult(
        passed=True,
        reason="implementation scope and verification valid",
        status=RunStatus.SUCCESS,
    )
