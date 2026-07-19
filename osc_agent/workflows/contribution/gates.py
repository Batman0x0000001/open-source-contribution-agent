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
from osc_agent.workflows.contribution.models import DesignContract, ScopeContract


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quality gate check."""

    passed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    status: RunStatus | None = None
    metadata: dict[str, object] = field(default_factory=dict)


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

    try:
        ScopeContract.model_validate(data)
    except ValueError as exc:
        return GateResult(
            passed=False,
            reason=f"invalid design scope contract: {exc}",
            status=RunStatus.FAILED_VALIDATION,
        )

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
    spec = data.get("contribution_spec") or {}
    requirements = spec.get("requirements") or []
    requirement_ids = {str(item.get("id")) for item in requirements if isinstance(item, dict) and item.get("id")}
    if not requirements or len(requirement_ids) != len(requirements):
        return GateResult(
            passed=False,
            reason="contribution_spec must contain uniquely identified requirements",
            status=RunStatus.FAILED_VALIDATION,
        )
    task_type = str(spec.get("task_type") or "")
    if task_type not in {"behavior", "docs", "config"}:
        return GateResult(
            passed=False,
            reason="contribution_spec task_type must be behavior, docs, or config",
            status=RunStatus.FAILED_VALIDATION,
        )
    evidence_requirement_ids = {
        str(requirement_id)
        for evidence in data.get("source_evidence") or []
        for requirement_id in evidence.get("requirement_ids") or []
    }
    if data.get("source_evidence") and not requirement_ids <= evidence_requirement_ids:
        return GateResult(
            passed=False,
            reason="source evidence does not cover every requirement",
            status=RunStatus.FAILED_VALIDATION,
        )
    if task_type == "behavior":
        if not any(evidence.get("symbol") for evidence in data.get("source_evidence") or []):
            return GateResult(
                passed=False,
                reason="behavior changes require symbol-level source evidence",
                status=RunStatus.FAILED_VALIDATION,
            )
        baseline_checks = spec.get("baseline_checks") or []
        reproduction = spec.get("reproduction") or {}
        reproduction_mode = str(reproduction.get("mode") or ("existing" if baseline_checks else ""))
        if reproduction_mode not in {"existing", "generated_test"}:
            return GateResult(
                passed=False,
                reason="behavior changes require a failure baseline or a generated regression test plan",
                status=RunStatus.FAILED_VALIDATION,
            )
        if reproduction_mode == "existing":
            if not baseline_checks:
                return GateResult(
                    passed=False,
                    reason="behavior changes require a reproducible failure baseline",
                    status=RunStatus.FAILED_VALIDATION,
                )
            for index, check in enumerate(baseline_checks):
                if (
                    not check.get("command")
                    or not check.get("expected_exit_codes")
                    or not str(check.get("output_contains") or "").strip()
                ):
                    return GateResult(
                        passed=False,
                        reason=f"baseline_checks[{index}] must define command, expected exit codes, and output",
                        status=RunStatus.FAILED_VALIDATION,
                    )
        else:
            command = str(reproduction.get("command") or "")
            test_files = [str(path).replace("\\", "/") for path in reproduction.get("test_files") or []]
            allowed_files = {str(path).replace("\\", "/") for path in data.get("allowed_files") or []}
            allowed_dirs = [str(path).strip("/\\").replace("\\", "/") for path in data.get("allowed_new_dirs") or []]
            invalid_test_files = [
                path for path in test_files
                if not _is_test_path(path) or (
                    path not in allowed_files
                    and not any(path.startswith(f"{directory}/") for directory in allowed_dirs)
                )
            ]
            command_tokens = command.casefold().replace('"', "").replace("'", "").split()
            command_names_declared_test = any(path.casefold() in command.casefold() for path in test_files)
            if (
                not command
                or "pytest" not in command_tokens
                or not command_names_declared_test
                or not test_files
                or invalid_test_files
            ):
                return GateResult(
                    passed=False,
                    reason="generated regression plan requires an acceptance command and test-only files in approved scope",
                    status=RunStatus.FAILED_VALIDATION,
                )
        acceptance_commands = {
            str(check.get("command") or "") for check in data.get("acceptance_checks") or [] if check.get("command")
        }
        reproduction_commands = (
            [str(check.get("command")) for check in baseline_checks]
            if reproduction_mode == "existing"
            else [str(reproduction.get("command") or "")]
        )
        missing_post_checks = [command for command in reproduction_commands if command not in acceptance_commands]
        if missing_post_checks:
            return GateResult(
                passed=False,
                reason=f"failure baseline must be rerun after editing: {missing_post_checks[0]}",
                status=RunStatus.FAILED_VALIDATION,
            )
    validation = data.get("validation") or {}
    if not validation.get("ok", False):
        invalid_paths = validation.get("invalid_paths", [])
        if invalid_paths:
            return GateResult(
                passed=False,
                reason=f"design references paths outside the repository: {invalid_paths}",
                status=RunStatus.FAILED_VALIDATION,
            )
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
        mapped = {str(item) for item in check.get("requirement_ids") or []}
        if not mapped or not mapped <= requirement_ids:
            return GateResult(
                passed=False,
                reason=f"acceptance_checks[{index}] has invalid requirement mapping",
                status=RunStatus.FAILED_VALIDATION,
            )
        if task_type == "behavior" and check.get("manual_check"):
            return GateResult(
                passed=False,
                reason="behavior requirements cannot use manual-only acceptance checks",
                status=RunStatus.FAILED_VALIDATION,
            )
    covered_requirements = {
        str(requirement_id)
        for check in data.get("acceptance_checks") or []
        for requirement_id in check.get("requirement_ids") or []
    }
    if not requirement_ids <= covered_requirements:
        return GateResult(
            passed=False,
            reason="acceptance checks do not cover every requirement",
            status=RunStatus.FAILED_VALIDATION,
        )

    try:
        DesignContract.model_validate(data)
    except ValueError as exc:
        return GateResult(
            passed=False,
            reason=f"design contract is invalid: {exc}",
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
    spec = data.get("contribution_spec") or {}
    if spec.get("task_type") == "behavior":
        baseline_results = data.get("baseline_results") or []
        if not baseline_results or not all(item.get("expected_failure_matched") for item in baseline_results):
            return GateResult(
                passed=False,
                reason="behavior change has no verified pre-change failure baseline",
                status=RunStatus.FAILED_VALIDATION,
            )
        reproduction = spec.get("reproduction") or {}
        if reproduction.get("mode") == "generated_test" and not (
            data.get("reproduction_validation") or {}
        ).get("ok"):
            return GateResult(
                passed=False,
                reason="generated regression test was not preserved after implementation",
                status=RunStatus.FAILED_VALIDATION,
            )
        if reproduction.get("mode") == "generated_test" and not (
            (data.get("reproduction_evidence") or {}).get("semantic_binding") or {}
        ).get("ok"):
            return GateResult(
                passed=False,
                reason="generated regression test has no valid Issue-to-target semantic binding",
                status=RunStatus.FAILED_VALIDATION,
            )
    uncovered = [
        str(item.get("requirement_id"))
        for item in data.get("requirement_coverage") or []
        if not item.get("passed")
    ]
    if not data.get("requirement_coverage") or uncovered:
        return GateResult(
            passed=False,
            reason=f"requirements not verified: {', '.join(uncovered) or 'coverage missing'}",
            status=RunStatus.FAILED_VALIDATION,
        )

    return GateResult(
        passed=True,
        reason="implementation scope and verification valid",
        status=RunStatus.SUCCESS,
    )


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = normalized.rsplit("/", 1)[-1]
    return (
        name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
        and any(part in {"test", "tests"} for part in parts[:-1])
    )
