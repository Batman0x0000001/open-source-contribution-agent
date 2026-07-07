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

from osc_agent.tools.git import git_status


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quality gate check."""

    passed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


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

    warnings: list[str] = []
    if not data.get("agent_design"):
        warnings.append("No agent_design present in design artifact")

    return GateResult(passed=True, reason="design artifacts valid", warnings=warnings)


# ---------------------------------------------------------------------------
# Gate: implementation → PR draft
# ---------------------------------------------------------------------------


def gate_implementation(artifacts_dir: Path, repo_root: Path) -> GateResult:
    """Validate that the implementation stage produced real changes."""
    report = artifacts_dir / "03_implementation_report.md"
    if not report.exists():
        return GateResult(passed=False, reason="03_implementation_report.md not found")

    content = report.read_text(encoding="utf-8").strip()
    if not content:
        return GateResult(passed=False, reason="03_implementation_report.md is empty")

    status_output = git_status(repo_root=repo_root)
    if status_output == "(no output)" or not status_output.strip():
        return GateResult(
            passed=False,
            reason="No files changed in the working tree according to git status",
        )

    warnings: list[str] = []
    test_keywords = ("test", "Test", "TEST", "pytest", "unittest", "assert")
    if not any(kw in content for kw in test_keywords):
        warnings.append("No test evidence found in implementation report")

    return GateResult(
        passed=True, reason="implementation artifacts valid", warnings=warnings
    )
