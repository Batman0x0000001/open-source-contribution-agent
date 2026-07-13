from __future__ import annotations

import json
from unittest.mock import patch

from osc_agent.workflows.contribution.gates import gate_design, gate_discover, gate_implementation


# ---------------------------------------------------------------------------
# gate_discover
# ---------------------------------------------------------------------------


def test_gate_discover_passes_with_valid_directions(tmp_path):
    data = {
        "top_directions": [
            {
                "name": "Add retry logic",
                "description": "Implement retry on transient failures",
                "source": "issue #12",
            }
        ],
        "candidate_issues": [{"number": 12}],
        "repository_profile": {"supported": True},
    }
    (tmp_path / "01_discover.json").write_text(json.dumps(data), encoding="utf-8")

    result = gate_discover(tmp_path)

    assert result.passed is True
    assert not result.warnings


def test_gate_discover_fails_with_empty_directions(tmp_path):
    data = {"top_directions": []}
    (tmp_path / "01_discover.json").write_text(json.dumps(data), encoding="utf-8")

    result = gate_discover(tmp_path)

    assert result.passed is False
    assert "empty" in result.reason.lower()


def test_gate_discover_fails_with_missing_artifact(tmp_path):
    result = gate_discover(tmp_path)

    assert result.passed is False
    assert "not found" in result.reason


# ---------------------------------------------------------------------------
# gate_design
# ---------------------------------------------------------------------------


def test_gate_design_passes_with_valid_design(tmp_path):
    data = {
        "options": [{"id": "A", "summary": "Use built-in http"}],
        "recommended": "A",
        "selected_direction": "Add retry logic",
        "agent_design": {"steps": ["step1"]},
        "allowed_files": ["src/retry.py"],
        "allowed_new_dirs": ["tests"],
        "forbidden_paths": [".github/**"],
        "source_evidence": [
            {"file": "src/retry.py", "line_range": [1, 2], "content_hash": "abc"}
        ],
        "acceptance_checks": [{"criterion": "tests pass", "command": "pytest", "manual_check": False}],
        "max_changed_files": 5,
        "max_diff_lines": 400,
        "validation": {"ok": True, "missing_files": []},
    }
    (tmp_path / "02_design.json").write_text(json.dumps(data), encoding="utf-8")

    result = gate_design(tmp_path)

    assert result.passed is True
    assert not result.warnings


def test_gate_design_fails_with_no_options(tmp_path):
    data = {
        "options": [],
        "recommended": "A",
        "selected_direction": "Add retry logic",
    }
    (tmp_path / "02_design.json").write_text(json.dumps(data), encoding="utf-8")

    result = gate_design(tmp_path)

    assert result.passed is False
    assert "options" in result.reason.lower()


# ---------------------------------------------------------------------------
# gate_implementation
# ---------------------------------------------------------------------------


def test_gate_implementation_passes_with_report(tmp_path):
    report = tmp_path / "03_implementation.json"
    report.write_text(json.dumps({
        "scope_validation": {"ok": True},
        "verification_results": [{"command": "pytest", "exit_code": 0}],
    }), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"base_commit_sha": "abc"}), encoding="utf-8")

    with patch(
        "osc_agent.workflows.contribution.gates.git_head", return_value="abc"
    ):
        result = gate_implementation(tmp_path, repo_root=tmp_path)

    assert result.passed is True


def test_gate_implementation_blocks_without_test_or_waiver(tmp_path):
    report = tmp_path / "03_implementation.json"
    report.write_text(json.dumps({
        "scope_validation": {"ok": True},
        "verification_results": [],
    }), encoding="utf-8")
    (tmp_path / "run.json").write_text(json.dumps({"base_commit_sha": "abc"}), encoding="utf-8")

    with patch(
        "osc_agent.workflows.contribution.gates.git_head", return_value="abc"
    ):
        result = gate_implementation(tmp_path, repo_root=tmp_path)

    assert result.passed is False
    assert "waiver" in result.reason
