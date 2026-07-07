from __future__ import annotations

import json
from unittest.mock import patch

from osc_agent.harness.gates import gate_design, gate_discover, gate_implementation


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
    report = tmp_path / "03_implementation_report.md"
    report.write_text(
        "## Changes\nAdded test for retry logic.\n\npytest passed.\n",
        encoding="utf-8",
    )

    with patch(
        "osc_agent.harness.gates.git_status", return_value=" M src/retry.py\n"
    ):
        result = gate_implementation(tmp_path, repo_root=tmp_path)

    assert result.passed is True


def test_gate_implementation_warns_on_no_test_evidence(tmp_path):
    report = tmp_path / "03_implementation_report.md"
    report.write_text(
        "## Changes\nRefactored the logging module.\n",
        encoding="utf-8",
    )

    with patch(
        "osc_agent.harness.gates.git_status", return_value=" M src/logging.py\n"
    ):
        result = gate_implementation(tmp_path, repo_root=tmp_path)

    assert result.passed is True
    assert any("test evidence" in w.lower() for w in result.warnings)
