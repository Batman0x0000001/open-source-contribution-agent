from __future__ import annotations

import json

import pytest

from osc_agent.harness.contracts import ToolResult
from osc_agent.harness.runtime_state import (
    RUNTIME_STATE_SCHEMA_VERSION,
    RuntimeState,
    load_runtime_state,
    record_tool_observation,
    refresh_runtime_state,
    runtime_state_path,
    save_runtime_state,
)


def _write_run(repo_root, run_id, *, design=None, implementation=None, hashes=None):
    run_dir = repo_root / ".osc_agent" / "contribution_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": run_id,
                "repo_root": str(repo_root.resolve()),
                "repo_url": "https://example.test/repo.git",
                "stage": "implement",
                "selected_direction": "Test contribution",
                "artifacts_dir": str(run_dir.resolve()),
                "stage_status": {
                    "discover": "SUCCEEDED",
                    "design": "SUCCEEDED",
                    "implement": "RUNNING",
                    "draft_pr": "PENDING",
                },
                "critical_file_hashes": hashes or {},
            }
        ),
        encoding="utf-8",
    )
    if design is not None:
        (run_dir / "02_design.json").write_text(json.dumps(design), encoding="utf-8")
    if implementation is not None:
        (run_dir / "03_implementation.json").write_text(json.dumps(implementation), encoding="utf-8")


def test_refresh_requires_explicit_run_binding(tmp_path):
    _write_run(
        tmp_path,
        "latest",
        design={"allowed_files": ["latest.py"]},
    )

    state = refresh_runtime_state(tmp_path, "interactive task", session_id="interactive")

    assert state.active_run_id is None
    assert state.allowed_files == []


def test_switching_runs_clears_previous_run_state(tmp_path):
    _write_run(
        tmp_path,
        "run-a",
        design={
            "out_of_scope": ["Do not edit CI"],
            "allowed_files": ["a.py"],
            "forbidden_paths": ["tests/frozen.py"],
            "acceptance_checks": [{"criterion": "A passes"}],
        },
        implementation={"verification_results": [{"command": "pytest a", "exit_code": 0}]},
        hashes={"a.py": "abc"},
    )
    _write_run(tmp_path, "run-b")

    first = refresh_runtime_state(tmp_path, "first", run_id="run-a", session_id="shared")
    record_tool_observation(
        tmp_path,
        "bash",
        {"command": "python -m pytest tests/test_a.py"},
        ToolResult(ok=True, summary="passed"),
        session_id="shared",
    )
    second = refresh_runtime_state(tmp_path, "second", run_id="run-b", session_id="shared")

    assert first.active_run_id == "run-a"
    assert first.scope_exclusions == ["Do not edit CI"]
    assert first.integrity_anchors == {"a.py": "abc"}
    assert first.acceptance_checks == [{"criterion": "A passes"}]
    assert first.verification_results[0]["command"] == "pytest a"
    assert second.active_run_id == "run-b"
    assert second.scope_exclusions == []
    assert second.allowed_files == []
    assert second.integrity_anchors == {}
    assert second.acceptance_checks == []
    assert second.verification_results == []
    assert second.recent_test_observations == []


def test_invalid_run_does_not_replace_or_clear_active_state(tmp_path):
    _write_run(tmp_path, "run-valid", design={"allowed_files": ["agent.py"]})
    original = refresh_runtime_state(
        tmp_path,
        "edit files",
        objective="Valid contribution",
        run_id="run-valid",
        session_id="shared",
    )

    with pytest.raises(ValueError, match="invalid contribution run"):
        refresh_runtime_state(
            tmp_path,
            "invalid step",
            objective="Invalid contribution",
            run_id="../outside",
            session_id="shared",
        )

    persisted = load_runtime_state(tmp_path, session_id="shared")
    assert original.active_run_id == "run-valid"
    assert persisted.active_run_id == "run-valid"
    assert persisted.allowed_files == ["agent.py"]
    assert persisted.objective == "Valid contribution"


def test_load_runtime_state_resets_invalid_root_and_fields(tmp_path):
    path = runtime_state_path(tmp_path, "invalid")
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    invalid_root = load_runtime_state(tmp_path, session_id="invalid")

    assert "root must be an object" in invalid_root.unresolved_questions[0]

    path.write_text(
        json.dumps({"schema_version": RUNTIME_STATE_SCHEMA_VERSION, "unknown": True}),
        encoding="utf-8",
    )
    invalid_fields = load_runtime_state(tmp_path, session_id="invalid")

    assert "fields are incompatible" in invalid_fields.unresolved_questions[0]

    path.write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
                "allowed_files": "agent.py",
            }
        ),
        encoding="utf-8",
    )
    invalid_types = load_runtime_state(tmp_path, session_id="invalid")

    assert "fields are incompatible" in invalid_types.unresolved_questions[0]


def test_tool_observations_are_sanitized_and_separate_from_verification(tmp_path):
    save_runtime_state(
        tmp_path,
        RuntimeState(verification_results=[{"command": "controlled pytest", "exit_code": 0}]),
        session_id="observations",
    )
    record_tool_observation(
        tmp_path,
        "write_file",
        {
            "path": "secret.py",
            "content": "private source body",
            "metadata": {"api_key": "top-secret", "notes": "token=inline-secret"},
        },
        ToolResult(ok=False, summary="failed", error_code="TOOL_ERROR"),
        session_id="observations",
    )
    record_tool_observation(
        tmp_path,
        "bash",
        {"command": "API_KEY=test-secret python -m pytest tests/test_example.py"},
        ToolResult(ok=True, summary="passed", artifact_path="pytest.log"),
        session_id="observations",
    )

    state = load_runtime_state(tmp_path, session_id="observations")
    serialized = json.dumps(
        [item.model_dump(mode="json") for item in state.failed_strategies],
        ensure_ascii=False,
    )

    assert state.verification_results == [{"command": "controlled pytest", "exit_code": 0}]
    assert state.recent_test_observations[0].command == "API_KEY=[REDACTED] python -m pytest tests/test_example.py"
    assert state.recent_test_observations[0].artifact_path == "pytest.log"
    assert "private source body" not in serialized
    assert "top-secret" not in serialized
    assert "inline-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert "test-secret" not in json.dumps(
        [item.model_dump(mode="json") for item in state.recent_test_observations]
    )


def test_failed_strategies_deduplicate_by_fingerprint(tmp_path):
    first = ToolResult(
        ok=False,
        summary="first failure",
        error_code="TOOL_ERROR",
        fingerprint="stable-action",
    )
    second = ToolResult(
        ok=False,
        summary="same action, different detail",
        error_code="TOOL_ERROR",
        fingerprint="stable-action",
    )

    record_tool_observation(tmp_path, "read_file", {"path": "a.py"}, first)
    record_tool_observation(tmp_path, "read_file", {"path": "a.py"}, second)

    state = load_runtime_state(tmp_path)
    assert len(state.failed_strategies) == 1
    assert state.failed_strategies[0].occurrences == 2
    assert state.failed_strategies[0].summary == "same action, different detail"


def test_external_artifact_path_is_redacted(tmp_path):
    record_tool_observation(
        tmp_path,
        "bash",
        {"command": "pytest"},
        ToolResult(ok=True, summary="passed", artifact_path=str(tmp_path.parent / "secret.log")),
    )

    state = load_runtime_state(tmp_path)
    assert state.recent_test_observations[0].artifact_path == "[EXTERNAL_ARTIFACT]"


def test_unresolved_questions_are_deduplicated_and_bounded():
    state = RuntimeState(
        unresolved_questions=[" repeated ", "repeated", *[f"question-{i}" for i in range(25)]]
    )

    assert len(state.unresolved_questions) == 20
    assert state.unresolved_questions[-1] == "question-24"
