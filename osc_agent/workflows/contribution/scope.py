from __future__ import annotations

from pathlib import Path
from typing import Any

from osc_agent.harness.repository_boundary import repo_path_is_within, repo_path_matches
from osc_agent.tools.git import GitChange, GitChangeKind, git_changes
from osc_agent.workflows.contribution.models import ScopeContract


def validate_implementation_scope(repo_root: Path, design: dict[str, Any]) -> dict[str, Any]:
    contract = ScopeContract.model_validate(design)
    changes = [change for change in git_changes(repo_root=repo_root) if not _is_runtime_change(change)]
    allowed_files = set(contract.allowed_files)

    outside_scope = sorted(
        change.path
        for change in changes
        if change.path not in allowed_files
        and not (
            change.kind is GitChangeKind.ADDED
            and any(repo_path_is_within(change.path, directory) for directory in contract.allowed_new_dirs)
        )
    )
    forbidden_changes = sorted(
        {
            path
            for change in changes
            for path in (change.path, change.old_path)
            if path and any(repo_path_matches(path, pattern) for pattern in contract.forbidden_paths)
        }
    )
    added_files = _paths_for_kind(changes, GitChangeKind.ADDED)
    modified_files = _paths_for_kind(changes, GitChangeKind.MODIFIED)
    deleted_files = _paths_for_kind(changes, GitChangeKind.DELETED)
    renamed_files = sorted(
        [
            {"from": change.old_path or "", "to": change.path}
            for change in changes
            if change.kind is GitChangeKind.RENAMED
        ],
        key=lambda item: (item["to"], item["from"]),
    )
    copied_files = _paths_for_kind(changes, GitChangeKind.COPIED)
    unmerged_files = _paths_for_kind(changes, GitChangeKind.UNMERGED)
    binary_files = sorted(change.path for change in changes if change.binary)
    added_lines = sum(change.added_lines for change in changes)
    deleted_lines = sum(change.deleted_lines for change in changes)

    violations: list[str] = []
    if not changes:
        violations.append("no implementation files changed")
    if outside_scope:
        violations.append(f"files outside approved scope: {', '.join(outside_scope)}")
    if forbidden_changes:
        violations.append(f"forbidden files changed: {', '.join(forbidden_changes)}")
    if deleted_files:
        violations.append(f"file deletion is not approved by the Scope Contract: {', '.join(deleted_files)}")
    if renamed_files:
        names = ", ".join(f"{item['from']} -> {item['to']}" for item in renamed_files)
        violations.append(f"file rename is not approved by the Scope Contract: {names}")
    if copied_files:
        violations.append(f"file copy is not approved by the Scope Contract: {', '.join(copied_files)}")
    if unmerged_files:
        violations.append(f"unmerged files are not allowed: {', '.join(unmerged_files)}")
    if binary_files:
        violations.append(f"binary file changes are not approved: {', '.join(binary_files)}")
    if len(changes) > contract.max_changed_files:
        violations.append(
            f"changed file budget exceeded: {len(changes)} > {contract.max_changed_files}"
        )
    if added_lines + deleted_lines > contract.max_diff_lines:
        violations.append(
            f"diff line budget exceeded: {added_lines + deleted_lines} > {contract.max_diff_lines}"
        )
    return {
        "ok": not violations,
        "changed_files": sorted(change.path for change in changes),
        "added_files": added_files,
        "modified_files": modified_files,
        "deleted_files": deleted_files,
        "renamed_files": renamed_files,
        "copied_files": copied_files,
        "unmerged_files": unmerged_files,
        "binary_files": binary_files,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "outside_scope": outside_scope,
        "forbidden_changes": forbidden_changes,
        "violations": violations,
    }


def is_runtime_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return (
        normalized.startswith(".osc_agent/")
        or normalized in {".coverage", "coverage.xml"}
        or normalized.endswith((".pyc", ".pyo"))
        or any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"} for part in parts)
    )


def _is_runtime_change(change: GitChange) -> bool:
    return is_runtime_artifact(change.path) and (
        change.old_path is None or is_runtime_artifact(change.old_path)
    )


def _paths_for_kind(changes: list[GitChange], kind: GitChangeKind) -> list[str]:
    return sorted(change.path for change in changes if change.kind is kind)
