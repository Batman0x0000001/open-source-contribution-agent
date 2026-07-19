from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from osc_agent.harness.repository_boundary import (
    normalize_repo_relative_path,
    normalize_repo_relative_pattern,
    repo_path_matches,
)


READ_ONLY_CONTRIBUTION_TOOLS = {
    "read_file",
    "glob",
    "git_status",
    "git_diff",
    "git_log",
    "inspect_repo",
}


class AgentExecutionStage(str, Enum):
    INTERACTIVE = "interactive"
    UNDERSTANDING = "understanding"
    REPRODUCE = "reproduce"
    EDIT = "edit"
    REPAIR = "repair"
    VERIFICATION = "verification"


class AgentCapabilityScope(BaseModel):
    """声明一次 Agent 运行可以使用的工具和写入范围。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: AgentExecutionStage = AgentExecutionStage.INTERACTIVE
    run_id: str | None = None
    allowed_tools: frozenset[str] | None = None
    writable_paths: tuple[str, ...] = Field(default_factory=tuple)
    forbidden_paths: tuple[str, ...] = Field(default_factory=tuple)

    @classmethod
    def unrestricted(cls) -> "AgentCapabilityScope":
        return cls()

    @classmethod
    def contribution(
        cls,
        stage: AgentExecutionStage | str,
        design: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> "AgentCapabilityScope":
        active_stage = AgentExecutionStage(stage)
        if active_stage is AgentExecutionStage.INTERACTIVE:
            raise ValueError("interactive stage must use AgentCapabilityScope.unrestricted()")

        allowed = set(READ_ONLY_CONTRIBUTION_TOOLS)
        writable: list[str] = []
        spec = design.get("contribution_spec") or {}
        reproduction = spec.get("reproduction") or {}
        frozen_tests = _normalized_paths(reproduction.get("test_files"), "reproduction.test_files")
        design_forbidden = _normalized_patterns(design.get("forbidden_paths"), "forbidden_paths")
        if active_stage is AgentExecutionStage.REPRODUCE:
            allowed.update({"write_file", "edit_file"})
            writable = frozen_tests
        elif active_stage in {AgentExecutionStage.EDIT, AgentExecutionStage.REPAIR}:
            allowed.update({"write_file", "edit_file"})
            writable = _normalized_paths(design.get("allowed_files"), "allowed_files")
            writable.extend(
                path + "/**"
                for path in _normalized_paths(design.get("allowed_new_dirs"), "allowed_new_dirs")
            )
        return cls(
            stage=active_stage,
            run_id=run_id,
            allowed_tools=frozenset(allowed),
            writable_paths=tuple(writable),
            forbidden_paths=tuple(
                design_forbidden
                + (
                    frozen_tests
                    if active_stage in {AgentExecutionStage.EDIT, AgentExecutionStage.REPAIR}
                    else []
                )
            ),
        )

    def permits_tool(self, tool_name: str) -> bool:
        return self.allowed_tools is None or tool_name in self.allowed_tools

    def permits_write(self, path: str) -> bool:
        if self.allowed_tools is None:
            return True
        try:
            normalized = normalize_repo_relative_path(path, field_name="write path")
        except ValueError:
            return False
        if any(repo_path_matches(normalized, pattern) for pattern in self.forbidden_paths):
            return False
        return any(repo_path_matches(normalized, pattern) for pattern in self.writable_paths)

    def filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [tool for tool in tools if self.permits_tool(str(tool.get("name", "")))]


def _normalized_paths(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return [normalize_repo_relative_path(item, field_name=field_name) for item in value]


def _normalized_patterns(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return [normalize_repo_relative_pattern(item, field_name=field_name) for item in value]
