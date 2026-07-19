from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from osc_agent.harness.contracts import RunStatus, StageStatus
from osc_agent.harness.repository_boundary import (
    normalize_repo_relative_path,
    normalize_repo_relative_pattern,
)

STATE_SCHEMA_VERSION = 3
DEFAULT_FORBIDDEN_PATHS = [
    ".git/**",
    ".github/**",
    ".env*",
    "*lock*",
    "**/security/**",
]

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CleanStr = Annotated[str, StringConstraints(strip_whitespace=True)]
RequirementId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^REQ-[1-9][0-9]*$"),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _unique(values: Iterable[str]) -> list[str]:
    result = list(values)
    if len(result) != len(set(result)):
        raise ValueError("scope contract paths must be unique")
    return result


class ContractModel(BaseModel):
    """持久化契约的共同约束；名称不暗示 Pydantic strict 模式。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=True)


class ContributionStage(str, Enum):
    DISCOVER = "discover"
    DESIGN = "design"
    IMPLEMENT = "implement"
    DRAFT_PR = "draft_pr"


STAGE_ORDER = tuple(stage.value for stage in ContributionStage)
STAGES = frozenset(STAGE_ORDER)


class RequirementKind(str, Enum):
    BEHAVIOR = "behavior"
    DOCS = "docs"
    CONFIG = "config"


class DirectionSourceKind(str, Enum):
    ISSUE = "issue"
    ARCHITECTURE = "architecture"


class ReproductionMode(str, Enum):
    EXISTING = "existing"
    GENERATED_TEST = "generated_test"


class UnderstandingDecision(str, Enum):
    READY_TO_EDIT = "READY_TO_EDIT"
    CONTRACT_UPDATE_REQUIRED = "CONTRACT_UPDATE_REQUIRED"


class TransitionStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class Requirement(ContractModel):
    id: RequirementId
    text: NonBlankStr
    source: NonBlankStr
    source_excerpt: NonBlankStr
    kind: RequirementKind = RequirementKind.BEHAVIOR


class ContributionDirection(ContractModel):
    id: NonBlankStr
    name: NonBlankStr
    description: NonBlankStr
    source: NonBlankStr
    source_kind: DirectionSourceKind
    issue_number: int | None = Field(default=None, ge=1)
    entry: NonBlankStr
    effort: NonBlankStr
    interview: NonBlankStr
    risk: NonBlankStr

    @model_validator(mode="after")
    def validate_source_identity(self) -> "ContributionDirection":
        if self.source_kind == DirectionSourceKind.ISSUE.value and self.issue_number is None:
            raise ValueError("issue directions must define issue_number")
        if self.source_kind != DirectionSourceKind.ISSUE.value and self.issue_number is not None:
            raise ValueError("non-issue directions cannot define issue_number")
        return self


class AcceptanceCheck(ContractModel):
    criterion: NonBlankStr
    command: CleanStr = ""
    manual_check: bool = False
    requirement_ids: list[RequirementId] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_execution_method(self) -> "AcceptanceCheck":
        if not self.manual_check and not self.command:
            raise ValueError("automatic acceptance checks must define command")
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("acceptance check requirement_ids must be unique")
        return self


class BaselineCheck(ContractModel):
    command: NonBlankStr
    expected_exit_codes: list[int] = Field(min_length=1)
    output_contains: NonBlankStr

    @field_validator("expected_exit_codes")
    @classmethod
    def validate_failure_exit_codes(cls, value: list[int]) -> list[int]:
        if any(code <= 0 for code in value):
            raise ValueError("failure baseline expected_exit_codes must contain only non-zero exit codes")
        if len(value) != len(set(value)):
            raise ValueError("failure baseline expected_exit_codes must be unique")
        return value


class ReproductionPlan(ContractModel):
    mode: ReproductionMode | None = None
    command: CleanStr = ""
    test_files: list[NonBlankStr] = Field(default_factory=list)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_empty_mode(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("test_files")
    @classmethod
    def normalize_test_files(cls, value: list[str]) -> list[str]:
        normalized = [path.replace("\\", "/") for path in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("reproduction test_files must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "ReproductionPlan":
        if self.mode == ReproductionMode.GENERATED_TEST.value:
            if not self.command or not self.test_files:
                raise ValueError("generated reproduction requires command and test_files")
        return self


class ContributionSpec(ContractModel):
    task_type: RequirementKind
    requirements: list[Requirement] = Field(min_length=1)
    baseline_checks: list[BaselineCheck] = Field(default_factory=list)
    reproduction: ReproductionPlan = Field(default_factory=ReproductionPlan)

    @model_validator(mode="after")
    def validate_requirements(self) -> "ContributionSpec":
        ids = [item.id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement ids must be unique")
        mismatched = [item.id for item in self.requirements if item.kind != self.task_type]
        if mismatched:
            raise ValueError("requirement kind must match contribution task_type")
        return self


class SourceEvidence(ContractModel):
    commit_sha: CleanStr = ""
    file: NonBlankStr
    symbol: CleanStr = ""
    line_range: tuple[int, int]
    content_hash: NonBlankStr
    requirement_ids: list[RequirementId] = Field(min_length=1)


class ScopeContract(BaseModel):
    """从完整 Design Artifact 投影出的严格文件能力边界。"""

    model_config = ConfigDict(extra="ignore")

    allowed_files: list[str]
    allowed_new_dirs: list[str]
    forbidden_paths: list[str] = Field(min_length=1)
    max_changed_files: int = Field(gt=0, strict=True)
    max_diff_lines: int = Field(gt=0, strict=True)

    @field_validator("allowed_files", "allowed_new_dirs", mode="before")
    @classmethod
    def validate_path_list_type(cls, value: object, info: Any) -> object:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{info.field_name} must be a list of strings")
        return value

    @field_validator("forbidden_paths", mode="before")
    @classmethod
    def validate_pattern_list_type(cls, value: object) -> object:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("forbidden_paths must be a list of strings")
        return value

    @field_validator("allowed_files")
    @classmethod
    def normalize_allowed_files(cls, value: list[str]) -> list[str]:
        return _unique(
            normalize_repo_relative_path(path, field_name="allowed_files") for path in value
        )

    @field_validator("allowed_new_dirs")
    @classmethod
    def normalize_allowed_new_dirs(cls, value: list[str]) -> list[str]:
        return _unique(
            normalize_repo_relative_path(path, field_name="allowed_new_dirs") for path in value
        )

    @field_validator("forbidden_paths")
    @classmethod
    def normalize_forbidden_paths(cls, value: list[str]) -> list[str]:
        return _unique(
            normalize_repo_relative_pattern(path, field_name="forbidden_paths") for path in value
        )


class DesignContract(BaseModel):
    """Design artifact 中跨字段关联所需的稳定契约子集。"""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    selected_direction: NonBlankStr
    contribution_spec: ContributionSpec
    acceptance_checks: list[AcceptanceCheck] = Field(min_length=1)
    source_evidence: list[SourceEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_requirement_coverage(self) -> "DesignContract":
        requirement_ids = {item.id for item in self.contribution_spec.requirements}
        acceptance_ids = {
            requirement_id
            for check in self.acceptance_checks
            for requirement_id in check.requirement_ids
        }
        if not acceptance_ids <= requirement_ids:
            raise ValueError("acceptance checks reference unknown requirements")
        if not requirement_ids <= acceptance_ids:
            raise ValueError("acceptance checks must cover every requirement")

        evidence_ids = {
            requirement_id
            for evidence in self.source_evidence
            for requirement_id in evidence.requirement_ids
        }
        if not evidence_ids <= requirement_ids:
            raise ValueError("source evidence references unknown requirements")
        if self.source_evidence and not requirement_ids <= evidence_ids:
            raise ValueError("source evidence must cover every requirement")
        return self


class UnderstandingCheckpoint(ContractModel):
    """Understanding 阶段唯一允许的结构化决策输出。"""

    decision: UnderstandingDecision
    summary: NonBlankStr
    requirement_ids: list[RequirementId] = Field(min_length=1)
    files_to_modify: list[NonBlankStr] = Field(min_length=1)
    unresolved_questions: list[NonBlankStr] = Field(default_factory=list)

    @field_validator("requirement_ids", "files_to_modify")
    @classmethod
    def validate_unique_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("understanding checkpoint lists must contain unique items")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> "UnderstandingCheckpoint":
        if self.decision == UnderstandingDecision.READY_TO_EDIT.value and self.unresolved_questions:
            raise ValueError("READY_TO_EDIT cannot contain unresolved questions")
        if self.decision == UnderstandingDecision.CONTRACT_UPDATE_REQUIRED.value and not self.unresolved_questions:
            raise ValueError("CONTRACT_UPDATE_REQUIRED must explain unresolved questions")
        return self


class PRDraftNarrative(ContractModel):
    """LLM 只负责叙事字段，变更和测试事实由工作流填充。"""

    title: NonBlankStr
    problem: NonBlankStr
    solution: NonBlankStr
    reviewer_notes: list[NonBlankStr] = Field(min_length=1)


class PRDraftArtifact(PRDraftNarrative):
    schema_version: Literal[1] = 1
    generation_mode: Literal["llm"] = "llm"
    run_id: NonBlankStr
    run_revision: int = Field(ge=0)
    head_sha: NonBlankStr
    diff_hash: Sha256
    implementation_artifact_hash: Sha256
    generated_at: NonBlankStr
    changed_files: list[NonBlankStr] = Field(min_length=1)
    changes: list[NonBlankStr] = Field(min_length=1)
    testing: list[NonBlankStr] = Field(min_length=1)

    @field_validator("changed_files")
    @classmethod
    def validate_changed_files(cls, value: list[str]) -> list[str]:
        normalized = [path.replace("\\", "/") for path in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("changed_files must be unique")
        return normalized


class TransitionRecord(ContractModel):
    stage: ContributionStage | None = None
    from_stage: ContributionStage | None = Field(default=None, alias="from")
    to: ContributionStage | None = None
    status: TransitionStatus
    reason: NonBlankStr | None = None
    at: NonBlankStr

    @model_validator(mode="after")
    def validate_transition_shape(self) -> "TransitionRecord":
        if self.stage is None and self.to is None:
            raise ValueError("transition must define stage or to")
        if self.status == TransitionStatus.BLOCKED.value and not self.reason:
            raise ValueError("blocked transition must define reason")
        return self


class ContributionRun(ContractModel):
    run_id: NonBlankStr
    repo_root: NonBlankStr
    repo_url: NonBlankStr
    stage: ContributionStage
    selected_direction: NonBlankStr | None = None
    selected_direction_id: NonBlankStr | None = None
    selected_issue_number: int | None = Field(default=None, ge=1)
    artifacts_dir: NonBlankStr
    schema_version: Literal[3] = STATE_SCHEMA_VERSION
    base_commit_sha: str = ""
    issue_snapshot_at: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    stage_status: dict[ContributionStage, StageStatus]
    stage_hashes: dict[str, str] = Field(default_factory=dict)
    critical_file_hashes: dict[str, str] = Field(default_factory=dict)
    final_status: RunStatus | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    worktree_root: str | None = None
    implementation_checkpoint: dict[str, Any] = Field(default_factory=dict)
    last_transition: TransitionRecord | None = None
    recovery_stage: ContributionStage | None = None
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_stage_status(self) -> "ContributionRun":
        actual = set(self.stage_status)
        if actual != STAGES:
            missing = sorted(STAGES - actual)
            unknown = sorted(actual - STAGES)
            raise ValueError(f"stage_status must cover every stage; missing={missing}, unknown={unknown}")
        return self
