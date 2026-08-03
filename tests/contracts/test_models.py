"""验证Runtime 数据模型的契约、边界条件与回归行为。"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from osc_agent.completion.models import CompletionRequirements
from osc_agent.runtime.state import AgentRunState, CapabilityScope
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.query_models import QueryConfig
from osc_agent.runtime.tool_models import (
    Allow,
    PermissionDecision,
    ToolResult,
)
from osc_agent.workspaces.models import (
    FileObservation,
    RepositoryInstructionState,
    WorktreeSession,
)


def test_contracts_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        QueryConfig.model_validate({"unexpected": True})


def test_contracts_reject_implicit_type_coercion() -> None:
    with pytest.raises(ValidationError, match="int_type"):
        QueryConfig.model_validate({"max_rounds": "30"})


def test_frozen_contract_cannot_be_modified() -> None:
    config = QueryConfig()

    with pytest.raises(ValidationError, match="frozen_instance"):
        config.max_rounds = 1


def test_permission_decision_uses_discriminator() -> None:
    adapter = TypeAdapter(PermissionDecision)

    assert adapter.validate_python({"decision": "allow"}) == Allow()
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"decision": "maybe"})


def test_tool_result_rejects_non_json_data() -> None:
    with pytest.raises(ValidationError):
        ToolResult(data=object())


def test_frozen_base_is_itself_strict() -> None:
    class Sample(FrozenContractModel):
        count: int

    with pytest.raises(ValidationError, match="int_type"):
        Sample.model_validate({"count": "1"})


def test_tool_context_cannot_mutate_authoritative_workspace_state() -> None:
    state = AgentRunState.start(
        workspace_root="repository",
        capabilities=CapabilityScope(),
        completion_requirements=CompletionRequirements(),
        instruction_state=RepositoryInstructionState(active_paths=("AGENTS.md",)),
    )
    state.workspace.file_observations["a.py"] = FileObservation(
        path="a.py",
        content_hash="before",
        mtime_ns=1,
        complete=True,
    )

    context = state.tool_context(session_id="session", state_directory="state")
    context.workspace.file_observations["a.py"] = FileObservation(
        path="a.py",
        content_hash="after",
        mtime_ns=2,
        complete=True,
    )

    assert context.workspace is not state.workspace
    assert state.workspace.file_observations["a.py"].content_hash == "before"
    with pytest.raises(ValidationError, match="frozen_instance"):
        context.workspace.instruction_state.active_paths = ("CLAUDE.md",)


def test_workspace_value_records_are_frozen() -> None:
    worktree = WorktreeSession(
        path="worktree",
        original_working_directory="repository",
        branch="branch",
        base_commit="commit",
    )
    observation = FileObservation(
        path="a.py",
        content_hash="hash",
        mtime_ns=1,
        complete=True,
    )

    with pytest.raises(ValidationError, match="frozen_instance"):
        worktree.branch = "changed"
    with pytest.raises(ValidationError, match="frozen_instance"):
        observation.complete = False


def test_child_capabilities_can_only_narrow_the_caller_scope() -> None:
    caller = CapabilityScope(allowed_tools=frozenset({"read", "write"}))
    child = CapabilityScope(allowed_tools=frozenset({"read", "shell"}))

    assert caller.intersect(child).allowed_tools == frozenset({"read"})


def test_completion_requirements_tighten_without_adding_waiver_to_existing_rule() -> None:
    existing = CompletionRequirements(
        required_evidence=frozenset({"successful_test"}),
    )
    incoming = CompletionRequirements(
        required_evidence=frozenset(
            {"successful_test", "git_change_snapshot"}
        ),
        waivable_evidence=frozenset({"successful_test"}),
    )

    tightened = existing.tighten(incoming)

    assert tightened.required_evidence == {
        "successful_test",
        "git_change_snapshot",
    }
    assert tightened.waivable_evidence == frozenset()


def test_completion_requirements_tighten_is_order_independent_and_idempotent() -> None:
    waivable = CompletionRequirements(
        required_evidence=frozenset({"successful_test", "git_change_snapshot"}),
        waivable_evidence=frozenset({"successful_test"}),
    )
    strict = CompletionRequirements(
        required_evidence=frozenset({"successful_test", "independent_verification"}),
    )
    delivery = CompletionRequirements(
        required_evidence=frozenset({"delivery_draft"}),
        waivable_evidence=frozenset({"delivery_draft"}),
    )

    assert waivable.tighten(strict) == strict.tighten(waivable)
    assert waivable.tighten(waivable) == waivable
    assert waivable.tighten(strict).tighten(delivery) == waivable.tighten(
        strict.tighten(delivery)
    )
    assert "successful_test" not in waivable.tighten(strict).waivable_evidence
