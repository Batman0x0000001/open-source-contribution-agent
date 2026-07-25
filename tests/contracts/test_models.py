from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from osc_agent.runtime.models import (
    Allow,
    CapabilityScope,
    FrozenContractModel,
    PermissionDecision,
    QueryConfig,
    QueryState,
    ToolExecutionUpdate,
    ToolResult,
    ToolUseContext,
    Transition,
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


def test_mutable_state_validates_assignment() -> None:
    state = QueryState(session_id="session-1")

    with pytest.raises(ValidationError, match="literal_error"):
        state.status = "unknown"


def test_permission_decision_uses_discriminator() -> None:
    adapter = TypeAdapter(PermissionDecision)

    assert adapter.validate_python({"decision": "allow"}) == Allow()
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"decision": "maybe"})


def test_transition_rejects_unknown_kind() -> None:
    adapter = TypeAdapter(Transition)

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"kind": "loop_again"})


def test_tool_result_rejects_non_json_data() -> None:
    with pytest.raises(ValidationError):
        ToolResult(data=object())


def test_frozen_base_is_itself_strict() -> None:
    class Sample(FrozenContractModel):
        count: int

    with pytest.raises(ValidationError, match="int_type"):
        Sample.model_validate({"count": "1"})


def test_tool_execution_update_requires_result_and_identity_together() -> None:
    context = ToolUseContext(session_id="session-1", working_directory="C:/repo", repository_root="C:/repo", state_directory="C:/state")

    with pytest.raises(ValidationError, match="must either both be set or both be omitted"):
        ToolExecutionUpdate(tool_use_id="call-1", context=context)


def test_child_capabilities_can_only_narrow_the_caller_scope() -> None:
    caller = CapabilityScope(allowed_tools=frozenset({"read", "write"}))
    child = CapabilityScope(allowed_tools=frozenset({"read", "shell"}))

    assert caller.intersect(child).allowed_tools == frozenset({"read"})
