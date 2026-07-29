"""验证Runtime 数据模型的契约、边界条件与回归行为。"""

from __future__ import annotations

from tests.runtime_factories import tool_context

import pytest
from pydantic import TypeAdapter, ValidationError

from osc_agent.completion.models import CompletionRequirements
from osc_agent.runtime.state import CapabilityScope
from osc_agent.contracts import FrozenContractModel
from osc_agent.runtime.query_models import QueryConfig
from osc_agent.runtime.tool_models import (
    Allow,
    PermissionDecision,
    ToolResult,
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
