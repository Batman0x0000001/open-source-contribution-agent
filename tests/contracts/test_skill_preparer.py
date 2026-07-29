"""验证 Skill 参数渲染、授权和能力收窄。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from osc_agent.runtime.models import CapabilityScope
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import PreparedSkill, SkillPreparationFailure, SkillRequest
from osc_agent.skills.preparer import SkillPreparer


def _write_skill(root: Path, *, source_text: str = "Review carefully.") -> Path:
    path = root / "review" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: review
description: Review a change
when_to_use: Before submitting a change
allowed_tools: [read_file, edit_file]
product_tools: [submit_artifact]
completion:
  required_evidence: [successful_test]
---
{source_text}
""",
        encoding="utf-8",
    )
    return path


def _request(*, trigger: str = "user", arguments=None, allowed_tools=None) -> SkillRequest:
    return SkillRequest(
        name="review",
        arguments=arguments or {"target": 123, "options": [True, None]},
        trigger=trigger,
        caller_capabilities=CapabilityScope(
            allowed_tools=frozenset(allowed_tools or {"read_file", "grep"})
        ),
    )


def test_preparer_accepts_free_json_and_tightens_capabilities(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    preparer = SkillPreparer(SkillCatalog([SkillLoader(tmp_path, source="builtin")]))

    result = asyncio.run(preparer.prepare(_request()))

    assert isinstance(result, PreparedSkill)
    assert '"target": 123' in result.prompt
    assert '"options": [' in result.prompt
    assert result.capabilities.allowed_tools == frozenset({"read_file"})
    assert result.completion_requirements.required_evidence == {"successful_test"}


def test_request_requires_an_explicit_supported_trigger() -> None:
    with pytest.raises(ValidationError):
        SkillRequest.model_validate(
            {"name": "review", "arguments": {}, "caller_capabilities": {}}
        )
    with pytest.raises(ValidationError):
        _request(trigger="internal")


def test_model_cannot_invoke_external_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    preparer = SkillPreparer(SkillCatalog([SkillLoader(tmp_path, source="project")]))

    result = asyncio.run(preparer.prepare(_request(trigger="model")))

    assert isinstance(result, SkillPreparationFailure)
    assert result.error == "skill is not available for model invocation"


def test_only_product_invocation_can_receive_product_tools(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    preparer = SkillPreparer(SkillCatalog([SkillLoader(tmp_path, source="builtin")]))
    caller_tools = {"read_file", "submit_artifact"}

    user = asyncio.run(
        preparer.prepare(_request(trigger="user", allowed_tools=caller_tools))
    )
    product = asyncio.run(
        preparer.prepare(_request(trigger="product", allowed_tools=caller_tools))
    )

    assert isinstance(user, PreparedSkill)
    assert isinstance(product, PreparedSkill)
    assert user.capabilities.allowed_tools == {"read_file"}
    assert product.capabilities.allowed_tools == {"read_file", "submit_artifact"}
