"""将声明式 Skill 准备为当前 Conversation 的输入。"""

from __future__ import annotations

import json

from osc_agent.runtime.models import CapabilityScope
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import (
    PreparedSkill,
    SkillPreparationFailure,
    SkillPreparationResult,
    SkillRequest,
)


class SkillPreparer:
    def __init__(self, catalog: SkillCatalog) -> None:
        self.catalog = catalog

    async def prepare(self, request: SkillRequest) -> SkillPreparationResult:
        descriptor = self.catalog.resolve(request.name, request.trigger)
        if descriptor is None:
            discovered = self.catalog.get(request.name)
            return SkillPreparationFailure(
                name=request.name,
                error=(
                    f"skill is not available for {request.trigger} invocation"
                    if discovered is not None
                    else "skill not found"
                ),
            )
        try:
            body = await SkillLoader.read_body(descriptor)
        except (OSError, UnicodeError, ValueError) as exc:
            return SkillPreparationFailure(name=request.name, error=str(exc))
        prompt = f"{body}\n\nArguments:\n{json.dumps(request.arguments, ensure_ascii=False, indent=2)}"
        allowed_tools = descriptor.manifest.allowed_tools
        if request.trigger == "product":
            allowed_tools |= descriptor.manifest.product_tools
        capabilities = request.caller_capabilities.intersect(
            CapabilityScope(allowed_tools=allowed_tools)
        )
        return PreparedSkill(
            name=request.name,
            prompt=prompt,
            capabilities=capabilities,
            completion_requirements=descriptor.manifest.completion,
        )
