from __future__ import annotations

import json
from pydantic import ConfigDict, JsonValue, ValidationError, create_model

from osc_agent.agents.definitions import AgentDefinition, AgentInvocation
from osc_agent.agents.runner import AgentRunner
from osc_agent.runtime.models import CapabilityScope
from osc_agent.runtime.models import QueryConfig
from osc_agent.skills.catalog import SkillCatalog
from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import SkillInvocation, SkillResult


class SkillExecutor:
    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        agent_runner: AgentRunner | None = None,
        query_config: QueryConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.agent_runner = agent_runner
        self.query_config = query_config or QueryConfig()

    async def execute(self, invocation: SkillInvocation) -> SkillResult:
        descriptor = self.catalog.get(invocation.name)
        if descriptor is None:
            return SkillResult(name=invocation.name, status="failed", error="skill not found")
        if invocation.trigger == "user" and not descriptor.manifest.user_invocable:
            return SkillResult(name=invocation.name, status="failed", error="skill is not user invocable")
        if invocation.trigger == "model" and descriptor.manifest.disable_model_invocation:
            return SkillResult(name=invocation.name, status="failed", error="model invocation is disabled")
        try:
            input_model = _contract_model(f"{invocation.name}Input", descriptor.manifest.input_schema)
            validated_input = input_model.model_validate(invocation.arguments)
        except (TypeError, ValueError, ValidationError) as exc:
            return SkillResult(name=invocation.name, status="failed", error=str(exc))

        body = SkillLoader.read_body(descriptor)
        prompt = f"{body}\n\nArguments:\n{validated_input.model_dump_json(indent=2)}"
        capabilities = invocation.caller_capabilities.intersect(
            CapabilityScope(allowed_tools=descriptor.manifest.allowed_tools)
        )
        if descriptor.manifest.context == "inline":
            return SkillResult(
                name=invocation.name,
                status="inline",
                rendered_prompt=prompt,
                capabilities=capabilities,
                completion_requirements=descriptor.manifest.completion,
            )
        if self.agent_runner is None:
            return SkillResult(name=invocation.name, status="failed", error="fork skill requires AgentRunner")

        definition = AgentDefinition(
            name=f"skill:{invocation.name}",
            description=descriptor.manifest.description,
            system_prompt="Execute the supplied skill instructions and return only its declared JSON output.",
            model=descriptor.manifest.model,
            capabilities=capabilities,
            config=self.query_config,
        )
        run = await self.agent_runner.run_with_definition(
            definition,
            AgentInvocation(
                agent_name=definition.name,
                prompt=prompt,
                mode="fork" if invocation.parent_messages else "inline",
                parent_session_id=invocation.session_id,
                working_directory=invocation.working_directory,
                caller_capabilities=invocation.caller_capabilities,
                parent_messages=invocation.parent_messages,
            ),
        )
        if run.status != "completed":
            return SkillResult(name=invocation.name, status="failed", error=run.error or run.status)
        try:
            output_model = _contract_model(f"{invocation.name}Output", descriptor.manifest.output_schema)
            output = output_model.model_validate(json.loads(run.output))
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            return SkillResult(name=invocation.name, status="failed", error=str(exc))
        return SkillResult(
            name=invocation.name,
            status="completed",
            output=output.model_dump(mode="json"),
            completion_requirements=descriptor.manifest.completion,
        )


def _contract_model(name: str, schema: dict[str, JsonValue]):
    if schema.get("type") != "object":
        raise ValueError("skill schema root must be an object")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("skill schema properties must be an object")
    required_value = schema.get("required", [])
    if not isinstance(required_value, list) or not all(isinstance(item, str) for item in required_value):
        raise ValueError("skill schema required must be a string list")
    required = set(required_value)
    fields = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_name, str) or not isinstance(field_schema, dict):
            raise ValueError("skill schema properties are invalid")
        annotation = _annotation(field_schema)
        fields[field_name] = (
            annotation if field_name in required else annotation | None,
            ... if field_name in required else None,
        )
    return create_model(
        name.replace(":", "_"),
        __config__=ConfigDict(
            extra="allow" if schema.get("additionalProperties") is True else "forbid",
            strict=True,
        ),
        **fields,
    )


def _annotation(schema: dict[str, JsonValue]):
    kind = schema.get("type")
    primitive = {"string": str, "integer": int, "number": float, "boolean": bool}
    if isinstance(kind, str) and kind in primitive:
        return primitive[kind]
    if kind == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            raise ValueError("array schema requires items")
        return list[_annotation(items)]
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("object schema properties must be an object")
        nested = _contract_model("NestedSkillObject", schema)
        return nested
    if isinstance(kind, list) and "null" in kind and len(kind) == 2:
        other = next(item for item in kind if item != "null")
        return _annotation({"type": other}) | None
    raise ValueError(f"unsupported skill schema type: {kind}")
