"""验证 AgentApplication/Conversation 生命周期边界。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from osc_agent.application import AgentProfile, SkillInput, UserPrompt
from osc_agent.application.service import AgentApplication
from osc_agent.runtime.models import (
    CapabilityScope,
    Complete,
    CompletionRequirements,
    QueryConfig,
    ResumeQueryParams,
    RunCompleted,
    StartQueryParams,
)
from osc_agent.skills.models import PreparedSkill


class RecordingRuntime:
    def __init__(self) -> None:
        self.params = []

    async def query(self, params):
        self.params.append(params)
        yield RunCompleted(transition=Complete(reason="done"))


class SessionStore:
    def __init__(self, existing=None) -> None:
        self.existing = existing

    def load(self, _session_id):
        return self.existing


class SkillPreparer:
    def __init__(self) -> None:
        self.invocations = []

    async def prepare(self, request):
        self.invocations.append(request)
        return PreparedSkill(
            name=request.name,
            prompt="rendered skill",
            capabilities=CapabilityScope(allowed_tools=frozenset({"read_file"})),
            completion_requirements=CompletionRequirements(
                required_evidence=frozenset({"successful_test"})
            ),
        )


def _application(root: Path, *, existing=None, allowed_initial_skills=frozenset()):
    runtime = RecordingRuntime()
    skill_preparer = SkillPreparer()
    graph = SimpleNamespace(
        runtime=runtime,
        session_store=SessionStore(existing),
        skill_preparer=skill_preparer,
        query_config=QueryConfig(),
        general_capabilities=CapabilityScope(allowed_tools=frozenset({"read_file", "grep"})),
        discovery_prompt="discovery",
    )
    application = AgentApplication(
        graph,  # type: ignore[arg-type]
        repository_root=root,
        model="persisted-model",
        profile=AgentProfile(
            profile_id="custom-product",
            system_prompt="system",
            allowed_initial_skills=allowed_initial_skills,
            required_evidence=frozenset({"independent_verification"}),
        ),
    )
    return application, runtime, skill_preparer


async def _collect(events):
    return [event async for event in events]


def test_start_binds_repository_root_and_resolves_user_prompt(tmp_path: Path) -> None:
    application, runtime, _ = _application(tmp_path)

    asyncio.run(_collect(application.open_session("new").submit(UserPrompt(text="task"))))

    params = runtime.params[0]
    assert isinstance(params, StartQueryParams)
    assert params.repository_root == str(tmp_path.resolve())
    assert params.messages[0].content[0].text == "task"
    assert params.model == "persisted-model"


def test_new_session_rejects_empty_input(tmp_path: Path) -> None:
    application, _, _ = _application(tmp_path)

    with pytest.raises(ValueError, match="new Agent Session"):
        asyncio.run(_collect(application.open_session("new").submit(None)))


@pytest.mark.parametrize("input", [None, UserPrompt(text="follow up")])
def test_existing_session_resumes_with_optional_prompt(tmp_path: Path, input) -> None:
    application, runtime, _ = _application(tmp_path, existing=object())

    asyncio.run(_collect(application.open_session("existing").submit(input)))

    params = runtime.params[0]
    assert isinstance(params, ResumeQueryParams)
    assert params.repository_root == str(tmp_path.resolve())
    assert len(params.messages) == (0 if input is None else 1)


def test_existing_session_rejects_skill_input(tmp_path: Path) -> None:
    application, _, _ = _application(tmp_path, existing=object())

    with pytest.raises(ValueError, match="can only start"):
        asyncio.run(
            _collect(application.open_session("existing").submit(SkillInput(name="skill")))
        )


def test_skill_start_uses_shared_preparer_and_tightens_requirements(tmp_path: Path) -> None:
    application, runtime, skill_preparer = _application(
        tmp_path,
        allowed_initial_skills=frozenset({"skill"}),
    )

    asyncio.run(
        _collect(
            application.open_session("skill").submit(
                SkillInput(name="skill", arguments={"value": "x"})
            )
        )
    )

    assert skill_preparer.invocations[0].trigger == "product"
    params = runtime.params[0]
    assert params.messages[0].content[0].text == "rendered skill"
    assert params.capabilities.allowed_tools == frozenset({"read_file"})
    assert params.completion_requirements.required_evidence == frozenset(
        {"successful_test", "independent_verification"}
    )


def test_skill_start_requires_explicit_profile_authorization(tmp_path: Path) -> None:
    application, runtime, skill_preparer = _application(tmp_path)

    with pytest.raises(ValueError, match="not allowed by Agent Profile"):
        asyncio.run(
            _collect(
                application.open_session("skill").submit(SkillInput(name="hidden"))
            )
        )

    assert runtime.params == []
    assert skill_preparer.invocations == []
