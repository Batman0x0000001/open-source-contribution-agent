"""验证 Bot 只从已绑定的业务合同构造 Skill 输入。"""

from __future__ import annotations

from uuid import uuid4

import pytest

from osc_agent.bot.worker.agent_jobs import (
    _build_implementation_application,
    _build_plan_application,
    _bot_session_events,
    build_implementation_skill_input,
    build_planning_skill_input,
)
from osc_agent.bot.config import BotWorkerSettings
from osc_agent.bot.domain.artifacts import IssuePlanArtifact
from osc_agent.bot.domain.execution import ExecutionContract
from osc_agent.bot.domain.jobs import BotJob
from osc_agent.bot.domain.repositories import RepositoryBotConfig
from osc_agent.bot.persistence.session_store import SqliteSessionStore
from osc_agent.bot.persistence.store import BotStore
from osc_agent.processes.runner import DisabledProcessRunner
from tests.runtime_factories import tool_context
from tests.settings_factory import make_agent_settings as Settings


def _contract() -> ExecutionContract:
    return ExecutionContract(
        repository_id=1,
        repository_full_name="owner/repo",
        installation_id=2,
        base_branch="main",
        base_sha="a" * 40,
        issue_number=3,
        issue_input_hash="b" * 64,
        model_id="test-model",
        plan_allowed_tools=frozenset({"read_file", "submit_issue_plan"}),
        implementation_allowed_tools=frozenset(
            {"read_file", "edit_file", "submit_delivery_draft"}
        ),
        validation_commands=("python -m pytest",),
        denied_paths=(".git/**",),
        max_changed_files=10,
        max_patch_bytes=10_000,
        image_id="sha256:" + "c" * 64,
        command_timeout_seconds=60,
        container_cpus=1.0,
        container_memory="1g",
        container_pids=64,
    )


def _job(contract: ExecutionContract) -> BotJob:
    return BotJob(
        job_id=str(uuid4()),
        repository_id=contract.repository_id,
        repository_full_name=contract.repository_full_name,
        installation_id=contract.installation_id,
        issue_number=contract.issue_number,
        issue_url="https://github.com/owner/repo/issues/3",
        base_sha=contract.base_sha,
        issue_input_hash=contract.issue_input_hash,
        execution_contract_hash=contract.contract_hash,
        image_id=contract.image_id,
        status="running_plan",
    )


def test_bot_builds_planning_and_implementation_skill_inputs_from_bound_models() -> None:
    contract = _contract()
    job = _job(contract)
    planning = build_planning_skill_input(
        job,
        contract,
        {"content_source": "github", "issue": {"number": 3}, "comments": []},
        skill_name=contract.planning_skill_name,
    )
    plan = IssuePlanArtifact(
        status="ready",
        base_sha=job.base_sha,
        execution_contract_hash=job.execution_contract_hash,
        summary="Implement the approved fix",
        plan_markdown="1. Change parser\n2. Run tests",
    )
    implementation = build_implementation_skill_input(
        job,
        contract,
        plan,
        skill_name=contract.implementation_skill_name,
    )

    assert planning.name == "issue-planning"
    assert planning.arguments["execution_contract_hash"] == contract.contract_hash
    assert implementation.name == "open-source-contribution"
    assert implementation.arguments["mode"] == "approved_implementation"
    assert implementation.arguments["automation"]["approved_plan"] == plan.plan_markdown


def test_bot_rejects_skill_input_when_job_contract_binding_changes() -> None:
    contract = _contract()
    job = _job(contract).model_copy(update={"base_sha": "d" * 40})

    with pytest.raises(ValueError, match="does not match"):
        build_planning_skill_input(job, contract, {}, skill_name="issue-planning")


@pytest.mark.parametrize(
    ("existing", "expected"),
    [(False, "start"), (True, "resume")],
)
def test_bot_session_dispatch_preserves_crash_safe_start_or_resume(existing, expected) -> None:
    calls: list[tuple[str, object]] = []

    class Conversation:
        def snapshot(self):
            return object() if existing else None

        def start(self, input):
            calls.append(("start", input))
            return "start-events"

        def resume(self):
            calls.append(("resume", None))
            return "resume-events"

    initial = build_planning_skill_input(
        _job(_contract()),
        _contract(),
        {},
        skill_name="issue-planning",
    )

    events = _bot_session_events(Conversation(), initial)  # type: ignore[arg-type]

    assert events == f"{expected}-events"
    assert calls == [(expected, initial if expected == "start" else None)]


def test_bot_applications_expose_exact_contract_tool_schemas(tmp_path) -> None:
    contract = _contract()
    job = _job(contract)
    store = BotStore(tmp_path / "bot.sqlite3")
    store.initialize()
    sessions = SqliteSessionStore(store)
    workspace = tmp_path / "workspaces" / "job"
    workspace.mkdir(parents=True)
    config_path = tmp_path / "repositories.yml"
    config_path.write_text("repositories: {}\n", encoding="utf-8")
    bot_settings = BotWorkerSettings(
        database_path=tmp_path / "bot.sqlite3",
        workspace_root=tmp_path / "workspaces",
        repositories_config=config_path,
        worker_id="worker",
    )
    repository = RepositoryBotConfig(
        image=contract.image_id,
        validation_commands=contract.validation_commands,
    )
    settings = Settings(model_id=contract.model_id)
    model_gateway = object()

    plan = _build_plan_application(
        settings=settings,
        bot_settings=bot_settings,
        store=store,
        sessions=sessions,
        model_gateway=model_gateway,  # type: ignore[arg-type]
        job=job,
        repository=repository,
        contract=contract,
        workspace=workspace,
    )
    implementation = _build_implementation_application(
        settings=settings,
        bot_settings=bot_settings,
        store=store,
        sessions=sessions,
        model_gateway=model_gateway,  # type: ignore[arg-type]
        job=job,
        repository=repository,
        contract=contract,
        workspace=workspace,
        process_runner=DisabledProcessRunner(),
        on_policy_violation=lambda _reason: None,
    )

    def schemas(application):
        context = tool_context(
            session_id="bot",
            working_directory=str(workspace),
            state_directory=str(tmp_path / "state"),
            capabilities=application._capabilities,
        )
        return {
            schema["name"]
            for schema in application._runtime.dependencies.tool_executor.registry.schemas(context)
        }

    assert schemas(plan) == contract.plan_allowed_tools
    assert schemas(implementation) == contract.implementation_allowed_tools
