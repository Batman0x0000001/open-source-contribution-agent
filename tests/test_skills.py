from __future__ import annotations

from types import SimpleNamespace

from osc_agent.agent_loop import TOOLS, agent_loop, build_tool_handlers
from osc_agent.config import Settings
from osc_agent.harness.capabilities import AgentCapabilityScope
from osc_agent.harness.prompt import assemble_system_prompt, update_context
from osc_agent.skills.registry import (
    list_skill_catalog,
    load_skill,
    scan_skills,
    suggest_skills_for_repo,
)
from osc_agent.tools.repo import inspect_repo


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def test_scan_skills_reads_frontmatter():
    skills = scan_skills()

    assert "python" in skills
    assert skills["python"].description.startswith("Python packaging")


def test_load_skill_returns_full_content_and_unknown_name_is_safe():
    docs = load_skill("docs")

    assert "# Docs Skill" in docs
    assert load_skill("../docs") == "Skill not found: ../docs"


def test_new_skill_can_be_added_without_agent_loop_changes(tmp_path):
    skill_dir = tmp_path / "security"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nschema_version: 1\nname: security\ndescription: Security review guidance.\n"
        "version: 1.0.0\napplies_to: [security]\nrequired_tools: [read_file]\n"
        "permissions: [read]\ninput_contract: Repository evidence.\noutput_contract: Review guidance.\n"
        "---\n\n# Security\n",
        encoding="utf-8",
    )

    assert "- security: Security review guidance." in list_skill_catalog(tmp_path)
    assert "# Security" in load_skill("security", skills_root=tmp_path)


def test_invalid_skill_contract_reports_missing_fields(tmp_path):
    skill_dir = tmp_path / "invalid"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: invalid\n---\n", encoding="utf-8")

    try:
        scan_skills(tmp_path)
    except ValueError as exc:
        assert "missing fields" in str(exc)
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("invalid skill must be rejected")


def test_skill_contract_cannot_grant_missing_tool_or_permission():
    assert "missing tools" in load_skill("python", available_tools=set(), granted_permissions={"read"})
    assert "permissions not granted" in load_skill(
        "python",
        available_tools={"read_file", "glob", "bash"},
        granted_permissions={"read"},
    )


def test_system_prompt_contains_catalog_not_full_skill_body(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    context = update_context(
        repo_root=tmp_path,
        objective="Inspect repository skills",
        current_instruction="inspect skills",
        enabled_tools=["load_skill"],
        capabilities=AgentCapabilityScope.unrestricted(),
    )
    prompt = assemble_system_prompt(context)

    assert "Skills available:" in prompt
    assert "- docs:" in prompt
    assert "# Docs Skill" not in prompt
    assert "Use load_skill(name)" in prompt


def test_repo_inspect_suggests_relevant_skills(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    suggestions = suggest_skills_for_repo(tmp_path)
    overview = inspect_repo(repo_root=tmp_path)

    assert {"python", "docs", "tests", "open-source"} <= set(suggestions)
    assert "Suggested skills:" in overview
    assert "- python" in overview


class LoadSkillMessages:
    def __init__(self) -> None:
        self.calls = 0
        self.first_system = ""
        self.first_tools: list[dict] = []

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.first_system = kwargs["system"]
            self.first_tools = kwargs["tools"]
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="load_skill",
                        id="toolu_skill_1",
                        input={"name": "docs"},
                    )
                ],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        )


class FakeClient:
    def __init__(self) -> None:
        self.messages = LoadSkillMessages()


def test_agent_loop_exposes_load_skill_tool(tmp_path):
    messages = [{"role": "user", "content": "load docs skill"}]
    client = FakeClient()

    response = agent_loop(
        messages,
        client=client,
        settings=_settings(),
        repo_root=tmp_path,
    )

    assert response.stop_reason == "end_turn"
    assert "load_skill" in {tool["name"] for tool in TOOLS}
    assert "load_skill" in build_tool_handlers(tmp_path)
    assert "# Docs Skill" in messages[2]["content"][0]["content"]
    assert "# Docs Skill" not in client.messages.first_system
    assert "load_skill" in {tool["name"] for tool in client.messages.first_tools}
