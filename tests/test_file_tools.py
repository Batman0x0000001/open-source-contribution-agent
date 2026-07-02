from __future__ import annotations

from types import SimpleNamespace

from osc_agent.agent_loop import agent_loop
from osc_agent.config import Settings
from osc_agent.tools.files import edit_file, glob_files, read_file, write_file
from osc_agent.tools.repo import inspect_repo


def test_file_tools_read_write_edit_and_glob(tmp_path):
    assert write_file(repo_root=tmp_path, path="docs/example.md", content="alpha beta beta") == (
        "Wrote docs/example.md"
    )

    assert read_file(repo_root=tmp_path, path="docs/example.md", limit=5) == "alpha"
    assert edit_file(
        repo_root=tmp_path,
        path="docs/example.md",
        old_text="beta",
        new_text="gamma",
    ) == "Edited docs/example.md"
    assert read_file(repo_root=tmp_path, path="docs/example.md") == "alpha gamma beta"
    assert glob_files(repo_root=tmp_path, pattern="docs/*.md") == "docs/example.md"


def test_edit_file_returns_error_when_old_text_is_missing(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    output = edit_file(
        repo_root=tmp_path,
        path="README.md",
        old_text="missing",
        new_text="replacement",
    )

    assert output == "Error: old_text not found in README.md"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hello"


def test_inspect_repo_reports_key_files_and_tests(tmp_path):
    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    output = inspect_repo(repo_root=tmp_path)

    assert "README.md" in output
    assert "pyproject.toml" in output
    assert "tests" in output


class FakeMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_names = {tool["name"] for tool in kwargs["tools"]}
            assert {"bash", "read_file", "edit_file", "git_status", "inspect_repo"} <= tool_names
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="read_file",
                        id="toolu_1",
                        input={"path": "README.md"},
                    )
                ],
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        )


class FakeClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


def test_agent_loop_dispatches_tools_through_handler_map(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    messages = [{"role": "user", "content": "read README"}]
    settings = Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )

    agent_loop(messages, client=FakeClient(), settings=settings, repo_root=tmp_path)

    tool_results = messages[2]["content"]
    assert isinstance(tool_results, list)
    first_result = tool_results[0]
    assert isinstance(first_result, dict)
    assert first_result["content"] == "hello"
