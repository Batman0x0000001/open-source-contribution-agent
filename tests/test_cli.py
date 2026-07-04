from __future__ import annotations

from typer.testing import CliRunner

import osc_agent.cli as cli

runner = CliRunner()


def test_inspect_command_prints_repo_map(tmp_path):
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["inspect", "--repo", str(tmp_path)])

    assert result.exit_code == 0
    assert "Repository:" in result.output
    assert "README.md" in result.output


def test_draft_pr_command_uses_pr_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "draft_pr", lambda repo_root: f"Title: Draft for {repo_root.name}")

    result = runner.invoke(cli.app, ["draft-pr", "--repo", str(tmp_path)])

    assert result.exit_code == 0
    assert f"Title: Draft for {tmp_path.name}" in result.output
