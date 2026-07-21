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


def test_draft_pr_is_only_available_in_contribute_command_group():
    root_result = runner.invoke(cli.app, ["--help"])
    contribute_result = runner.invoke(cli.app, ["contribute", "--help"])

    assert root_result.exit_code == 0
    assert "draft-pr" not in root_result.output
    assert contribute_result.exit_code == 0
    assert "draft-pr" in contribute_result.output
