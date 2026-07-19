from __future__ import annotations

import pytest

from osc_agent.tools.repo import (
    analyze_issue_code_candidates,
    _is_safe_repo_file,
    analyze_architecture_dimensions,
    analyze_python_repository,
    detect_entrypoints,
    detect_repository_profile,
    find_functions,
    repo_tree,
)


def test_issue_terms_map_to_symbol_level_code_candidates(tmp_path):
    (tmp_path / "retry.py").write_text(
        "class RetryManager:\n"
        "    def execute_with_retry(self):\n"
        "        raise RetryError('retry exhausted')\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text("def render_docs():\n    return 'ok'\n", encoding="utf-8")

    candidates = analyze_issue_code_candidates(
        repo_root=tmp_path,
        issue={
            "title": "execute_with_retry raises RetryError",
            "body": "The RetryManager should recover instead of reporting retry exhausted.",
        },
    )

    assert candidates[0]["file"] == "retry.py"
    assert {item["name"] for item in candidates[0]["symbols"]} >= {"RetryManager", "execute_with_retry"}
    assert any("Issue references symbol execute_with_retry" in reason for reason in candidates[0]["reasons"])


def test_repo_tree_limits_depth_and_skips_cache(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.py").write_text("def run_agent():\n    pass\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "noise.pyc").write_text("x", encoding="utf-8")

    output = repo_tree(repo_root=tmp_path, depth=3)

    assert "src/" in output
    assert "mod.py" in output
    assert "__pycache__" not in output


def test_detect_entrypoints_and_python_functions(tmp_path):
    (tmp_path / "cli.py").write_text("class Agent:\n    pass\n\ndef plan_task():\n    pass\n", encoding="utf-8")

    assert detect_entrypoints(repo_root=tmp_path) == ["cli.py"]
    symbols = find_functions(repo_root=tmp_path, query="plan")

    assert symbols == [{"file": "cli.py", "name": "plan_task", "kind": "function"}]


def test_analyze_architecture_dimensions_marks_missing_locations(tmp_path):
    rows = analyze_architecture_dimensions(repo_root=tmp_path)

    assert len(rows) == 7
    assert any(row["location"] == "未定位到具体实现" for row in rows)


def test_python_analysis_builds_import_reference_test_and_call_evidence(tmp_path):
    (tmp_path / "agent.py").write_text(
        "import json\n\ndef helper():\n    return 1\n\ndef run_agent():\n    return helper()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_agent.py").write_text(
        "from agent import run_agent\n\ndef test_run_agent():\n    assert run_agent() == 1\n",
        encoding="utf-8",
    )

    analysis = analyze_python_repository(repo_root=tmp_path)

    assert analysis["imports"]["agent.py"] == ["json"]
    assert any(item["name"] == "run_agent" and item["content_hash"] for item in analysis["definitions"])
    assert analysis["references"]["helper"]
    assert analysis["test_mapping"]["agent.py"] == ["tests/test_agent.py"]
    assert "helper" in analysis["call_expansion"]["agent.py::run_agent"]


def test_repository_profile_accepts_agent_llm_python_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='llm-agent'\n", encoding="utf-8")
    (tmp_path / "agent.py").write_text("class Agent:\n    pass\n", encoding="utf-8")

    profile = detect_repository_profile(repo_root=tmp_path)

    assert profile["supported"] is True
    assert profile["language"] == "python"


def test_repository_analysis_does_not_follow_external_file_symlinks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def secret_function():\n    return 'secret'\n", encoding="utf-8")
    link = repo / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are not available: {exc}")

    assert find_functions(repo_root=repo, query="secret") == []
    analysis = analyze_python_repository(repo_root=repo)
    assert "linked.py" not in analysis["imports"]
    assert detect_repository_profile(repo_root=repo)["python_file_count"] == 0


def test_repository_file_boundary_rejects_external_paths(tmp_path):
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")

    assert _is_safe_repo_file(repo, outside) is False


def test_repository_index_reports_budget_truncation(tmp_path):
    from osc_agent.tools.repo import RepositoryIndex

    for index in range(3):
        (tmp_path / f"file_{index}.py").write_text("pass\n", encoding="utf-8")

    repository_index = RepositoryIndex.build(tmp_path, max_files=2)

    assert len(repository_index.paths) == 2
    assert repository_index.truncated is True
