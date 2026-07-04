from __future__ import annotations

from osc_agent.tools.repo import analyze_architecture_dimensions, detect_entrypoints, find_functions, repo_tree


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
