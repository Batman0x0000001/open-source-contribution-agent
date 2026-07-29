"""验证 Skill 清单发现、错误隔离和延迟正文读取。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from osc_agent.skills.loader import SkillLoader


def _write_skill(root: Path, name: str, *, extra: str = "", body: str = "Method.") -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""---
name: {name}
description: {name} description
when_to_use: Use {name} for tests
{extra}---
{body}
""",
        encoding="utf-8",
    )
    return path


def test_loader_keeps_body_lazy_and_reads_it_off_the_event_loop(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "review", body="Original body.")
    discovery = SkillLoader(tmp_path, source="project").discover()
    path.write_text(path.read_text(encoding="utf-8").replace("Original", "Updated"), encoding="utf-8")

    body = asyncio.run(SkillLoader.read_body(discovery.descriptors[0]))

    assert "Updated body." in body


def test_invalid_external_skill_is_skipped_without_hiding_valid_sibling(tmp_path: Path) -> None:
    _write_skill(tmp_path, "valid")
    _write_skill(tmp_path, "legacy", extra="input_schema: {type: object}\n")

    discovery = SkillLoader(tmp_path, source="project").discover()

    assert [item.manifest.name for item in discovery.descriptors] == ["valid"]
    assert discovery.diagnostics[0].code == "INVALID_SKILL_MANIFEST"
    assert discovery.diagnostics[0].skill_name == "legacy"


def test_invalid_builtin_skill_fails_fast(tmp_path: Path) -> None:
    _write_skill(tmp_path, "legacy", extra="context: fork\n")

    with pytest.raises(ValueError, match="invalid built-in Skill"):
        SkillLoader(tmp_path, source="builtin").discover()


def test_body_delimiter_must_be_a_standalone_frontmatter_line(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "review", body="Expected body.")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "description: review description",
            "description: review---description",
        ),
        encoding="utf-8",
    )

    discovery = SkillLoader(tmp_path, source="project").discover()
    body = asyncio.run(SkillLoader.read_body(discovery.descriptors[0]))

    assert body == "Expected body.\n"


def test_external_skill_directory_cannot_escape_loader_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    outside = tmp_path / "outside"
    _write_skill(outside, "target")
    root.mkdir()
    link = root / "escaped"
    try:
        os.symlink(outside / "target", link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    discovery = SkillLoader(root, source="project").discover()

    assert discovery.descriptors == ()
    assert discovery.diagnostics[0].code == "INVALID_SKILL_MANIFEST"
    assert "escapes" in discovery.diagnostics[0].message
