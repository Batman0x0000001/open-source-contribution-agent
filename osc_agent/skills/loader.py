from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from osc_agent.skills.models import SkillDescriptor, SkillManifest


class SkillLoader:
    def __init__(self, root: Path, *, source: Literal["builtin", "user", "project"]) -> None:
        self.root = root
        self.source = source

    def discover(self) -> list[SkillDescriptor]:
        if not self.root.exists():
            return []
        descriptors: list[SkillDescriptor] = []
        for path in sorted(self.root.glob("*/SKILL.md")):
            raw_manifest = _read_frontmatter(path)
            if not isinstance(raw_manifest, dict):
                raise ValueError(f"skill frontmatter must be an object: {path}")
            # YAML 列表在外部输入边界显式规范化；核心 Pydantic 契约继续保持 strict。
            if isinstance(raw_manifest.get("allowed_tools"), list):
                raw_manifest["allowed_tools"] = frozenset(raw_manifest["allowed_tools"])
            if isinstance(raw_manifest.get("resources"), list):
                raw_manifest["resources"] = tuple(raw_manifest["resources"])
            manifest = SkillManifest.model_validate(raw_manifest)
            descriptors.append(
                SkillDescriptor(
                    manifest=manifest,
                    path=str(path.resolve()),
                    root=str(path.parent.resolve()),
                    source=self.source,
                )
            )
        return descriptors

    @staticmethod
    def read_body(descriptor: SkillDescriptor) -> str:
        raw = Path(descriptor.path).read_text(encoding="utf-8")
        first, separator, remainder = raw.partition("---")
        if first.strip() or not separator:
            raise ValueError(f"skill has no frontmatter: {descriptor.path}")
        _metadata, separator, body = remainder.partition("---")
        if not separator:
            raise ValueError(f"skill has unclosed frontmatter: {descriptor.path}")
        return body.lstrip("\r\n")


def _read_frontmatter(path: Path) -> object:
    """发现阶段只读取 YAML 头，正文保持延迟加载。"""

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"skill has no frontmatter: {path}")
        for line in handle:
            if line.strip() == "---":
                break
            lines.append(line)
        else:
            raise ValueError(f"skill has unclosed frontmatter: {path}")
    return yaml.safe_load("".join(lines)) or {}
