"""从 SKILL.md 发现清单，并按需异步读取正文。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from osc_agent.skills.models import (
    SkillDescriptor,
    SkillDiagnostic,
    SkillDiscovery,
    SkillManifest,
    SkillSource,
)


class SkillLoader:
    def __init__(self, root: Path, *, source: SkillSource) -> None:
        self.root = root
        self.source = source

    def discover(self) -> SkillDiscovery:
        if not self.root.exists():
            return SkillDiscovery()
        descriptors: list[SkillDescriptor] = []
        diagnostics: list[SkillDiagnostic] = []
        for path in sorted(self.root.glob("*/SKILL.md")):
            try:
                descriptors.append(self._load_descriptor(path))
            except Exception as exc:  # noqa: BLE001 - 外部 Skill 必须逐个隔离诊断。
                if self.source == "builtin":
                    raise ValueError(f"invalid built-in Skill {path}: {exc}") from exc
                diagnostics.append(
                    SkillDiagnostic(
                        code="INVALID_SKILL_MANIFEST",
                        message=str(exc),
                        path=str(path.resolve()),
                        source=self.source,
                        skill_name=path.parent.name,
                    )
                )
        return SkillDiscovery(
            descriptors=tuple(descriptors),
            diagnostics=tuple(diagnostics),
        )

    def _load_descriptor(self, path: Path) -> SkillDescriptor:
        raw_manifest = _read_frontmatter(path)
        if not isinstance(raw_manifest, dict):
            raise ValueError("skill frontmatter must be an object")
        # YAML 列表在外部输入边界规范化，核心 Pydantic 契约继续保持 strict。
        resolved_root = self.root.resolve()
        resolved_path = path.resolve(strict=True)
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(f"skill path escapes loader root: {path}")
        for field in ("allowed_tools", "product_tools"):
            if isinstance(raw_manifest.get(field), list):
                raw_manifest[field] = frozenset(raw_manifest[field])
        if isinstance(raw_manifest.get("resources"), list):
            raw_manifest["resources"] = tuple(raw_manifest["resources"])
        completion = raw_manifest.get("completion")
        if isinstance(completion, dict):
            for field in ("required_evidence", "waivable_evidence"):
                if isinstance(completion.get(field), list):
                    completion[field] = frozenset(completion[field])
        manifest = SkillManifest.model_validate(raw_manifest)
        return SkillDescriptor(
            manifest=manifest,
            path=str(resolved_path),
            root=str(resolved_path.parent),
            source=self.source,
        )

    @staticmethod
    async def read_body(descriptor: SkillDescriptor) -> str:
        return await asyncio.to_thread(_read_body, Path(descriptor.path))


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


def _read_body(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"skill has no frontmatter: {path}")
        for line in handle:
            if line.strip() == "---":
                return handle.read().lstrip("\r\n")
    raise ValueError(f"skill has unclosed frontmatter: {path}")
