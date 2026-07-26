from __future__ import annotations

from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import SkillDescriptor


class SkillCatalog:
    """合并仅含元数据的 Skill 目录，并保留内置 Skill 名称。"""

    def __init__(self, loaders: list[SkillLoader]) -> None:
        self._descriptors: dict[str, SkillDescriptor] = {}
        self._blocked_overrides: list[SkillDescriptor] = []
        priority = {"builtin": 0, "user": 1, "project": 2}
        for loader in sorted(loaders, key=lambda item: priority[item.source]):
            for descriptor in loader.discover():
                current = self._descriptors.get(descriptor.manifest.name)
                if (
                    descriptor.source != "builtin"
                    and current is not None
                    and current.source == "builtin"
                ):
                    self._blocked_overrides.append(descriptor)
                    continue
                self._descriptors[descriptor.manifest.name] = descriptor

    def get(self, name: str) -> SkillDescriptor | None:
        return self._descriptors.get(name)

    def list(self) -> list[SkillDescriptor]:
        return [self._descriptors[name] for name in sorted(self._descriptors)]

    def blocked_overrides(self) -> list[SkillDescriptor]:
        return sorted(
            self._blocked_overrides,
            key=lambda item: (item.manifest.name, item.source, item.path),
        )
