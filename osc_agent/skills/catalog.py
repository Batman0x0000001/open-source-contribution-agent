from __future__ import annotations

from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import SkillDescriptor


class SkillCatalog:
    """按 builtin < user < project 合并仅含元数据的 Skill 目录。"""

    def __init__(self, loaders: list[SkillLoader]) -> None:
        self._descriptors: dict[str, SkillDescriptor] = {}
        priority = {"builtin": 0, "user": 1, "project": 2}
        for loader in sorted(loaders, key=lambda item: priority[item.source]):
            for descriptor in loader.discover():
                self._descriptors[descriptor.manifest.name] = descriptor

    def get(self, name: str) -> SkillDescriptor | None:
        return self._descriptors.get(name)

    def list(self) -> list[SkillDescriptor]:
        return [self._descriptors[name] for name in sorted(self._descriptors)]
