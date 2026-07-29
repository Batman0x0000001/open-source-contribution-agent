"""合并 Skill 来源，并集中处理发现和调用策略。"""

from __future__ import annotations

from pathlib import Path

from osc_agent.skills.loader import SkillLoader
from osc_agent.skills.models import SkillDescriptor, SkillDiagnostic, SkillTrigger


class SkillCatalog:
    """合并清单元数据；项目可覆盖用户 Skill，但不能覆盖内置 Skill。"""

    def __init__(self, loaders: list[SkillLoader]) -> None:
        self._descriptors: dict[str, SkillDescriptor] = {}
        self._diagnostics: list[SkillDiagnostic] = []
        priority = {"builtin": 0, "user": 1, "project": 2}
        for loader in sorted(loaders, key=lambda item: priority[item.source]):
            discovery = loader.discover()
            self._diagnostics.extend(discovery.diagnostics)
            for descriptor in discovery.descriptors:
                current = self._descriptors.get(descriptor.manifest.name)
                if (
                    descriptor.source != "builtin"
                    and current is not None
                    and current.source == "builtin"
                ):
                    self._diagnostics.append(
                        SkillDiagnostic(
                            code="BUILTIN_SKILL_NAME_RESERVED",
                            message="built-in Skill name is reserved",
                            path=descriptor.path,
                            source=descriptor.source,
                            skill_name=descriptor.manifest.name,
                        )
                    )
                    continue
                self._descriptors[descriptor.manifest.name] = descriptor

    def get(self, name: str) -> SkillDescriptor | None:
        return self._descriptors.get(name)

    def resolve(self, name: str, trigger: SkillTrigger) -> SkillDescriptor | None:
        descriptor = self.get(name)
        if descriptor is None:
            return None
        if trigger == "user" and not _is_user_invocable(descriptor):
            return None
        if trigger == "model" and not _is_model_invocable(descriptor):
            return None
        return descriptor

    def list(self) -> list[SkillDescriptor]:
        return [self._descriptors[name] for name in sorted(self._descriptors)]

    def list_user_invocable(self) -> list[SkillDescriptor]:
        return [item for item in self.list() if _is_user_invocable(item)]

    def list_model_invocable(self) -> list[SkillDescriptor]:
        return [
            item
            for item in self.list()
            if _is_model_invocable(item)
        ]

    def diagnostics(self) -> list[SkillDiagnostic]:
        return sorted(
            self._diagnostics,
            key=lambda item: (item.source, item.skill_name or "", item.path, item.code),
        )

    def validate_allowed_tools(self, known_tools: set[str]) -> None:
        """在完整产品工具集合已知时验证声明；外部错误只移除对应 Skill。"""

        for name, descriptor in list(self._descriptors.items()):
            unknown = sorted(descriptor.manifest.allowed_tools - known_tools)
            if not unknown:
                continue
            message = "unknown allowed_tools: " + ", ".join(unknown)
            if descriptor.source == "builtin":
                raise ValueError(f"invalid built-in Skill {name}: {message}")
            self._diagnostics.append(
                SkillDiagnostic(
                    code="UNKNOWN_SKILL_TOOL",
                    message=message,
                    path=descriptor.path,
                    source=descriptor.source,
                    skill_name=name,
                )
            )
            del self._descriptors[name]


def build_skill_catalog(repository_root: Path) -> SkillCatalog:
    """从内置、用户和项目三个固定来源构建 Skill Catalog。"""

    builtin = Path(__file__).resolve().parent / "builtins"
    return SkillCatalog(
        [
            SkillLoader(builtin, source="builtin"),
            SkillLoader(Path.home() / ".osc_agent" / "skills", source="user"),
            SkillLoader(repository_root.resolve() / ".osc_agent" / "skills", source="project"),
        ]
    )


def _is_user_invocable(descriptor: SkillDescriptor) -> bool:
    return descriptor.manifest.user_invocable


def _is_model_invocable(descriptor: SkillDescriptor) -> bool:
    return descriptor.source == "builtin" and not descriptor.manifest.disable_model_invocation
