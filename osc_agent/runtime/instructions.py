from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from osc_agent.runtime.models import InstructionDocument, RepositoryInstructionState
from osc_agent.tools.path_policy import safe_repo_path


MAX_INSTRUCTION_CHARS = 40_000
_INSTRUCTION_FILES = (("AGENTS.md", "agents"), ("CLAUDE.md", "claude"))


class RepositoryInstructionResolver:
    """只在当前仓库内发现项目指令；不解释自然语言优先级。"""

    def activate_root(self, working_directory: Path) -> RepositoryInstructionState:
        return self.activate_for_path(
            working_directory,
            ".",
            RepositoryInstructionState(),
        )

    def activate_for_path(
        self,
        working_directory: Path,
        relative_path: str,
        state: RepositoryInstructionState,
    ) -> RepositoryInstructionState:
        root = working_directory.resolve()
        target = safe_repo_path(root, relative_path)
        directory = root if relative_path == "." else (target if target.is_dir() else target.parent)
        if directory != root and root not in directory.parents:
            raise ValueError("instruction target escapes the repository")

        discovered = set(state.active_paths)
        for scope in _scopes(root, directory):
            for filename, _kind in _INSTRUCTION_FILES:
                candidate = scope / filename
                if not candidate.exists():
                    continue
                resolved = candidate.resolve()
                if resolved != root and root not in resolved.parents:
                    raise ValueError(f"instruction file escapes the repository: {candidate}")
                if not resolved.is_file():
                    continue
                discovered.add(resolved.relative_to(root).as_posix())
        return RepositoryInstructionState(active_paths=sorted(discovered, key=_path_depth))

    def load(
        self,
        working_directory: Path,
        state: RepositoryInstructionState,
    ) -> list[InstructionDocument]:
        root = working_directory.resolve()
        documents: list[InstructionDocument] = []
        for relative in state.active_paths:
            path = safe_repo_path(root, relative)
            resolved = path.resolve()
            if not resolved.is_file():
                continue
            name = resolved.name.casefold()
            if name not in {"agents.md", "claude.md"}:
                raise ValueError(f"unsupported instruction file: {relative}")
            try:
                raw_bytes = resolved.read_bytes()
                raw = raw_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not raw.strip():
                continue
            content = raw
            if len(raw) > MAX_INSTRUCTION_CHARS:
                content = (
                    raw[:MAX_INSTRUCTION_CHARS]
                    + "\n\n[Instruction file truncated at 40000 characters.]"
                )
            documents.append(
                InstructionDocument(
                    path=relative,
                    kind="agents" if name == "agents.md" else "claude",
                    scope_directory=resolved.parent.relative_to(root).as_posix()
                    if resolved.parent != root
                    else ".",
                    content_hash=sha256(raw_bytes).hexdigest(),
                    content=content,
                )
            )
        return sorted(
            documents,
            key=lambda item: (_path_depth(item.path), item.path.casefold()),
        )


def _scopes(root: Path, directory: Path) -> list[Path]:
    relative = directory.relative_to(root)
    scopes = [root]
    current = root
    for part in relative.parts:
        current = current / part
        scopes.append(current)
    return scopes


def _path_depth(value: str) -> tuple[int, str]:
    return len(Path(value).parts), value.casefold()
