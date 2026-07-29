"""统一计算仓库级运行状态与会话目录。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from osc_agent.contracts import FrozenContractModel


class ApplicationStatePaths(FrozenContractModel):
    """当前仓库在用户级数据目录中的全部运行时路径。"""

    repository_root: str
    state_root: str
    repository_key: str

    @classmethod
    def for_repository(cls, repository_root: Path, *, state_root: Path | None = None) -> "ApplicationStatePaths":
        repository = repository_root.resolve()
        canonical = os.path.normcase(str(repository))
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        root = (state_root or default_state_root()).expanduser().resolve()
        return cls(repository_root=str(repository), state_root=str(root), repository_key=key)

    @property
    def repository(self) -> Path:
        return Path(self.state_root) / "repositories" / self.repository_key

    @property
    def sessions(self) -> Path:
        return self.repository / "sessions"

    @property
    def plans(self) -> Path:
        return self.repository / "plans"

    @property
    def tool_results(self) -> Path:
        return self.repository / "tool-results"

    @property
    def worktrees(self) -> Path:
        return self.repository / "worktrees"


def default_state_root() -> Path:
    override = os.environ.get("OSC_AGENT_STATE_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "osc-agent"
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "osc-agent"
    return Path.home() / ".local" / "share" / "osc-agent"
