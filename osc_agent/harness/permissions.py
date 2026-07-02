"""
Agent 准备执行操作
        ↓
权限检查
        ↓
allow → 执行
deny  → 拒绝
ask   → 暂停，需要用户确认
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PermissionDecision:
    action: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


def allow(reason: str = "allowed") -> PermissionDecision:
    return PermissionDecision(action="allow", reason=reason)


def deny(reason: str) -> PermissionDecision:
    return PermissionDecision(action="deny", reason=reason)


def ask(reason: str) -> PermissionDecision:
    return PermissionDecision(action="ask", reason=reason)


def safe_repo_path(repo_root: Path, path: str) -> Path:
    """解析目标路径，并确保最终路径没有逃出 repo root。"""
    root = repo_root.resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes repository: {path}")
    return target


def format_blocked(decision: PermissionDecision) -> str:
    """把权限决策转换成工具可直接返回给模型的文本。"""
    if decision.action == "deny":
        return f"Permission denied: {decision.reason}"
    if decision.action == "ask":
        return f"Permission required: {decision.reason}"
    return decision.reason


def check_shell_command(command: str) -> PermissionDecision:
    """检查 shell 命令风险；S03 阶段只阻止或要求确认，不做审批交互。"""
    lowered = command.lower()
    deny_patterns = [
        "rm -rf /",
        "sudo",
        "shutdown",
        "reboot",
        "mkfs",
        "dd if=",
        "git push",
        "gh pr create",
    ]
    for pattern in deny_patterns:
        if pattern in lowered:
            return deny(f"dangerous shell command contains {pattern!r}")

    ask_patterns = [
        "rm ",
        "del ",
        "erase ",
        "remove-item",
        "pip install",
        "npm install",
        "git commit",
    ]
    for pattern in ask_patterns:
        if pattern in lowered:
            return ask(f"shell command requires explicit confirmation because it contains {pattern!r}")

    return allow("shell command allowed")


def check_file_write(path: str, content: str = "") -> PermissionDecision:
    """检查文件写入规模；路径边界由 safe_repo_path 单独负责。"""
    if len(content) > 500_000:
        return ask("large file write requires explicit confirmation")
    return allow("file write allowed")
