"""评估操作本身的风险，不负责能力授权或用户审批。"""

from __future__ import annotations

from dataclasses import dataclass


DENIED_SHELL_PATTERNS = (
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "git push",
    "gh pr create",
)

APPROVAL_SHELL_PATTERNS = (
    "rm ",
    "del ",
    "erase ",
    "remove-item",
    "pip install",
    "npm install",
    "git commit",
)

LARGE_FILE_WRITE_LIMIT = 500_000


@dataclass(frozen=True)
class RiskDecision:
    action: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


def format_risk_block(decision: RiskDecision) -> str:
    """把风险判断转换为工具可直接返回的稳定文本。"""
    if decision.action == "deny":
        return f"Permission denied: {decision.reason}"
    if decision.action == "ask":
        return f"Permission required: {decision.reason}"
    return decision.reason


def assess_shell_risk(command: str) -> RiskDecision:
    """评估 shell 命令风险；审批交互由调用方负责。"""
    lowered = command.lower()
    for pattern in DENIED_SHELL_PATTERNS:
        if pattern in lowered:
            return RiskDecision("deny", f"dangerous shell command contains {pattern!r}")

    for pattern in APPROVAL_SHELL_PATTERNS:
        if pattern in lowered:
            return RiskDecision("ask", f"shell command requires explicit confirmation because it contains {pattern!r}")

    return RiskDecision("allow", "shell command allowed")


def assess_file_write_risk(path: str, content: str = "") -> RiskDecision:
    """评估文件写入规模；路径边界由 repository_boundary 单独负责。"""
    if len(content) > LARGE_FILE_WRITE_LIMIT:
        return RiskDecision("ask", "large file write requires explicit confirmation")
    return RiskDecision("allow", "file write allowed")


def risk_policy_summary() -> str:
    """根据实际风险规则生成 Prompt 摘要，避免说明与执行逻辑漂移。"""
    denied = ", ".join(DENIED_SHELL_PATTERNS)
    approval = ", ".join(APPROVAL_SHELL_PATTERNS)
    return (
        f"Shell patterns blocked without override: {denied}. "
        f"Shell patterns requiring explicit approval: {approval}. "
        f"File writes larger than {LARGE_FILE_WRITE_LIMIT:,} characters require explicit approval."
    )
