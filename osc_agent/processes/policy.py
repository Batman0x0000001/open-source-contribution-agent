"""集中处理命令分类和子进程环境过滤策略。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
import re

from osc_agent.processes.contracts import CommandKind


DEFAULT_SUBPROCESS_ENV_NAMES = frozenset(
    {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
        "TMPDIR", "USERPROFILE", "HOME", "LANG", "LANGUAGE", "LC_ALL", "TERM",
        "NO_COLOR", "PYTHONIOENCODING", "PYTHONUTF8", "VIRTUAL_ENV", "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV", "CONDA_EXE", "JAVA_HOME", "DOTNET_ROOT",
    }
)
_PROTECTED_ENV_EXACT = frozenset(
    {
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_OAUTH_TOKEN", "GITHUB_TOKEN",
        "GH_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK", "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS", "RIPGREP_CONFIG_PATH", "SSH_AUTH_SOCK",
    }
)
_PROTECTED_ENV_MARKERS = (
    "API_KEY", "AUTH_TOKEN", "AUTHORIZATION", "BEARER_TOKEN", "CLIENT_SECRET",
    "COOKIE", "CREDENTIAL", "PASSWORD", "PRIVATE_KEY", "SECRET", "SESSION_TOKEN",
)


def classify_command(command: str) -> CommandKind:
    lowered = command.casefold()
    test_patterns = (
        r"(^|\s)(pytest|py\.test|tox|nox)(\s|$)",
        r"python\s+-m\s+(pytest|unittest)",
        r"(^|\s)(npm|pnpm|yarn)\s+(run\s+)?test(\s|$)",
        r"(^|\s)cargo\s+test(\s|$)",
        r"(^|\s)go\s+test(\s|$)",
        r"(^|\s)dotnet\s+test(\s|$)",
        r"(^|\s)(mvn|mvnw|gradle|gradlew)(\.cmd)?\s+.*\btest\b",
    )
    if any(re.search(pattern, lowered) for pattern in test_patterns):
        return CommandKind.TEST
    if re.search(
        r"(^|\s)(npm|pnpm|yarn)\s+(run\s+)?build(\s|$)|(^|\s)(cargo|go|dotnet)\s+build(\s|$)",
        lowered,
    ):
        return CommandKind.BUILD
    if re.search(
        r"(^|\s)(ruff|eslint|pylint)(\s|$)|(^|\s)(npm|pnpm|yarn)\s+(run\s+)?lint(\s|$)",
        lowered,
    ):
        return CommandKind.LINT
    if re.search(
        r"(^|\s)(mypy|pyright|tsc)(\s|$)|(^|\s)(npm|pnpm|yarn)\s+(run\s+)?typecheck(\s|$)",
        lowered,
    ):
        return CommandKind.TYPECHECK
    return CommandKind.OTHER


def build_subprocess_environment(
    additional_names: Iterable[str] = (),
    *,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """只复制运行必需和用户显式允许的非敏感环境变量。"""

    available = os.environ if source is None else source
    requested = {name.strip().upper() for name in additional_names if name.strip()}
    names = DEFAULT_SUBPROCESS_ENV_NAMES | requested
    environment: dict[str, str] = {}
    for name, value in available.items():
        canonical = name.upper()
        if canonical in names and not is_protected_environment_name(canonical):
            environment[name] = value
    return environment


def is_protected_environment_name(name: str) -> bool:
    canonical = name.strip().upper()
    return canonical in _PROTECTED_ENV_EXACT or any(
        marker in canonical for marker in _PROTECTED_ENV_MARKERS
    )

