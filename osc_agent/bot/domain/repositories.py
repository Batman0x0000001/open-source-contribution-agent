"""定义 Bot 仓库执行策略与仓库目录。"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from osc_agent.contracts import FrozenContractModel


class RepositoryBotConfig(FrozenContractModel):
    enabled: bool = True
    image: str = Field(
        pattern=r"^sha256:[0-9a-f]{64}$",
        description="Worker 已预构建并固定的不可变 Docker image ID。",
    )
    validation_commands: tuple[str, ...] = Field(min_length=1, max_length=20)
    denied_paths: tuple[str, ...] = (
        ".git/**",
        ".github/workflows/**",
        ".gitmodules",
        ".osc_agent/**",
    )
    max_changed_files: int = Field(default=100, ge=1, le=1_000)
    max_patch_bytes: int = Field(default=2_000_000, ge=1, le=50_000_000)
    command_timeout_seconds: int = Field(default=1_800, ge=1, le=7_200)
    container_cpus: float = Field(default=2.0, gt=0, le=32)
    container_memory: str = Field(default="4g", pattern=r"^[1-9][0-9]*(?:m|g)$")
    container_pids: int = Field(default=256, ge=16, le=4_096)
    pull_request_mode: Literal["draft", "ready"] = "draft"

    @field_validator("validation_commands", "denied_paths", mode="before")
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_commands_and_paths(self) -> "RepositoryBotConfig":
        if any(not command.strip() for command in self.validation_commands):
            raise ValueError("validation commands must be non-empty")
        from osc_agent.processes.contracts import CommandKind
        from osc_agent.processes.policy import classify_command

        if not any(classify_command(command) == CommandKind.TEST for command in self.validation_commands):
            raise ValueError("validation commands must include at least one recognized test command")
        for pattern in self.denied_paths:
            normalized = pattern.replace("\\", "/")
            if not normalized or PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
                raise ValueError("denied paths must be repository-relative patterns")
        return self


class RepositoryBotCatalog(FrozenContractModel):
    repositories: dict[str, RepositoryBotConfig]

    @model_validator(mode="after")
    def validate_names(self) -> "RepositoryBotCatalog":
        for name in self.repositories:
            parts = name.split("/")
            if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
                raise ValueError(f"invalid GitHub repository name: {name}")
        return self
