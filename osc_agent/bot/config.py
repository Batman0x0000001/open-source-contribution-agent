"""加载并校验 GitHub Bot、Worker 与仓库策略配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from osc_agent.bot.models import RepositoryBotCatalog


class BotSettings(BaseSettings):
    """Bot 进程显式配置；本地 CLI 不加载这些必需项。"""

    model_config = SettingsConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    github_app_id: int = Field(gt=0, validation_alias="OSC_AGENT_GITHUB_APP_ID")
    github_app_private_key_path: Path = Field(validation_alias="OSC_AGENT_GITHUB_APP_PRIVATE_KEY_PATH")
    github_webhook_secret: str = Field(min_length=16, validation_alias="OSC_AGENT_GITHUB_WEBHOOK_SECRET")
    github_public_url: str = Field(pattern=r"^https://", validation_alias="OSC_AGENT_GITHUB_PUBLIC_URL")
    database_path: Path = Field(validation_alias="OSC_AGENT_BOT_DATABASE_PATH")
    workspace_root: Path = Field(validation_alias="OSC_AGENT_BOT_WORKSPACE_ROOT")
    repositories_config: Path = Field(validation_alias="OSC_AGENT_BOT_REPOSITORIES_CONFIG")
    bind_host: str = Field(default="127.0.0.1", validation_alias="OSC_AGENT_BOT_BIND_HOST")
    bind_port: int = Field(default=8080, ge=1, le=65_535, validation_alias="OSC_AGENT_BOT_BIND_PORT")
    github_commit_name: str = Field(min_length=1, validation_alias="OSC_AGENT_GITHUB_COMMIT_NAME")
    github_commit_email: str = Field(min_length=3, validation_alias="OSC_AGENT_GITHUB_COMMIT_EMAIL")
    plan_approval_days: int = Field(default=7, ge=1, le=30, validation_alias="OSC_AGENT_BOT_PLAN_APPROVAL_DAYS")
    completed_workspace_hours: int = Field(default=24, ge=1, le=168, validation_alias="OSC_AGENT_BOT_WORKSPACE_RETENTION_HOURS")
    audit_retention_days: int = Field(default=30, ge=1, le=365, validation_alias="OSC_AGENT_BOT_AUDIT_RETENTION_DAYS")
    model_id: str = Field(min_length=1, validation_alias="MODEL_ID")

    @model_validator(mode="after")
    def protect_control_state_from_workspace_mounts(self) -> "BotSettings":
        workspace = self.workspace_root.resolve()
        for name, path in (
            ("GitHub App private key", self.github_app_private_key_path),
            ("SQLite database", self.database_path),
            ("repository configuration", self.repositories_config),
        ):
            if path.resolve().is_relative_to(workspace):
                raise ValueError(f"{name} must be outside the bot workspace root")
        return self


class BotWorkerSettings(BaseSettings):
    """Worker 不接受 GitHub App 私钥、Webhook Secret 或 commit 身份。"""

    model_config = SettingsConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    database_path: Path = Field(validation_alias="OSC_AGENT_BOT_DATABASE_PATH")
    workspace_root: Path = Field(validation_alias="OSC_AGENT_BOT_WORKSPACE_ROOT")
    repositories_config: Path = Field(validation_alias="OSC_AGENT_BOT_REPOSITORIES_CONFIG")
    worker_id: str = Field(min_length=1, validation_alias="OSC_AGENT_BOT_WORKER_ID")
    max_concurrent_plans: int = Field(default=2, ge=1, le=8, validation_alias="OSC_AGENT_BOT_MAX_CONCURRENT_PLANS")
    max_concurrent_implementations: int = Field(default=1, ge=1, le=4, validation_alias="OSC_AGENT_BOT_MAX_CONCURRENT_IMPLEMENTATIONS")
    shutdown_timeout_seconds: int = Field(default=30, ge=1, le=300, validation_alias="OSC_AGENT_BOT_SHUTDOWN_TIMEOUT_SECONDS")

    @model_validator(mode="after")
    def protect_worker_state_from_workspace_mounts(self) -> "BotWorkerSettings":
        workspace = self.workspace_root.resolve()
        for name, path in (
            ("SQLite database", self.database_path),
            ("repository configuration", self.repositories_config),
        ):
            if path.resolve().is_relative_to(workspace):
                raise ValueError(f"{name} must be outside the bot workspace root")
        return self


class BotMaintenanceSettings(BaseSettings):
    """Bot 状态维护只加载所需路径和保留策略，不加载 GitHub 凭据。"""

    model_config = SettingsConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    database_path: Path = Field(validation_alias="OSC_AGENT_BOT_DATABASE_PATH")
    workspace_root: Path = Field(validation_alias="OSC_AGENT_BOT_WORKSPACE_ROOT")
    bind_port: int = Field(default=8080, ge=1, le=65_535, validation_alias="OSC_AGENT_BOT_BIND_PORT")
    completed_workspace_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        validation_alias="OSC_AGENT_BOT_WORKSPACE_RETENTION_HOURS",
    )
    audit_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        validation_alias="OSC_AGENT_BOT_AUDIT_RETENTION_DAYS",
    )

    @model_validator(mode="after")
    def protect_state_from_recursive_reset(self) -> "BotMaintenanceSettings":
        workspace = self.workspace_root.resolve()
        database = self.database_path.resolve()
        if workspace == Path(workspace.anchor):
            raise ValueError("Bot workspace root cannot be a filesystem root")
        if database.is_relative_to(workspace):
            raise ValueError("SQLite database must be outside the Bot workspace root")
        return self


def load_repository_catalog(path: Path) -> RepositoryBotCatalog:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read bot repositories config: {exc}") from exc
    if isinstance(raw, dict) and "runtime" in raw:
        unknown = set(raw) - {"runtime", "repositories"}
        if unknown:
            raise ValueError(f"unknown production config sections: {', '.join(sorted(unknown))}")
        raw = {"repositories": raw.get("repositories")}
    return RepositoryBotCatalog.model_validate(raw)
