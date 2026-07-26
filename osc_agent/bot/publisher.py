from __future__ import annotations

import asyncio
from pathlib import Path, PurePosixPath
import shutil

from osc_agent.bot.config import BotSettings
from osc_agent.bot.github_app import GitHubControlClient, basic_git_auth_header
from osc_agent.bot.models import (
    BotJob,
    OutboxEvent,
    RepositoryBotConfig,
    validate_implementation_approval,
)
from osc_agent.bot.policy import _matches
from osc_agent.bot.store import BotStore
from osc_agent.tools.git import git_snapshot, git_workspace_fingerprint
from osc_agent.tools.process import build_subprocess_environment
from osc_agent.bot.policy import ConfiguredValidationStopHook
from osc_agent.bot.store import SqliteSessionStore
from osc_agent.runtime.completion import CompletionEvidenceStopHook
from osc_agent.runtime.hooks import StopHookPayload
from osc_agent.runtime.models import CapabilityScope, CompletionRequirements, ToolUseContext


class TrustedPublisher:
    """唯一持有远程写权限的交付组件；不执行仓库代码。"""

    def __init__(self, *, settings: BotSettings, store: BotStore, github: GitHubControlClient) -> None:
        self.settings = settings
        self.store = store
        self.github = github
        self.git = shutil.which("git") or ""
        if not self.git:
            raise ValueError("Git executable was not found")

    async def publish(self, job: BotJob, config: RepositoryBotConfig) -> BotJob:
        if job.status not in {"ready_to_publish", "publishing"}:
            raise ValueError("job is not ready for trusted publication")
        if not job.workspace_path or not job.approval_id:
            raise ValueError("publish job is missing workspace or approval")
        approval = self.store.get_approval(job.approval_id)
        plan = self.store.get_plan_artifact(job.plan_artifact_id or "")
        validate_implementation_approval(approval, plan)
        workspace = Path(job.workspace_path).resolve()
        draft = self.store.get_delivery_draft(job.job_id)
        if draft is None:
            raise ValueError("publish job has no DeliveryDraft")
        await self._verify_repository_metadata(job, workspace)
        branch_base, remote_sha = await self.github.repository_head(job.installation_id, job.repository_full_name)
        if remote_sha != job.base_sha:
            return self._mark_stale(job)
        head = (await self._git(["rev-parse", "HEAD"], workspace)).strip()
        if job.commit_sha is None and head == job.base_sha:
            await self._verify_completion(job, config, workspace)
            fingerprint = git_workspace_fingerprint(repo_root=workspace)
            if fingerprint != draft.snapshot_fingerprint:
                raise ValueError("workspace fingerprint differs from DeliveryDraft")
        elif job.commit_sha is not None:
            if head != job.commit_sha:
                raise ValueError("workspace HEAD differs from the persisted publish commit")
            await self._verify_persisted_commit(job, draft.commit_message, workspace)
        elif job.status == "publishing":
            # Recovery for a crash after `git commit` but before commit_sha was
            # persisted. The worker cannot write .git; accept only one clean,
            # publisher-shaped commit directly on the approved base.
            await self._verify_persisted_commit(job, draft.commit_message, workspace)
        else:
            raise ValueError("workspace HEAD differs from the approved base")
        snapshot = git_snapshot(repo_root=workspace, base_commit=job.base_sha)
        files = [str(item["path"]) for item in snapshot["files"]]
        if not files or len(files) > config.max_changed_files:
            raise ValueError("final changed file count violates repository policy")
        if len(str(snapshot["patch"]).encode()) > config.max_patch_bytes:
            raise ValueError("final patch exceeds repository policy")
        for value in files:
            path = PurePosixPath(value.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"publisher rejected invalid changed path: {value}")
            if any(_matches(path, pattern) for pattern in config.denied_paths):
                raise ValueError(f"publisher rejected protected path: {value}")
            candidate = workspace.joinpath(*path.parts)
            if candidate.exists() and (
                candidate.is_symlink() or not candidate.resolve().is_relative_to(workspace)
            ):
                raise ValueError(f"publisher rejected changed symlink or path escape: {value}")
        current = job
        branch = job.branch or f"osa/issue-{job.issue_number}-{job.job_id[:8]}"
        if current.status != "publishing":
            current = self.store.update_job(current.job_id, expected_version=current.version, status="publishing", branch=branch)
        if current.commit_sha is None:
            current_head = (await self._git(["rev-parse", "HEAD"], workspace)).strip()
            if current_head == current.base_sha:
                await self._git(["config", "user.name", self.settings.github_commit_name], workspace)
                await self._git(["config", "user.email", self.settings.github_commit_email], workspace)
                # The validated snapshot already covers every Git-visible change.
                # Stage that whole workspace so rename source paths are not lost by
                # a target-only pathspec, then compare the staged projection below.
                await self._git(["add", "--all"], workspace)
                staged = set(filter(None, (await self._git(["diff", "--cached", "--name-only", "-z"], workspace)).split("\0")))
                if staged != set(files):
                    raise ValueError("staged files differ from the final Git snapshot")
                staged_snapshot = git_snapshot(
                    repo_root=workspace,
                    base_commit=current.base_sha,
                )
                if (
                    staged_snapshot["patch"] != snapshot["patch"]
                    or {str(item["path"]) for item in staged_snapshot["files"]}
                    != set(files)
                ):
                    raise ValueError("workspace changed while Publisher was staging the final snapshot")
                await self._git(["commit", "-m", draft.commit_message], workspace)
                commit_sha = (await self._git(["rev-parse", "HEAD"], workspace)).strip()
            else:
                await self._verify_persisted_commit(current, draft.commit_message, workspace)
                commit_sha = current_head
            current = self.store.update_job(current.job_id, expected_version=current.version, branch=branch, commit_sha=commit_sha)
        await self._verify_persisted_commit(current, draft.commit_message, workspace)
        await self._verify_repository_metadata(current, workspace, allow_publisher_identity=True)
        latest_base, latest_sha = await self.github.repository_head(
            current.installation_id,
            current.repository_full_name,
        )
        if latest_sha != current.base_sha or latest_base != branch_base:
            return self._mark_stale(current)
        token = await self.github.installation_token(current.installation_id, contents="write")
        env = build_subprocess_environment()
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.https://github.com/.extraHeader",
                "GIT_CONFIG_VALUE_0": basic_git_auth_header(token),
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        await self._git(["push", "origin", f"HEAD:refs/heads/{branch}"], workspace, env=env)
        self.store.append_job_event(current.job_id, "BranchPushed", {"branch": branch, "commit_sha": current.commit_sha or ""})
        found = await self.github.find_pull_request(
            current.installation_id, current.repository_full_name, head=branch, base=branch_base
        )
        if found is None:
            marker = f"<!-- osa-job:{current.job_id}:delivery -->"
            number, url = await self.github.create_draft_pull_request(
                current.installation_id,
                current.repository_full_name,
                title=draft.title,
                body=f"{draft.body}\n\nRefs #{current.issue_number}\n\n{marker}",
                head=branch,
                base=branch_base,
            )
        else:
            number, url = found
        completed = self.store.transition_with_outbox(
            job_id=current.job_id,
            expected_version=current.version,
            status="completed",
            pull_request_number=number,
            pull_request_url=url,
            lease_owner=None,
            lease_until=None,
            event=OutboxEvent(
                event_id=f"comment-{current.job_id}",
                job_id=current.job_id,
                kind="issue_comment",
                idempotency_key=f"comment:{current.job_id}:completed",
                payload={"body": f"Draft PR created: {url}\n\n<!-- osa-job:{current.job_id}:completed -->"},
            ),
        )
        self.store.append_job_event(completed.job_id, "PullRequestCreated", {"number": number, "url": url})
        return completed

    def _mark_stale(self, job: BotJob) -> BotJob:
        return self.store.transition_with_outbox(
            job_id=job.job_id,
            expected_version=job.version,
            status="stale",
            event=OutboxEvent(
                event_id=f"stale-{job.job_id}-{job.version}",
                job_id=job.job_id,
                kind="issue_comment",
                idempotency_key=f"comment:{job.job_id}:stale",
                payload={
                    "body": "The default branch changed before publication. Run `/osa plan` again."
                    f"\n\n<!-- osa-job:{job.job_id}:stale -->"
                },
            ),
        )

    async def _verify_persisted_commit(
        self,
        job: BotJob,
        expected_message: str,
        workspace: Path,
    ) -> None:
        status = await self._git(["status", "--porcelain=v1", "--untracked-files=all"], workspace)
        if status.strip():
            raise ValueError("published workspace contains changes outside the persisted commit")
        parent = (await self._git(["rev-parse", "HEAD^"], workspace)).strip()
        if parent != job.base_sha:
            raise ValueError("persisted publish commit is not directly based on the approved SHA")
        count = (await self._git(["rev-list", "--count", f"{job.base_sha}..HEAD"], workspace)).strip()
        if count != "1":
            raise ValueError("persisted publication must contain exactly one commit")
        message = (await self._git(["log", "-1", "--format=%s"], workspace)).strip()
        if message != expected_message:
            raise ValueError("persisted publish commit message differs from DeliveryDraft")

    async def _verify_repository_metadata(
        self,
        job: BotJob,
        workspace: Path,
        *,
        allow_publisher_identity: bool = False,
    ) -> None:
        git_dir = (await self._git(["rev-parse", "--git-dir"], workspace)).strip()
        resolved_git_dir = (
            Path(git_dir).resolve()
            if Path(git_dir).is_absolute()
            else (workspace / git_dir).resolve()
        )
        if resolved_git_dir != (workspace / ".git").resolve():
            raise ValueError("workspace Git metadata is not the expected fresh clone")
        origin = (await self._git(["remote", "get-url", "origin"], workspace)).strip()
        if origin != f"https://github.com/{job.repository_full_name}.git":
            raise ValueError("workspace origin differs from the authenticated repository")
        names = set(
            filter(
                None,
                (await self._git(["config", "--local", "--name-only", "-z", "--list"], workspace)).split("\0"),
            )
        )
        denied_prefixes = ("credential.", "http.", "url.", "include.", "includeif.")
        if any(name.casefold().startswith(denied_prefixes) for name in names):
            raise ValueError("workspace contains forbidden local Git authentication configuration")
        if "core.hookspath" in {name.casefold() for name in names}:
            raise ValueError("workspace contains a custom Git hooks path")
        identity = {name.casefold() for name in names if name.casefold().startswith("user.")}
        allowed_identity = {"user.name", "user.email"} if allow_publisher_identity else set()
        if not identity.issubset(allowed_identity):
            raise ValueError("workspace contains an unexpected local Git identity")
        hooks = workspace / ".git" / "hooks"
        if hooks.exists() and any(
            entry.is_file() and not entry.name.endswith(".sample") for entry in hooks.iterdir()
        ):
            raise ValueError("workspace contains an active Git hook")

    async def _verify_completion(self, job: BotJob, config: RepositoryBotConfig, workspace: Path) -> None:
        if not job.implementation_session_id:
            raise ValueError("publish job has no implementation Session")
        snapshot = SqliteSessionStore(self.store).load(job.implementation_session_id)
        if snapshot is None or snapshot.runtime_state.last_status != "completed":
            raise ValueError("implementation Session is not completed")
        requirements = CompletionRequirements(
            required_evidence=frozenset(
                {"successful_test", "independent_verification", "git_change_snapshot", "delivery_draft"}
            )
        )
        context = ToolUseContext(
            session_id=job.implementation_session_id,
            working_directory=str(workspace),
            repository_root=str(workspace),
            state_directory=str(workspace),
            capabilities=snapshot.metadata.capabilities,
            completion_requirements=requirements,
        )
        payload = StopHookPayload(messages=snapshot.messages)
        evidence = await CompletionEvidenceStopHook()(payload, context)
        configured = await ConfiguredValidationStopHook(config.validation_commands)(payload, context)
        reasons = evidence.blocking_reasons + configured.blocking_reasons
        if reasons:
            raise ValueError("publisher completion validation failed: " + "; ".join(reasons))

    async def _git(self, arguments: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            self.git,
            *arguments,
            cwd=cwd,
            env=env or build_subprocess_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ValueError("trusted publish Git operation failed: " + stderr.decode(errors="replace")[:1_000])
        return stdout.decode("utf-8", errors="replace")
