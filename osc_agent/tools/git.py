"""提供 Git 状态、差异、历史和工作区指纹工具。"""

from __future__ import annotations

import asyncio
import difflib
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from pydantic import Field

from osc_agent.runtime.models import ContractModel, ToolError, ToolResult, ToolUseContext
from osc_agent.runtime.session_store import ToolResultStore
from osc_agent.runtime.tool import BaseTool
from osc_agent.tools.process_runner import build_subprocess_environment


MAX_GIT_OUTPUT_CHARS = 50_000


class GitStatusInput(ContractModel):
    pass


class GitStatusOutput(ContractModel):
    status: str


class GitStatusTool(BaseTool[GitStatusInput, GitStatusOutput]):
    name = "git_status"
    description = "Show the concise Git working-tree status for the current repository."
    input_model = GitStatusInput
    output_model = GitStatusOutput

    def is_read_only(self, input: GitStatusInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitStatusInput) -> bool:
        return True

    async def call(self, input: GitStatusInput, context: ToolUseContext) -> ToolResult:
        return _git_tool_result(
            "status",
            await asyncio.to_thread(git_status, repo_root=Path(context.working_directory)),
        )


class GitDiffInput(ContractModel):
    pass


class GitChangedFile(ContractModel):
    path: str
    status: str
    staged: bool
    unstaged: bool
    untracked: bool


class GitDiffOutput(ContractModel):
    base_commit: str
    files: list[GitChangedFile]
    commits_ahead: int = Field(ge=0)
    stat: str
    patch: str
    truncated: bool
    full_result_id: str | None = None
    has_changes: bool
    workspace_fingerprint: str


class GitDiffTool(BaseTool[GitDiffInput, GitDiffOutput]):
    name = "git_diff"
    description = "Show tracked Git changes in the current repository."
    input_model = GitDiffInput
    output_model = GitDiffOutput

    def __init__(self, tool_result_store: ToolResultStore) -> None:
        self.tool_result_store = tool_result_store

    def is_read_only(self, input: GitDiffInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitDiffInput) -> bool:
        return True

    async def call(self, input: GitDiffInput, context: ToolUseContext) -> ToolResult:
        try:
            before = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.working_directory),
            )
            snapshot = await asyncio.to_thread(
                git_snapshot,
                repo_root=Path(context.working_directory),
                base_commit=context.worktree.base_commit if context.worktree else None,
            )
            after = await asyncio.to_thread(
                git_workspace_fingerprint,
                repo_root=Path(context.working_directory),
            )
            if before != after:
                raise ValueError("Git workspace changed while the snapshot was being generated")
        except ValueError as exc:
            return ToolResult(error=ToolError(code="GIT_COMMAND_FAILED", message=str(exc)))
        patch = snapshot["patch"]
        assert isinstance(patch, str)
        truncated = len(patch) > MAX_GIT_OUTPUT_CHARS
        result_id = None
        if truncated:
            result_id = self.tool_result_store.persist(
                session_id=context.session_id,
                tool_use_id=f"git-diff-{sha256(patch.encode('utf-8')).hexdigest()[:16]}",
                content=patch,
            )
        snapshot["patch"] = patch[:MAX_GIT_OUTPUT_CHARS]
        snapshot["truncated"] = truncated
        snapshot["full_result_id"] = result_id
        snapshot["workspace_fingerprint"] = after
        return ToolResult(data=snapshot)


class GitLogInput(ContractModel):
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of recent commits to return.")


class GitLogOutput(ContractModel):
    log: str


class GitLogTool(BaseTool[GitLogInput, GitLogOutput]):
    name = "git_log"
    description = "Show a bounded list of recent Git commits."
    input_model = GitLogInput
    output_model = GitLogOutput

    def is_read_only(self, input: GitLogInput) -> bool:
        return True

    def is_concurrency_safe(self, input: GitLogInput) -> bool:
        return True

    async def call(self, input: GitLogInput, context: ToolUseContext) -> ToolResult:
        return _git_tool_result(
            "log",
            await asyncio.to_thread(
                git_log,
                repo_root=Path(context.working_directory),
                limit=input.limit,
            ),
        )


def _git_tool_result(field: str, output: str) -> ToolResult:
    if output.startswith("Error: "):
        return ToolResult(
            error=ToolError(code="GIT_COMMAND_FAILED", message=output.removeprefix("Error: "))
        )
    return ToolResult(data={field: output})


def _run_git(
    repo_root: Path,
    arguments: list[str],
    *,
    max_chars: int | None = MAX_GIT_OUTPUT_CHARS,
) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"safe.directory={repo_root.resolve()}",
                *arguments,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={
                **build_subprocess_environment(),
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Error: {exc}"
    stdout = (completed.stdout or "").strip("\r\n")
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (stdout, (completed.stderr or "").strip("\r\n")) if part
        )
        return f"Error: git command failed with exit code {completed.returncode}: {detail}"[
            :MAX_GIT_OUTPUT_CHARS
        ]
    output = stdout or "(no output)"
    return output if max_chars is None else output[:max_chars]


def git_status(*, repo_root: Path) -> str:
    return _run_git(repo_root, ["status", "--short"])


def git_diff(*, repo_root: Path) -> str:
    return _run_git(repo_root, ["diff", "--no-ext-diff", "--no-textconv", "--"])


def git_log(*, repo_root: Path, limit: int = 5) -> str:
    return _run_git(repo_root, ["log", f"-{min(max(limit, 1), 50)}", "--oneline"])


def git_workspace_fingerprint(*, repo_root: Path) -> str:
    """计算所有 Git 可见状态的稳定指纹，供只读执行边界做前后校验。"""

    root = repo_root.resolve()
    top_level = _require_git(root, ["rev-parse", "--show-toplevel"])
    if Path(top_level).resolve() != root:
        raise ValueError("agent working directory must be the Git worktree root")

    digest = sha256()
    for label, arguments in (
        ("head", ["rev-parse", "HEAD"]),
        (
            "status",
            [
                "-c",
                "diff.ignoreSubmodules=none",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
        ),
        ("index", ["diff", "--binary", "--cached", "--no-ext-diff", "--no-textconv", "--"]),
        ("worktree", ["diff", "--binary", "--no-ext-diff", "--no-textconv", "--"]),
        ("submodules", ["submodule", "status", "--recursive"]),
    ):
        _update_fingerprint(digest, label, _require_git_bytes(root, arguments))

    untracked = _require_git_bytes(
        root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    paths = sorted(path for path in untracked.split(b"\0") if path)
    for encoded_path in paths:
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts:
            raise ValueError(f"invalid untracked Git path: {relative}")
        path = root / relative
        stat = path.lstat()
        metadata = json.dumps(
            {
                "path": relative.replace("\\", "/"),
                "mode": stat.st_mode,
                "size": stat.st_size,
                "symlink": path.is_symlink(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        _update_fingerprint(digest, "untracked-metadata", metadata)
        if path.is_symlink():
            _update_fingerprint(
                digest,
                "untracked-content",
                os.readlink(path).encode("utf-8", errors="surrogateescape"),
            )
            continue
        content_hash = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content_hash.update(chunk)
        _update_fingerprint(digest, "untracked-content", content_hash.digest())
    return digest.hexdigest()


def git_snapshot(*, repo_root: Path, base_commit: str | None = None) -> dict[str, object]:
    base = base_commit or _require_git(repo_root, ["rev-parse", "HEAD"])
    status_text = _require_git(
        repo_root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    files = _merge_changed_files(
        _parse_base_changes(
            _require_git(
                repo_root,
                ["diff", "--name-status", "-z", base, "--"],
                empty_ok=True,
                max_chars=None,
            )
        ),
        _parse_status(status_text),
    )
    patch = _require_git(
        repo_root,
        ["diff", "--binary", "--no-ext-diff", "--no-textconv", base, "--"],
        empty_ok=True,
        max_chars=None,
    )
    stat_text = _require_git(
        repo_root,
        ["diff", "--stat", "--no-ext-diff", base, "--"],
        empty_ok=True,
    )
    for item in files:
        if not item["untracked"]:
            continue
        path = repo_root / str(item["path"])
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            patch += f"\nBinary or unreadable untracked file: {item['path']}\n"
            continue
        if len(raw) > 500_000:
            patch += f"\nOversized untracked file omitted: {item['path']} ({len(raw)} bytes)\n"
            continue
        patch += "\n" + "".join(
            difflib.unified_diff(
                [],
                text.splitlines(keepends=True),
                fromfile="/dev/null",
                tofile=f"b/{item['path']}",
            )
        )
    untracked_paths = [str(item["path"]) for item in files if item["untracked"]]
    if untracked_paths:
        stat_text = (
            (stat_text + "\n" if stat_text else "")
            + f"{len(untracked_paths)} untracked file(s): "
            + ", ".join(untracked_paths)
        )
    commits = int(
        _require_git(repo_root, ["rev-list", "--count", f"{base}..HEAD"])
    )
    return {
        "base_commit": base,
        "files": files,
        "commits_ahead": commits,
        "stat": stat_text or "(no tracked changes)",
        "patch": patch or "(no changes)",
        "has_changes": bool(files or commits),
    }


def _parse_base_changes(output: str) -> list[dict[str, object]]:
    records = output.split("\0") if output else []
    files: list[dict[str, object]] = []
    index = 0
    while index < len(records):
        status = records[index]
        index += 1
        if not status or index >= len(records):
            continue
        if status.startswith(("R", "C")):
            if index + 1 >= len(records):
                break
            _old_path = records[index]
            path = records[index + 1]
            index += 2
        else:
            path = records[index]
            index += 1
        files.append(
            {
                "path": path,
                "status": status,
                "staged": False,
                "unstaged": False,
                "untracked": False,
            }
        )
    return files


def _merge_changed_files(
    base_files: list[dict[str, object]],
    working_files: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged = {str(item["path"]): item for item in base_files}
    for item in working_files:
        path = str(item["path"])
        previous = merged.get(path)
        merged[path] = (
            {
                **previous,
                **item,
                "status": item["status"] if previous is None else f"{previous['status']}|{item['status']}",
            }
            if previous is not None
            else item
        )
    return [merged[path] for path in sorted(merged)]


def _parse_status(output: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    records = output.split("\0") if "\0" in output else output.splitlines()
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 3:
            continue
        code = record[:2]
        path = record[3:]
        if "\0" not in output and " -> " in path:
            path = path.split(" -> ", 1)[1]
        elif "\0" in output and any(marker in code for marker in ("R", "C")):
            # porcelain -z 在 rename/copy 记录后附带原路径；当前快照只报告目标路径。
            index += 1
        untracked = code == "??"
        files.append(
            {
                "path": path.strip('"'),
                "status": code,
                "staged": not untracked and code[0] not in {" ", "?"},
                "unstaged": not untracked and code[1] not in {" ", "?"},
                "untracked": untracked,
            }
        )
    return files


def _require_git(
    repo_root: Path,
    arguments: list[str],
    *,
    empty_ok: bool = False,
    max_chars: int | None = MAX_GIT_OUTPUT_CHARS,
) -> str:
    output = _run_git(repo_root, arguments, max_chars=max_chars)
    if output.startswith("Error: "):
        raise ValueError(output.removeprefix("Error: "))
    if output == "(no output)":
        return "" if empty_ok else output
    return output


def _require_git_bytes(repo_root: Path, arguments: list[str]) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"safe.directory={repo_root.resolve()}",
                *arguments,
            ],
            cwd=repo_root,
            capture_output=True,
            timeout=30,
            env={
                **build_subprocess_environment(),
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(str(exc)) from exc
    if completed.returncode != 0:
        detail = b"\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        ).decode("utf-8", errors="replace")
        raise ValueError(
            f"git command failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stdout


def _update_fingerprint(digest, label: str, value: bytes) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(4, "big"))
    digest.update(encoded_label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
