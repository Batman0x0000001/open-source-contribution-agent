"""解析 GitHub issue_comment Payload 与 `/osc-agent` 命令。"""

from __future__ import annotations

import re

from osc_agent.bot.control.models import IssueCommentContext, WebhookCommand


_JOB_ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_COMMAND_PREFIX = "/osc-agent"
_IMPLEMENT = re.compile(rf"^{re.escape(_COMMAND_PREFIX)} implement ({_JOB_ID})$")
_CANCEL = re.compile(rf"^{re.escape(_COMMAND_PREFIX)} cancel ({_JOB_ID})$")


def parse_webhook_command(body: str) -> WebhookCommand | None:
    command = body.strip()
    if command == f"{_COMMAND_PREFIX} plan":
        return WebhookCommand(action="plan")
    if command == f"{_COMMAND_PREFIX} implement":
        return WebhookCommand(action="implement")
    if match := _IMPLEMENT.fullmatch(command):
        return WebhookCommand(action="implement", job_id=match.group(1))
    if command == f"{_COMMAND_PREFIX} cancel":
        return WebhookCommand(action="cancel")
    if match := _CANCEL.fullmatch(command):
        return WebhookCommand(action="cancel", job_id=match.group(1))
    if command == f"{_COMMAND_PREFIX} status":
        return WebhookCommand(action="status")
    if command == f"{_COMMAND_PREFIX} retry":
        return WebhookCommand(action="retry")
    reply_prefix = f"{_COMMAND_PREFIX} reply "
    if command.startswith(reply_prefix) and command[len(reply_prefix) :].strip():
        return WebhookCommand(
            action="reply", message=command[len(reply_prefix) :].strip()
        )
    return None


def parse_issue_comment(payload: dict[str, object]) -> IssueCommentContext:
    if payload.get("action") != "created":
        raise ValueError("only newly created issue comments are supported")
    installation = payload.get("installation")
    repository = payload.get("repository")
    issue = payload.get("issue")
    comment = payload.get("comment")
    sender = payload.get("sender")
    if not all(isinstance(item, dict) for item in (installation, repository, issue, comment, sender)):
        raise ValueError("GitHub issue_comment payload is incomplete")
    assert isinstance(installation, dict) and isinstance(repository, dict)
    assert isinstance(issue, dict) and isinstance(comment, dict) and isinstance(sender, dict)
    if issue.get("pull_request") is not None:
        raise ValueError("pull request comments are not supported")
    login = sender.get("login")
    if sender.get("type") == "Bot" or (isinstance(login, str) and login.endswith("[bot]")):
        raise ValueError("bot comments are ignored")
    values = {
        "installation_id": installation.get("id"),
        "repository_id": repository.get("id"),
        "repository": repository.get("full_name"),
        "issue_number": issue.get("number"),
        "issue_url": issue.get("html_url"),
        "body": comment.get("body"),
        "actor_id": sender.get("id"),
        "actor_login": login,
        "comment_id": comment.get("id"),
    }
    if not isinstance(values["installation_id"], int) or not isinstance(values["repository_id"], int):
        raise ValueError("GitHub installation or repository id is invalid")
    if not isinstance(values["issue_number"], int) or not isinstance(values["actor_id"], int):
        raise ValueError("GitHub issue or actor id is invalid")
    if not isinstance(values["comment_id"], int):
        raise ValueError("GitHub comment id is invalid")
    if any(
        not isinstance(values[key], str) or not values[key]
        for key in ("repository", "issue_url", "body", "actor_login")
    ):
        raise ValueError("GitHub issue_comment text fields are invalid")
    return IssueCommentContext.model_validate(values)
