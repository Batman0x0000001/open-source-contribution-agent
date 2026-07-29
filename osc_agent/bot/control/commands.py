"""解析 GitHub issue_comment Payload 与 `/osa` 命令。"""

from __future__ import annotations

import re

from osc_agent.bot.control.models import IssueCommentContext, WebhookCommand


_JOB_ID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_IMPLEMENT = re.compile(rf"^/osa implement ({_JOB_ID})$")
_CANCEL = re.compile(rf"^/osa cancel ({_JOB_ID})$")


def parse_webhook_command(body: str) -> WebhookCommand | None:
    command = body.strip()
    if command == "/osa plan":
        return WebhookCommand(action="plan")
    if command == "/osa implement":
        return WebhookCommand(action="implement")
    if match := _IMPLEMENT.fullmatch(command):
        return WebhookCommand(action="implement", job_id=match.group(1))
    if command == "/osa cancel":
        return WebhookCommand(action="cancel")
    if match := _CANCEL.fullmatch(command):
        return WebhookCommand(action="cancel", job_id=match.group(1))
    if command == "/osa status":
        return WebhookCommand(action="status")
    if command == "/osa retry":
        return WebhookCommand(action="retry")
    if command.startswith("/osa reply ") and command[11:].strip():
        return WebhookCommand(action="reply", message=command[11:].strip())
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
