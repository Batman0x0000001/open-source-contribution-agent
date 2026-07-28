"""实现 GitHub App 身份认证、安装令牌和 API 请求。"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Protocol

from osc_agent.bot.config import BotSettings


class GitHubApiError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.retryable = status_code in {408, 409, 429} or status_code >= 500
        super().__init__(f"GitHub API request failed with HTTP {status_code}")


class GitHubControlClient(Protocol):
    async def collaborator_permission(self, installation_id: int, repository: str, login: str) -> str: ...

    async def repository_head(self, installation_id: int, repository: str) -> tuple[str, str]: ...

    async def issue(self, installation_id: int, repository: str, number: int) -> dict[str, object]: ...

    async def create_issue_comment(self, installation_id: int, repository: str, number: int, body: str) -> str: ...

    async def find_issue_comment(
        self,
        installation_id: int,
        repository: str,
        number: int,
        marker: str,
    ) -> str | None: ...

    async def create_draft_pull_request(
        self,
        installation_id: int,
        repository: str,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = True,
    ) -> tuple[int, str]: ...

    async def find_pull_request(
        self, installation_id: int, repository: str, *, head: str, base: str
    ) -> tuple[int, str] | None: ...

    async def installation_token(
        self, installation_id: int, *, contents: str = "read"
    ) -> str: ...


class GitHubAppClient:
    """最小 GitHub App REST 客户端；所有响应在边界处校验。"""

    def __init__(self, settings: BotSettings) -> None:
        self.settings = settings
        self._token_cache: dict[tuple[int, str], tuple[str, datetime]] = {}

    async def collaborator_permission(self, installation_id: int, repository: str, login: str) -> str:
        data = await self._request(
            installation_id,
            "GET",
            f"/repos/{repository}/collaborators/{login}/permission",
            contents="read",
        )
        permission = data.get("permission")
        if not isinstance(permission, str):
            raise ValueError("GitHub permission response is invalid")
        return permission

    async def repository_head(self, installation_id: int, repository: str) -> tuple[str, str]:
        repo = await self._request(installation_id, "GET", f"/repos/{repository}")
        branch = repo.get("default_branch")
        if not isinstance(branch, str) or not branch:
            raise ValueError("GitHub repository has no valid default branch")
        ref = await self._request(
            installation_id, "GET", f"/repos/{repository}/git/ref/heads/{branch}"
        )
        obj = ref.get("object")
        sha = obj.get("sha") if isinstance(obj, dict) else None
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ValueError("GitHub default branch head is invalid")
        return branch, sha

    async def issue(self, installation_id: int, repository: str, number: int) -> dict[str, object]:
        issue = await self._request(
            installation_id, "GET", f"/repos/{repository}/issues/{number}"
        )
        comments = await self._request(
            installation_id,
            "GET",
            f"/repos/{repository}/issues/{number}/comments?per_page=30",
        )
        return {
            "content_source": "github",
            "trust": "untrusted_external",
            "issue": issue,
            "comments": comments if isinstance(comments, list) else [],
        }

    async def create_issue_comment(self, installation_id: int, repository: str, number: int, body: str) -> str:
        data = await self._request(
            installation_id,
            "POST",
            f"/repos/{repository}/issues/{number}/comments",
            json_body={"body": sanitize_github_markdown(body)},
            contents="write",
        )
        url = data.get("html_url")
        if not isinstance(url, str):
            raise ValueError("GitHub comment response is invalid")
        return url

    async def find_issue_comment(
        self,
        installation_id: int,
        repository: str,
        number: int,
        marker: str,
    ) -> str | None:
        data = await self._request(
            installation_id,
            "GET",
            f"/repos/{repository}/issues/{number}/comments?per_page=100",
        )
        if not isinstance(data, list):
            raise ValueError("GitHub issue comment lookup response is invalid")
        for item in data:
            if not isinstance(item, dict):
                continue
            body, url = item.get("body"), item.get("html_url")
            if isinstance(body, str) and marker in body and isinstance(url, str):
                return url
        return None

    async def create_draft_pull_request(
        self,
        installation_id: int,
        repository: str,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
        draft: bool = True,
    ) -> tuple[int, str]:
        data = await self._request(
            installation_id,
            "POST",
            f"/repos/{repository}/pulls",
            json_body={
                "title": sanitize_github_markdown(title, limit=72),
                "body": sanitize_github_markdown(body),
                "head": head,
                "base": base,
                "draft": draft,
            },
            contents="write",
        )
        number, url = data.get("number"), data.get("html_url")
        if not isinstance(number, int) or not isinstance(url, str):
            raise ValueError("GitHub pull request response is invalid")
        return number, url

    async def find_pull_request(
        self, installation_id: int, repository: str, *, head: str, base: str
    ) -> tuple[int, str] | None:
        owner = repository.split("/", 1)[0]
        data = await self._request(
            installation_id,
            "GET",
            f"/repos/{repository}/pulls?state=open&head={owner}:{head}&base={base}&per_page=10",
            contents="read",
        )
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        if not isinstance(first, dict) or not isinstance(first.get("number"), int) or not isinstance(first.get("html_url"), str):
            raise ValueError("GitHub pull request lookup response is invalid")
        return int(first["number"]), str(first["html_url"])

    async def installation_token(self, installation_id: int, *, contents: str = "read") -> str:
        key = (installation_id, contents)
        cached = self._token_cache.get(key)
        now = datetime.now(timezone.utc)
        if cached is not None and cached[1] > now + timedelta(minutes=2):
            return cached[0]
        payload = await self._app_request(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            json_body={
                "permissions": {
                    "contents": contents,
                    "issues": "write" if contents == "write" else "read",
                    "pull_requests": "write" if contents == "write" else "read",
                }
            },
        )
        token, expires = payload.get("token"), payload.get("expires_at")
        if not isinstance(token, str) or not isinstance(expires, str):
            raise ValueError("GitHub installation token response is invalid")
        expiry = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        self._token_cache[key] = (token, expiry)
        return token

    async def _request(
        self,
        installation_id: int,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        contents: str = "read",
    ) -> Any:
        token = await self.installation_token(installation_id, contents=contents)
        return await _http_request(method, path, token, json_body)

    async def _app_request(
        self, method: str, path: str, *, json_body: dict[str, object] | None = None
    ) -> Any:
        try:
            import jwt
        except ImportError as exc:
            raise ValueError("GitHub bot dependencies are missing; install .[bot]") from exc
        private_key = self.settings.github_app_private_key_path.read_text(encoding="utf-8")
        now = datetime.now(timezone.utc)
        token = jwt.encode(
            {
                "iat": int((now - timedelta(seconds=30)).timestamp()),
                "exp": int((now + timedelta(minutes=9)).timestamp()),
                "iss": str(self.settings.github_app_id),
            },
            private_key,
            algorithm="RS256",
        )
        return await _http_request(method, path, token, json_body)


async def _http_request(
    method: str, path: str, token: str, json_body: dict[str, object] | None
) -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ValueError("GitHub bot dependencies are missing; install .[bot]") from exc
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "osc-agent-bot",
    }
    async with httpx.AsyncClient(base_url="https://api.github.com", timeout=20) as client:
        response = await client.request(method, path, headers=headers, json=json_body)
    if response.status_code >= 400:
        raise GitHubApiError(response.status_code)
    return response.json()


def sanitize_github_markdown(value: str, *, limit: int = 60_000) -> str:
    safe = re.sub(r"(?<![\w`])@(?=[A-Za-z0-9])", "@\u200b", value)
    return safe[:limit]


def basic_git_auth_header(token: str) -> str:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"Authorization: Basic {encoded}"
