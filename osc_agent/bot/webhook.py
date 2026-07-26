from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from osc_agent.bot.control import BotControlService
from osc_agent.bot.github_app import GitHubApiError
from osc_agent.bot.store import BotStore


MAX_WEBHOOK_BYTES = 1_000_000


def verify_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    if not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def create_webhook_app(
    *,
    secret: str,
    store: BotStore,
    control: BotControlService,
    outbox_processor: Any | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
    except ImportError as exc:
        raise ValueError("GitHub bot dependencies are missing; install .[bot]") from exc

    globals()["Request"] = Request
    app = FastAPI(title="OSA Agent GitHub App", docs_url=None, redoc_url=None)
    task: Any = None

    if outbox_processor is not None:
        import asyncio

        async def poll_outbox() -> None:
            while True:
                worked = await outbox_processor.run_once()
                if not worked:
                    await asyncio.sleep(1)

        @app.on_event("startup")
        async def start_outbox() -> None:
            nonlocal task
            task = asyncio.create_task(poll_outbox())

        @app.on_event("shutdown")
        async def stop_outbox() -> None:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        store.initialize()
        return {"status": "ready"}

    @app.post("/webhooks/github")
    async def webhook(
        request: Request,
        x_hub_signature_256: str | None = Header(default=None),
        x_github_event: str | None = Header(default=None),
        x_github_delivery: str | None = Header(default=None),
    ) -> dict[str, object]:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid content length") from exc
            if declared_length > MAX_WEBHOOK_BYTES:
                raise HTTPException(status_code=413, detail="webhook payload is too large")
        body = await request.body()
        if len(body) > MAX_WEBHOOK_BYTES:
            raise HTTPException(status_code=413, detail="webhook payload is too large")
        if not x_hub_signature_256 or not verify_webhook_signature(body, x_hub_signature_256, secret):
            raise HTTPException(status_code=401, detail="invalid webhook signature")
        if not x_github_event or not x_github_delivery:
            raise HTTPException(status_code=400, detail="missing GitHub webhook headers")
        if not store.register_delivery(x_github_delivery, x_github_event):
            return {"accepted": True, "duplicate": True}
        if x_github_event != "issue_comment":
            return {"accepted": True, "ignored": True}
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            result = await control.handle_issue_comment(payload)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except GitHubApiError as exc:
            if exc.retryable:
                store.release_delivery(x_github_delivery)
                raise HTTPException(
                    status_code=503,
                    detail="temporary GitHub API failure",
                ) from None
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            store.release_delivery(x_github_delivery)
            raise HTTPException(status_code=503, detail="temporary control service failure") from None
        return {"accepted": True, "result": result}

    return app
