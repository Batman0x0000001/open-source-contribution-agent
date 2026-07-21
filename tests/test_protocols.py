from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

from osc_agent.agent_loop import TOOLS, build_tool_handlers
from osc_agent.config import Settings
from osc_agent.harness.protocols import (
    consume_inbox,
    load_protocol_state,
    match_response,
    request_plan_review,
    request_shutdown,
    request_write_approval,
    review_plan,
)
from osc_agent.harness.teams import MessageBus, collect_team_notifications, spawn_teammate


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def _request_id(output: str) -> str:
    return json.loads(output)["request_id"]


def test_request_plan_review_creates_pending_state_and_lead_message(tmp_path):
    request_id = _request_id(request_plan_review(repo_root=tmp_path, sender="alice", plan="Refactor auth."))

    state = load_protocol_state(tmp_path, request_id)
    inbox = consume_inbox(tmp_path, "lead")

    assert state is not None
    assert state.type == "plan_approval"
    assert state.status == "pending"
    assert inbox[0]["type"] == "plan_approval_request"
    assert inbox[0]["metadata"]["request_id"] == request_id


def test_review_plan_updates_state_and_sends_response(tmp_path):
    request_id = _request_id(request_plan_review(repo_root=tmp_path, sender="alice", plan="Update docs."))
    consume_inbox(tmp_path, "lead")

    result = review_plan(repo_root=tmp_path, request_id=request_id, approve=True, feedback="Looks good.")
    state = load_protocol_state(tmp_path, request_id)
    teammate_inbox = consume_inbox(tmp_path, "alice")

    assert result == f"Request {request_id} approved"
    assert state is not None and state.status == "approved"
    assert teammate_inbox[0]["type"] == "plan_approval_response"
    assert teammate_inbox[0]["content"] == "Looks good."


def test_match_response_rejects_wrong_response_type(tmp_path):
    request_id = _request_id(request_plan_review(repo_root=tmp_path, sender="alice", plan="Risky change."))

    result = match_response(
        repo_root=tmp_path,
        response_type="shutdown_response",
        request_id=request_id,
        approve=True,
    )

    assert "does not match plan_approval" in result
    assert load_protocol_state(tmp_path, request_id).status == "pending"  # type: ignore[union-attr]


def test_protocol_rejects_response_from_wrong_agent(tmp_path):
    request_id = _request_id(request_plan_review(repo_root=tmp_path, sender="alice", plan="Safe plan."))
    MessageBus(tmp_path).send(
        "mallory",
        "alice",
        "approved",
        "plan_approval_response",
        {"request_id": request_id, "approve": True, "protocol_type": "plan_approval"},
    )

    consume_inbox(tmp_path, "alice")

    assert load_protocol_state(tmp_path, request_id).status == "pending"  # type: ignore[union-attr]


def test_concurrent_protocol_creation_does_not_lose_requests(tmp_path):
    barrier = threading.Barrier(3)
    request_ids: list[str] = []

    def create(sender: str) -> None:
        barrier.wait()
        request_ids.append(_request_id(request_plan_review(repo_root=tmp_path, sender=sender, plan="Plan.")))

    threads = [threading.Thread(target=create, args=(sender,)) for sender in ("alice", "bob")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(request_ids) == 2
    assert all(load_protocol_state(tmp_path, request_id) is not None for request_id in request_ids)


def test_shutdown_request_round_trip_updates_state(tmp_path):
    request_id = _request_id(request_shutdown(repo_root=tmp_path, target="alice", reason="done"))

    teammate_messages = consume_inbox(tmp_path, "alice")
    lead_messages = consume_inbox(tmp_path, "lead")
    state = load_protocol_state(tmp_path, request_id)

    assert teammate_messages[0]["type"] == "shutdown_request"
    assert lead_messages[0]["type"] == "shutdown_response"
    assert state is not None and state.status == "approved"


def test_request_write_approval_uses_matching_response_type(tmp_path):
    request_id = _request_id(
        request_write_approval(repo_root=tmp_path, sender="tester", path="README.md", reason="fix typo")
    )
    consume_inbox(tmp_path, "lead")
    MessageBus(tmp_path).send(
        "lead",
        "tester",
        "approved",
        "write_approval_response",
        {"request_id": request_id, "approve": True, "protocol_type": "write_approval"},
    )

    consume_inbox(tmp_path, "tester")

    assert load_protocol_state(tmp_path, request_id).status == "approved"  # type: ignore[union-attr]


class PlanThenContinueMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="request_plan_review",
                        id="toolu_plan",
                        input={"sender": "alice", "plan": "Rewrite auth module."},
                    )
                ],
            )
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="continued")])


class FakeClient:
    def __init__(self, messages) -> None:
        self.messages = messages


def test_teammate_resumes_after_plan_approval(tmp_path):
    fake_messages = PlanThenContinueMessages()

    spawn_teammate(
        name="alice",
        role="reviewer",
        prompt="Submit a plan first.",
        repo_root=tmp_path,
        client=FakeClient(fake_messages),
        settings=_settings(),
        autonomous=False,
    )

    request = None
    for _ in range(50):
        inbox = MessageBus(tmp_path).read_inbox("lead")
        request = next((message for message in inbox if message["type"] == "plan_approval_request"), None)
        if request is not None:
            break
        time.sleep(0.02)

    assert request is not None
    request_id = request["metadata"]["request_id"]
    review_plan(repo_root=tmp_path, request_id=request_id, approve=True, feedback="Proceed.")

    notifications: list[str] = []
    for _ in range(100):
        notifications.extend(collect_team_notifications(tmp_path))
        if any("continued" in notification for notification in notifications):
            break
        time.sleep(0.02)

    assert len(fake_messages.calls) == 2
    assert any("continued" in notification for notification in notifications)


def test_agent_loop_registers_protocol_tools(tmp_path):
    tool_names = {tool["name"] for tool in TOOLS}
    handlers = build_tool_handlers(tmp_path, client=FakeClient(PlanThenContinueMessages()), settings=_settings())

    assert {"request_shutdown", "request_plan_review", "review_plan", "request_write_approval"} <= tool_names
    output = handlers["request_shutdown"](target="alice", reason="complete")
    assert json.loads(output)["type"] == "shutdown"
