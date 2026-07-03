from __future__ import annotations

from types import SimpleNamespace

from osc_agent.agent_loop import TOOLS, agent_loop, build_tool_handlers
from osc_agent.config import Settings
from osc_agent.harness.compact import (
    apply_compaction,
    estimate_size,
    micro_compact,
    reactive_compact,
    snip_compact,
    tool_result_budget,
)


def _settings() -> Settings:
    return Settings(
        anthropic_api_key=None,
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def _tool_pair(index: int, content: str = "ok") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "bash", "id": f"toolu_{index}", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"toolu_{index}", "content": content}],
        },
    ]


def test_estimate_size_counts_message_content():
    small = estimate_size([{"role": "user", "content": "hello"}])
    larger = estimate_size([{"role": "user", "content": "hello" * 100}])

    assert larger > small


def test_snip_compact_keeps_tool_use_result_pairs():
    messages = [{"role": "user", "content": "start"}]
    for index in range(12):
        messages.extend(_tool_pair(index))

    compacted = snip_compact(messages, max_messages=8)

    assert len(compacted) < len(messages)
    for position, message in enumerate(compacted):
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    previous = compacted[position - 1]
                    assert any(
                        item.get("type") == "tool_use" and item.get("id") == block["tool_use_id"]
                        for item in previous["content"]
                    )


def test_micro_compact_replaces_old_tool_results_only():
    messages = []
    for index in range(5):
        messages.extend(_tool_pair(index, content=f"large-{index}-" + ("x" * 200)))

    compacted = micro_compact(messages, keep_recent=2, min_chars=20)
    result_contents = [
        block["content"]
        for message in compacted
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]

    assert result_contents[0].startswith("[Earlier tool result compacted.")
    assert result_contents[-1].startswith("large-4-")


def test_tool_result_budget_persists_large_output(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "big", "content": "A" * 300},
                {"type": "tool_result", "tool_use_id": "small", "content": "ok"},
            ],
        }
    ]

    compacted = tool_result_budget(messages, repo_root=tmp_path, max_chars=100)

    content = compacted[-1]["content"][0]["content"]
    assert content.startswith("<persisted-output>")
    assert (tmp_path / ".osc_agent" / "tool-results" / "big.txt").read_text(encoding="utf-8") == "A" * 300


def test_apply_compaction_runs_pipeline_without_orphan_results(tmp_path):
    messages = [{"role": "user", "content": "start"}]
    for index in range(30):
        messages.extend(_tool_pair(index, content="x" * 300))

    compacted = apply_compaction(messages, repo_root=tmp_path)

    assert estimate_size(compacted) < estimate_size(messages)
    for position, message in enumerate(compacted):
        if isinstance(message.get("content"), list):
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    previous = compacted[position - 1]
                    assert any(item.get("id") == block["tool_use_id"] for item in previous["content"])


def test_reactive_compact_keeps_recent_tail_and_writes_transcript(tmp_path):
    messages = [{"role": "user", "content": "start"}]
    for index in range(6):
        messages.extend(_tool_pair(index))

    compacted = reactive_compact(messages, repo_root=tmp_path)

    assert str(compacted[0]["content"]).startswith("[Reactive compact]")
    assert any((tmp_path / ".osc_agent" / "transcripts").glob("reactive-*.jsonl"))


class PromptTooLongMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("prompt_too_long")
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
        )


class FakeClient:
    def __init__(self) -> None:
        self.messages = PromptTooLongMessages()


def test_agent_loop_reactive_compacts_after_prompt_too_long(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    client = FakeClient()

    response = agent_loop(messages, client=client, settings=_settings(), repo_root=tmp_path)

    assert response.stop_reason == "end_turn"
    assert client.messages.calls == 2
    assert str(messages[0]["content"]).startswith("[Reactive compact]")
    assert any((tmp_path / ".osc_agent" / "transcripts").glob("reactive-*.jsonl"))


def test_compact_tool_is_registered_and_handler_compacts(tmp_path):
    messages = [{"role": "user", "content": "hello"}]
    handlers = build_tool_handlers(tmp_path, messages=messages)

    result = handlers["compact"](reason="manual")

    assert "compact" in {tool["name"] for tool in TOOLS}
    assert result == "[Compacted. History summarized.]"
    assert str(messages[0]["content"]).startswith("[Compacted: manual]")
