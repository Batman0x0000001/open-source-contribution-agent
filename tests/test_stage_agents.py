from __future__ import annotations

from dataclasses import dataclass

from osc_agent.config import Settings
from osc_agent.workflows.contribution.agents import (
    run_design_generation,
    run_discover_analysis,
    run_pr_draft_generation,
    score_candidate_issues,
)


@dataclass
class _Response:
    content: list[dict]


class _Messages:
    def __init__(self, tool_name: str, payload: dict):
        self.tool_name = tool_name
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(
            [
                {
                    "type": "tool_use",
                    "name": self.tool_name,
                    "input": self.payload,
                }
            ]
        )


class _Client:
    def __init__(self, tool_name: str, payload: dict):
        self.messages = _Messages(tool_name, payload)


class _FailingMessages:
    def create(self, **kwargs):
        raise RuntimeError("network unavailable")


class _FailingClient:
    messages = _FailingMessages()


def _settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        anthropic_base_url=None,
        model_id="test-model",
        fallback_model_id=None,
    )


def test_run_discover_analysis_returns_tool_input():
    payload = {
        "top_directions": [
            {
                "name": "Add retry tests",
                "description": "Cover transient failure behavior",
                "source": "code analysis",
                "entry": "tests/test_retry.py",
                "effort": "small",
                "interview": "Shows verification discipline",
                "risk": "low",
            }
        ],
        "analysis_summary": "A focused test contribution is viable.",
        "architecture_insights": [
            {
                "dimension": "Evaluation",
                "current": "Unit tests exist",
                "gap": "Retry paths are not covered",
                "impact": "medium",
                "improvement": "Add focused tests",
                "scope": "1 file",
                "interview_angle": "Testing strategy",
                "location": "tests/",
            }
        ],
    }
    client = _Client("submit_analysis", payload)

    result = run_discover_analysis(client, _settings(), {"repo_overview": "demo"})

    assert result == payload
    assert client.messages.calls[0]["tools"][0]["name"] == "submit_analysis"


def test_run_design_generation_returns_tool_input():
    payload = {
        "problem_boundary": "Add retry coverage.",
        "out_of_scope": ["No API changes"],
        "success_criteria": ["Focused tests pass"],
        "options": [{"name": "Test only", "idea": "Add tests", "pros": "Small", "cons": "Limited"}],
        "recommended": "Test only",
        "implementation_plan": "Modify tests/test_retry.py.",
        "files_to_modify": ["tests/test_retry.py"],
        "tests_to_run": ["python -m pytest tests/test_retry.py"],
        "task_type": "behavior",
        "baseline_checks": [
            {
                "command": "python -m pytest tests/test_retry.py",
                "expected_exit_codes": [1],
                "output_contains": "retry failed",
            }
        ],
        "reproduction_mode": "existing",
        "reproduction_test_files": [],
        "maintainer_comment": "Would focused retry tests be useful?",
        "interview_story": "I scoped the contribution to verifiable behavior.",
    }
    client = _Client("submit_design", payload)

    result = run_design_generation(client, _settings(), {"top_directions": []}, "Add retry coverage")

    assert result == payload
    assert client.messages.calls[0]["tools"][0]["name"] == "submit_design"
    properties = client.messages.calls[0]["tools"][0]["input_schema"]["properties"]
    assert "reproduction_mode" in properties
    assert properties["forbidden_paths"]["minItems"] == 1
    assert properties["baseline_checks"]["items"]["properties"]["expected_exit_codes"]["items"]["minimum"] == 1
    assert "never include symbols that the implementation will add" in properties["target_symbols"]["description"]
    assert "REQ-1 through REQ-N" in properties["acceptance_checks"]["description"]
    assert "forbidden_paths must be a non-empty list" in client.messages.calls[0]["system"]
    assert "union of acceptance_checks.requirement_ids" in client.messages.calls[0]["system"]
    assert "Successful import, smoke, or syntax commands" in client.messages.calls[0]["system"]


def test_score_candidate_issues_returns_scores():
    payload = {
        "scores": [
            {
                "number": 1,
                "title": "Add retry tests",
                "level": "HIGH",
                "dimensions": {
                    "clarity": "clear",
                    "unclaimed": "unclaimed",
                    "scope": "small",
                    "testability": "good",
                    "reviewability": "good",
                },
                "reason": "Clear and scoped",
                "rejection_reason": "",
            }
        ]
    }
    client = _Client("submit_issue_scores", payload)

    result = score_candidate_issues(
        client,
        _settings(),
        [{"number": 1, "title": "Add retry tests", "body": "Expected behavior is clear."}],
        {1: []},
    )

    assert result == payload["scores"]
    assert client.messages.calls[0]["tools"][0]["name"] == "submit_issue_scores"


def test_run_pr_draft_generation_returns_tool_input():
    payload = {
        "title": "test(agent): add retry coverage",
        "problem": "Retry behavior was not covered.",
        "solution": "Add focused tests.",
        "reviewer_notes": ["Review expected retry count"],
    }
    client = _Client("submit_pr_draft", payload)

    result = run_pr_draft_generation(client, _settings(), {"git_diff": "diff"})

    assert result == payload
    assert client.messages.calls[0]["tools"][0]["name"] == "submit_pr_draft"


def test_stage_agent_raises_when_client_call_fails():
    import pytest

    with pytest.raises(RuntimeError, match="stage model request failed"):
        run_discover_analysis(_FailingClient(), _settings(), {"repo_overview": "demo"})
