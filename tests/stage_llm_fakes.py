from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from osc_agent.config import Settings


@dataclass
class _Response:
    content: list[dict[str, Any]]


class _Messages:
    def create(self, **kwargs: Any) -> _Response:
        tool_name = str(kwargs["tools"][0]["name"])
        prompt = str(kwargs["messages"][0]["content"])
        if tool_name == "submit_issue_scores":
            payload: dict[str, Any] = {"scores": []}
        elif tool_name == "submit_analysis":
            number_match = re.search(r'"number"\s*:\s*(\d+)', prompt)
            title_match = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', prompt)
            if number_match:
                number = int(number_match.group(1))
                title = json.loads(f'"{title_match.group(1)}"') if title_match else "Contribution"
                direction = {
                    "name": f"Issue #{number}: {title}",
                    "description": f"Address the scoped request in {title}.",
                    "source": f"Issue #{number}",
                    "source_kind": "issue",
                    "issue_number": number,
                    "entry": "issue",
                    "effort": "small",
                    "interview": "Shows evidence-driven scope control",
                    "risk": "Requires focused verification",
                }
            else:
                direction = {
                    "name": "Improve repository architecture",
                    "description": "Address a focused gap found in repository evidence.",
                    "source": "Repository analysis",
                    "source_kind": "architecture",
                    "issue_number": None,
                    "entry": "README.md",
                    "effort": "small",
                    "interview": "Shows evidence-driven scope control",
                    "risk": "Requires maintainer confirmation",
                }
            payload = {
                "top_directions": [direction],
                "analysis_summary": "A focused, reviewable contribution direction is available.",
                "architecture_insights": [],
            }
        elif tool_name == "submit_design":
            payload = {
                "problem_boundary": "Implement the selected contribution as a focused change.",
                "files_to_modify": ["README.md"],
                "allowed_files": ["README.md"],
                "target_symbols": [],
                "requirements": [{
                    "text": "Document the selected contribution.",
                    "source_excerpt": "Selected contribution",
                }],
                "task_type": "docs",
                "acceptance_checks": [{
                    "criterion": "Documentation is reviewed",
                    "command": "",
                    "manual_check": True,
                    "requirement_ids": ["REQ-1"],
                }],
            }
        elif tool_name == "submit_pr_draft":
            payload = {
                "title": "docs: implement selected contribution",
                "problem": "The selected contribution was not yet addressed.",
                "solution": "Apply the scoped design and verify it.",
                "reviewer_notes": ["Review the focused scope"],
            }
        else:
            raise AssertionError(f"unexpected stage tool: {tool_name}")
        return _Response([{"type": "tool_use", "name": tool_name, "input": payload}])


class FakeDiscoverClient:
    def __init__(self) -> None:
        self.messages = _Messages()


def fake_settings() -> Settings:
    return Settings(anthropic_api_key="test-key", model_id="test-model")


def run_fake_discover(discover, **kwargs: Any):
    kwargs.setdefault("client", FakeDiscoverClient())
    kwargs.setdefault("settings", fake_settings())
    return discover(**kwargs)
