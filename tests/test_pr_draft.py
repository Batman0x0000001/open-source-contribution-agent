from __future__ import annotations

from osc_agent.agent_loop import build_tool_handlers
from osc_agent.tools.pr import build_pr_draft, format_pr_draft


def test_build_pr_draft_uses_structured_sections_for_docs_change():
    diff = "diff --git a/README.md b/README.md\n"
    status = " M README.md"

    draft = build_pr_draft(diff=diff, status=status)
    output = format_pr_draft(draft)

    assert draft.title == "Update documentation"
    assert "## Summary" in output
    assert "## Tests" in output
    assert "## Risk" in output
    assert "README.md" in output
    assert "documentation-only" in output


def test_agent_handlers_expose_draft_pr(tmp_path):
    handlers = build_tool_handlers(tmp_path)

    assert "draft_pr" in handlers
