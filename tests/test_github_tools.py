from __future__ import annotations

from datetime import datetime, timezone
import json

from osc_agent.tools.github import apply_issue_scores, filter_candidate_issues, load_issues_file, parse_github_repo


def test_parse_github_repo_accepts_standard_url():
    assert parse_github_repo("https://github.com/example/project") == ("example", "project")


def test_filter_candidate_issues_rejects_assigned_and_claimed():
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    issues = [
        {
            "number": 1,
            "title": "Good first bug",
            "state": "open",
            "labels": [{"name": "good first issue"}],
            "updated_at": now,
            "assignee": None,
            "assignees": [],
            "body": "Expected behavior is clear and steps to reproduce are provided.",
        },
        {
            "number": 2,
            "title": "Claimed issue",
            "state": "open",
            "labels": [{"name": "bug"}],
            "updated_at": now,
            "assignee": None,
            "assignees": [],
            "body": "Expected behavior is clear and steps to reproduce are provided.",
        },
        {
            "number": 3,
            "title": "Assigned issue",
            "state": "open",
            "labels": [{"name": "enhancement"}],
            "updated_at": now,
            "assignee": {"login": "maintainer"},
            "assignees": [{"login": "maintainer"}],
            "body": "Expected behavior is clear and steps to reproduce are provided.",
        },
        {
            "number": 4,
            "title": "Already has a PR",
            "state": "open",
            "labels": [{"name": "bug"}],
            "updated_at": now,
            "assignee": None,
            "assignees": [],
            "body": "Expected behavior is clear and steps to reproduce are provided.",
            "activity": {"linked_pull_requests": [{"number": 99, "state": "open"}]},
        },
    ]

    candidates = filter_candidate_issues(issues, {2: [{"body": "I'll take this"}]})

    assert [issue["number"] for issue in candidates] == [1]


def test_load_issues_file_supports_object_shape(tmp_path):
    path = tmp_path / "issues.json"
    path.write_text(json.dumps({"issues": [{"number": 1}], "comments_by_issue": {"1": []}}), encoding="utf-8")

    issues, comments = load_issues_file(str(path))

    assert issues == [{"number": 1}]
    assert comments == {"1": []}


def test_issue_review_levels_are_explainable_and_reject_low_candidates():
    candidates = [{"number": 1}, {"number": 2}]
    scores = [
        {"number": 1, "level": "HIGH", "dimensions": {"clarity": "clear"}, "reason": "small"},
        {"number": 2, "level": "REJECT", "rejection_reason": "linked PR exists"},
    ]

    ranked = apply_issue_scores(candidates, scores)

    assert [item["number"] for item in ranked] == [1]
    assert ranked[0]["review_level"] == "HIGH"
    assert ranked[0]["review_dimensions"]["clarity"] == "clear"
