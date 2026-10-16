"""Tests for the reviewer's disagreement → issue escalation flow."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from maestro_review.github import GitHubError
from maestro_review.review import PRReviewer, ReviewFinding, ReviewThread

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

MULTI_FILE_DIFF = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 line1
+added
 line3
 line4
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,3 +5,4 @@
 existing
+new_line
 more
"""


def _make_finding(
    file: str = "maestro/review.py",
    title: str = "Unused import detected",
    severity: str = "suggestion",
) -> ReviewFinding:
    return ReviewFinding(
        file=file, line=10, severity=severity,
        title=title, description="desc", fix_plan="fix",
    )


def _graphql_thread_node(
    *,
    thread_id: str = "PRRT_1",
    is_resolved: bool = False,
    is_outdated: bool = False,
    comment_id: int = 100,
    path: str = "maestro/review.py",
    body: str = "\U0001f4a1 **Unused import detected**\n\nos is unused",
    reply_bodies: list[str] | None = None,
) -> dict[str, object]:
    comments: list[dict[str, object]] = [
        {"databaseId": comment_id, "path": path, "body": body},
    ]
    for i, reply in enumerate(reply_bodies or []):
        comments.append({
            "databaseId": comment_id + 1 + i,
            "path": path,
            "body": reply,
        })
    return {
        "id": thread_id,
        "isResolved": is_resolved,
        "isOutdated": is_outdated,
        "comments": {"nodes": comments},
    }


def _reviewer(host: MagicMock | None = None) -> PRReviewer:
    if host is None:
        host = MagicMock()
        host.list_issues.return_value = []
        host.get_pr_review_threads.return_value = []
    return PRReviewer("owner/repo", code_host=host)


# ------------------------------------------------------------------
# _find_matching_issue
# ------------------------------------------------------------------


class TestFindMatchingIssue:
    def test_returns_matching_issue(self) -> None:
        finding = _make_finding()
        issues = [
            {"number": 42, "title": "review.py: unused import and dead code",
             "body": "maestro/review.py has unused imports"},
        ]
        result = PRReviewer._find_matching_issue(finding, issues)
        assert result is not None
        assert result["number"] == 42

    def test_returns_none_when_no_match(self) -> None:
        finding = _make_finding()
        issues = [
            {"number": 10, "title": "Unrelated issue", "body": "nothing"},
        ]
        assert PRReviewer._find_matching_issue(finding, issues) is None

    def test_returns_none_for_empty_issues(self) -> None:
        assert PRReviewer._find_matching_issue(_make_finding(), []) is None

    def test_returns_first_match(self) -> None:
        finding = _make_finding()
        issues = [
            {"number": 1, "title": "review.py unused import", "body": "maestro/review.py detected"},
            {"number": 2, "title": "review.py unused import again", "body": "maestro/review.py found"},
        ]
        result = PRReviewer._find_matching_issue(finding, issues)
        assert result is not None
        assert result["number"] == 1


# ------------------------------------------------------------------
# _filter_tracked_findings — now applies to every severity
# ------------------------------------------------------------------


class TestFilterTrackedFindingsAllSeverities:
    """All severities are filtered when a matching open issue exists."""

    def test_filters_suggestion_tracked_by_issue(self) -> None:
        host = MagicMock()
        host.list_issues.return_value = [
            {"number": 42, "title": "review.py: unused import",
             "body": "maestro/review.py unused imports detected"},
        ]
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        result = reviewer._filter_tracked_findings([finding])

        assert result == []

    def test_filters_nit_tracked_by_issue(self) -> None:
        host = MagicMock()
        host.list_issues.return_value = [
            {"number": 42, "title": "review.py: unused import",
             "body": "maestro/review.py unused imports detected"},
        ]
        reviewer = _reviewer(host)
        finding = _make_finding(severity="nit")

        result = reviewer._filter_tracked_findings([finding])

        assert result == []

    def test_passes_through_untracked_findings(self) -> None:
        host = MagicMock()
        host.list_issues.return_value = []
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        result = reviewer._filter_tracked_findings([finding])

        assert result == [finding]


# ------------------------------------------------------------------
# _thread_disputes_finding
# ------------------------------------------------------------------


class TestThreadDisputesFinding:
    def test_dispute_marker_on_last_reply(self) -> None:
        thread = ReviewThread(
            thread_id="T1", comment_id=1, path="f.py",
            body="bug", all_bodies="bug",
            reply_bodies=["Could not prove this defect with a failing test."],
        )
        assert PRReviewer._thread_disputes_finding(thread) is True

    def test_legacy_bug_wording_still_recognized(self) -> None:
        thread = ReviewThread(
            thread_id="T1", comment_id=1, path="f.py",
            body="bug", all_bodies="bug",
            reply_bodies=["Could not prove this bug with a failing test."],
        )
        assert PRReviewer._thread_disputes_finding(thread) is True

    def test_no_replies(self) -> None:
        thread = ReviewThread(
            thread_id="T1", comment_id=1, path="f.py",
            body="bug", all_bodies="bug", reply_bodies=[],
        )
        assert PRReviewer._thread_disputes_finding(thread) is False

    def test_dispute_overridden_by_later_reply(self) -> None:
        """Only the last reply counts — a later reply (e.g. claim retry) cancels it."""
        thread = ReviewThread(
            thread_id="T1", comment_id=1, path="f.py",
            body="bug", all_bodies="bug",
            reply_bodies=[
                "Could not prove this defect",
                "\u23f3 Working on a fix...",
            ],
        )
        assert PRReviewer._thread_disputes_finding(thread) is False


# ------------------------------------------------------------------
# _handle_disagreements — disagreement → issue flow
# ------------------------------------------------------------------


class TestHandleDisagreements:
    def test_escalates_disputed_finding_creates_issue(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=[
                    "Could not prove this defect with a failing test. "
                    "Leaving for human review.",
                ],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == []
        assert escalated == [finding]
        host.create_issue.assert_called_once()
        title, _body = host.create_issue.call_args[0]
        assert "Unused import" in title
        assert "suggestion" in host.create_issue.call_args[1]["labels"]
        host.reply_to_review_comment.assert_called_once()
        _pr, comment_id, body = host.reply_to_review_comment.call_args[0]
        assert comment_id == 100
        assert "Tracked in #99" in body

    def test_reply_failure_still_escalates_finding(self) -> None:
        """A transient reply failure after the issue is created must not
        propagate — the issue exists, so the finding is escalated (removed
        from the in-flight list), not left to be re-posted as a comment."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        host.reply_to_review_comment.side_effect = GitHubError("network blip")
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == []
        assert escalated == [finding]
        host.create_issue.assert_called_once()
        host.reply_to_review_comment.assert_called_once()

    def test_reply_failure_logs_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The orphaned-issue scenario logs a warning so the silent-orphan
        regression stays locked."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        host.reply_to_review_comment.side_effect = GitHubError("network blip")
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        with caplog.at_level(logging.WARNING, logger="maestro_review"):
            reviewer._handle_disagreements(10, [finding])

        assert "issue created but thread not linked" in caplog.text
        assert "Tracked in #99" in caplog.text

    def test_reply_failure_skips_in_memory_mirror(self) -> None:
        """On the failure path the thread state must NOT be annotated — the
        reply never landed on GitHub, so the mirror would falsely short-circuit
        a sibling finding away from re-tracking on a later run."""
        host = MagicMock()
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        host.reply_to_review_comment.side_effect = GitHubError("network blip")
        reviewer = _reviewer(host)
        thread = ReviewThread(
            thread_id="PRRT_1", comment_id=100, path="maestro/review.py",
            body="\U0001f4a1 **Unused import detected**\nos is unused",
            all_bodies="\U0001f4a1 **Unused import detected**\nos is unused\n"
                       "Could not prove this defect.",
            reply_bodies=["Could not prove this defect."],
            is_resolved=False, is_outdated=False,
        )
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(
            10, [finding], threads=[thread],
        )

        assert escalated == [finding]
        assert remaining == []
        assert "Tracked in #99" not in thread.all_bodies
        assert all("Tracked in #99" not in b for b in thread.reply_bodies)

    def test_escalates_disputed_finding_reuses_existing_issue(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.list_issues.return_value = [
            {"number": 42, "title": "review.py: unused import",
             "body": "maestro/review.py has unused imports detected"},
        ]
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == []
        assert escalated == [finding]
        host.create_issue.assert_not_called()
        body = host.reply_to_review_comment.call_args[0][2]
        assert "Tracked in #42" in body

    def test_legacy_bug_wording_triggers_escalation(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this bug with a failing test."],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 7
        reviewer = _reviewer(host)
        finding = _make_finding(severity="bug")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert escalated == [finding]
        assert remaining == []
        host.create_issue.assert_called_once()

    def test_skips_thread_already_linked_to_issue(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=[
                    "Could not prove this defect.",
                    "Tracked in #42",
                ],
            ),
        ]
        host.list_issues.return_value = []
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == [finding]
        assert escalated == []
        host.create_issue.assert_not_called()
        host.reply_to_review_comment.assert_not_called()

    def test_skips_resolved_disputed_thread(self) -> None:
        """Resolved threads must not be escalated even when they end with a
        ``Could not prove`` reply — the human resolution wins."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                is_resolved=True,
                reply_bodies=[
                    "Could not prove this defect with a failing test.",
                ],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert escalated == []
        assert remaining == [finding]
        host.create_issue.assert_not_called()
        host.reply_to_review_comment.assert_not_called()

    def test_skips_outdated_disputed_thread(self) -> None:
        """Outdated threads (commented code has changed) must not be escalated
        — escalation would contradict the ``skip outdated`` invariant."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                is_outdated=True,
                reply_bodies=[
                    "Could not prove this defect with a failing test.",
                ],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert escalated == []
        assert remaining == [finding]
        host.create_issue.assert_not_called()
        host.reply_to_review_comment.assert_not_called()

    def test_skips_thread_without_dispute_reply(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["\u23f3 Working on a fix..."],
            ),
        ]
        host.list_issues.return_value = []
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == [finding]
        assert escalated == []
        host.create_issue.assert_not_called()

    def test_skips_finding_with_no_matching_thread(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(path="other/file.py"),
        ]
        host.list_issues.return_value = []
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == [finding]
        assert escalated == []

    def test_no_threads_returns_findings_unchanged(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = []
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == [finding]
        assert escalated == []

    def test_empty_findings_short_circuits(self) -> None:
        host = MagicMock()
        reviewer = _reviewer(host)

        remaining, escalated = reviewer._handle_disagreements(10, [])

        assert remaining == []
        assert escalated == []
        host.get_pr_review_threads.assert_not_called()

    def test_disputed_finding_with_suggested_code(self) -> None:
        """Suggested-code findings embed the snippet in the escalated issue body."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = 12
        reviewer = _reviewer(host)
        finding = ReviewFinding(
            file="maestro/review.py", line=10, severity="suggestion",
            title="Unused import detected",
            description="desc", fix_plan="",
            suggested_code="import sys",
        )

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == []
        assert escalated == [finding]
        _title, body = host.create_issue.call_args[0]
        assert "Suggested code" in body
        assert "import sys" in body

    def test_issue_creation_failure_leaves_finding_in_flight(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.list_issues.return_value = []
        host.create_issue.return_value = None
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == [finding]
        assert escalated == []
        host.reply_to_review_comment.assert_not_called()

    def test_second_finding_matching_same_thread_does_not_create_duplicate(
        self,
    ) -> None:
        """A single thread containing two bold-titled blocks must not be
        escalated twice: after the first escalation the helper mirrors its
        ``Tracked in #N`` reply into ``thread.reply_bodies`` /
        ``thread.all_bodies`` so the second finding short-circuits via
        ``_thread_has_issue_link``."""
        root_body = (
            "\U0001f4a1 **Title A**\n\nDetails for A.\n\n"
            "\U0001f4a1 **Title B**\n\nDetails for B."
        )
        thread_node = {
            "id": "PRRT_1",
            "isResolved": False,
            "comments": {"nodes": [
                {"databaseId": 100, "path": "maestro/review.py",
                 "body": root_body},
                {"databaseId": 101, "path": "maestro/review.py",
                 "body": "Could not prove this defect with a failing test."},
            ]},
        }
        host = MagicMock()
        host.get_pr_review_threads.return_value = [thread_node]
        host.list_issues.return_value = []
        host.create_issue.side_effect = [201, 202]
        reviewer = _reviewer(host)

        finding_a = _make_finding(title="Title A")
        finding_b = _make_finding(title="Title B")
        remaining, escalated = reviewer._handle_disagreements(
            10, [finding_a, finding_b],
        )

        assert host.create_issue.call_count == 1
        assert host.reply_to_review_comment.call_count == 1
        assert escalated == [finding_a]
        assert remaining == [finding_b]

    def test_matching_issue_without_number_falls_through_to_creation(self) -> None:
        """If `_find_matching_issue` returns a dict missing 'number', the handler
        falls through to creating a fresh issue rather than posting 'Tracked in #None'."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        # Matching issue present but the 'number' key is absent.
        host.list_issues.return_value = [
            {"title": "review.py unused import detected",
             "body": "maestro/review.py unused imports detected"},
        ]
        host.create_issue.return_value = 55
        reviewer = _reviewer(host)
        finding = _make_finding(severity="suggestion")

        remaining, escalated = reviewer._handle_disagreements(10, [finding])

        assert remaining == []
        assert escalated == [finding]
        host.create_issue.assert_called_once()
        body = host.reply_to_review_comment.call_args[0][2]
        assert "Tracked in #55" in body
        assert "#None" not in body


# ------------------------------------------------------------------
# post_review — escalation reply failure must not abort the pass
# ------------------------------------------------------------------


class TestPostReviewEscalationResilience:
    def test_reply_failure_does_not_abort_remaining_findings(self) -> None:
        """When a disputed finding's `Tracked in #N` reply raises, post_review
        must still post the other (non-disputed) findings — the reply failure
        cannot abort the issue-tracking filter, categorization, and posting
        stages that follow `_handle_disagreements`."""
        host = MagicMock()
        host.list_issues.return_value = []
        host.create_issue.return_value = 99
        host.reply_to_review_comment.side_effect = GitHubError("network blip")
        # One thread matching the disputed finding; the second finding has no
        # matching thread, so it must flow through to a new inline comment.
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        reviewer = _reviewer(host)
        disputed = _make_finding(title="Unused import detected")
        fresh = _make_finding(title="Dead code branch")

        result = reviewer.post_review(10, [disputed, fresh], "Summary")

        # The escalation reply was attempted once and failed — but post_review
        # returned normally and posted the surviving finding.
        host.reply_to_review_comment.assert_called_once()
        host.post_pr_review.assert_called_once()
        _pr, payload = host.post_pr_review.call_args[0]
        posted_bodies = " ".join(c["body"] for c in payload["comments"])
        assert "Dead code branch" in posted_bodies
        assert "Unused import detected" not in posted_bodies
        assert result.disputed == [disputed]
        assert result.posted == [fresh]


# ------------------------------------------------------------------
# _categorize_by_thread_state — nit/suggestion not reopened
# ------------------------------------------------------------------


class TestCategorizeNitsNotReopened:
    def test_nit_in_resolved_thread_skipped_not_reopened(self) -> None:
        """Nits in resolved threads without issue links are skipped (not reopened)."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(is_resolved=True),
        ]
        reviewer = _reviewer(host)
        nit = _make_finding(severity="nit")

        new, skip, reopen = reviewer._categorize_by_thread_state(10, [nit])

        assert new == []
        assert len(skip) == 1
        assert reopen == []

    def test_suggestion_in_resolved_thread_skipped(self) -> None:
        host = MagicMock()
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(is_resolved=True),
        ]
        reviewer = _reviewer(host)
        suggestion = _make_finding(severity="suggestion")

        new, skip, reopen = reviewer._categorize_by_thread_state(10, [suggestion])

        assert new == []
        assert len(skip) == 1
        assert reopen == []

    def test_bug_in_resolved_thread_reopened(self) -> None:
        """Bugs in resolved threads without issue links ARE reopened."""
        host = MagicMock()
        host.get_pr_review_threads.return_value = [{
            "id": "PRRT_1",
            "isResolved": True,
            "comments": {"nodes": [
                {"databaseId": 100, "path": "maestro/review.py",
                 "body": "\U0001f41b **Unused import detected**\n\nos is unused"},
            ]},
        }]
        reviewer = _reviewer(host)
        bug = ReviewFinding(
            file="maestro/review.py", line=10, severity="bug",
            title="Unused import detected", description="desc", fix_plan="fix",
        )

        new, skip, reopen = reviewer._categorize_by_thread_state(10, [bug])

        assert new == []
        assert skip == []
        assert len(reopen) == 1


# ------------------------------------------------------------------
# post_review — disagreement integration
# ------------------------------------------------------------------


class TestPostReviewDisputed:
    def test_disputed_finding_does_not_re_post(self) -> None:
        """When a thread disputes a finding, it is escalated to an issue and
        removed from the in-flight list — not re-posted as a new comment."""
        host = MagicMock()
        host.list_issues.return_value = []
        host.get_pr_review_threads.return_value = [
            _graphql_thread_node(
                reply_bodies=["Could not prove this defect."],
            ),
        ]
        host.create_issue.return_value = 77
        reviewer = _reviewer(host)

        findings = [_make_finding(severity="suggestion")]
        result = reviewer.post_review(10, findings, "Summary", diff=MULTI_FILE_DIFF)

        assert len(result.disputed) == 1
        assert result.posted == []
        host.create_issue.assert_called_once()
        # No new review payload should have been posted.
        host.post_pr_review.assert_not_called()

    def test_undisputed_finding_posts_normally(self) -> None:
        """A finding with no matching thread is posted as a new inline comment."""
        host = MagicMock()
        host.list_issues.return_value = []
        host.get_pr_review_threads.return_value = []
        reviewer = _reviewer(host)

        bug = ReviewFinding(
            file="a.py", line=2, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix",
        )
        result = reviewer.post_review(10, [bug], "Summary", diff=MULTI_FILE_DIFF)

        assert result.disputed == []
        assert result.posted == [bug]
        host.create_issue.assert_not_called()
