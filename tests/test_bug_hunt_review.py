"""Bug-hunting tests for maestro_review package.

Each test targets a suspected bug in the source code. Tests that reveal
real bugs are kept; tests that pass (no bug) are deleted.
"""

from __future__ import annotations

from maestro_review.review import PRReviewer, ReviewThread


class TestParseReviewThreadsNullBody:
    """Bug: _parse_review_threads crashes when GraphQL returns comments
    with ``"body": null`` (key present but value is None).

    ``c.get("body", "")`` returns None when the key exists with a null value,
    and ``"\\n".join(...)`` raises TypeError on None items.
    """

    def test_null_body_in_comment_does_not_crash(self) -> None:
        """A comment node with body=None should not crash the parser."""
        raw_nodes = [
            {
                "id": "T1",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {
                            "databaseId": 100,
                            "path": "foo.py",
                            "body": None,  # GraphQL null
                        },
                    ],
                },
            },
        ]
        # Should not raise TypeError
        threads = PRReviewer._parse_review_threads(raw_nodes)
        assert len(threads) == 1
        assert threads[0].body == ""
        assert threads[0].all_bodies == ""

    def test_null_body_in_second_comment(self) -> None:
        """A thread where only the second comment has body=None."""
        raw_nodes = [
            {
                "id": "T2",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {
                            "databaseId": 200,
                            "path": "bar.py",
                            "body": "first comment",
                        },
                        {
                            "databaseId": 201,
                            "path": "bar.py",
                            "body": None,
                        },
                    ],
                },
            },
        ]
        threads = PRReviewer._parse_review_threads(raw_nodes)
        assert len(threads) == 1
        # all_bodies should join without crashing
        assert "first comment" in threads[0].all_bodies


class TestMatchFindingToThreadNoneBody:
    """Bug: _match_finding_to_thread crashes when thread.body is None.

    If _parse_review_threads stores None in the body field (due to the
    null-body bug above), then _match_finding_to_thread calls
    ``thread.body.lower()`` which raises AttributeError on None.
    """

    def test_thread_with_none_body_does_not_crash(self) -> None:
        from maestro_review.review import ReviewFinding

        finding = ReviewFinding(
            file="foo.py", line=10, severity="bug",
            title="Test title", description="desc", fix_plan="plan",
        )
        # Simulate a thread that somehow got None body
        thread = ReviewThread(
            thread_id="T1", comment_id=100, path="foo.py",
            body=None,  # type: ignore[arg-type]
            all_bodies="", is_resolved=False,
        )
        # Should not raise AttributeError
        result = PRReviewer._match_finding_to_thread(finding, [thread])
        assert result is None


class TestThreadHasIssueLinkNullBodies:
    """Bug: _thread_has_issue_link crashes when all_bodies is None."""

    def test_none_all_bodies_does_not_crash(self) -> None:
        thread = ReviewThread(
            thread_id="T1", comment_id=100, path="foo.py",
            body="", all_bodies=None,  # type: ignore[arg-type]
            is_resolved=True,
        )
        # Should not raise TypeError from re.search on None
        result = PRReviewer._thread_has_issue_link(thread)
        assert result is False


