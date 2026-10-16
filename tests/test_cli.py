"""Tests for maestro_review.cli — all external calls are mocked."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from maestro_review import cli as review_pr_mod
from maestro_review.review import (
    OverflowFinding,
    PostReviewResult,
    ReviewThread,
    UnrepliedComments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    """Minimal stand-in for ReviewResult."""

    summary: str
    findings: list[object]
    diff: str | None = None


@dataclass
class FakeFinding:
    severity: str
    file: str = "a.py"
    line: int = 1
    title: str = "T"
    description: str = "D"


BUG_FINDING = FakeFinding(severity="bug")
SUGGESTION_FINDING = FakeFinding(severity="suggestion")

BASE_ARGV = ["maestro-review", "--repo", "owner/repo", "--pr", "1"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostFailureExitsOne:
    """post_review() exception must exit 1 to surface infra failures in CI."""

    def test_post_failure_with_bugs_exits_one(self) -> None:
        """When post_review raises and there are bug findings, exit 1."""
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.side_effect = RuntimeError("API down")

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1

    def test_post_failure_with_suggestions_exits_one(self) -> None:
        """When post_review raises and there are only suggestions, exit 1."""
        result = FakeResult(summary="s", findings=[SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.side_effect = RuntimeError("API down")

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1


class TestReviewExceptionExitsOne:
    """When reviewer.review() itself fails, exit 1."""

    def test_review_exception_exits_one(self) -> None:
        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer"),
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.side_effect = RuntimeError("SDK crash")

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1


def _post_result(
    posted: list[object] | None = None,
    reopened: list[object] | None = None,
    filtered: list[object] | None = None,
    skipped: list[object] | None = None,
) -> PostReviewResult:
    """Build a PostReviewResult with optional overrides."""
    return PostReviewResult(
        posted=posted or [],   # type: ignore[arg-type]
        reopened=reopened or [],  # type: ignore[arg-type]
        filtered=filtered or [],  # type: ignore[arg-type]
        skipped=skipped or [],  # type: ignore[arg-type]
    )


class TestNormalFlow:
    """Happy-path exit codes."""

    def test_bug_findings_exit_one(self) -> None:
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                posted=[BUG_FINDING],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1
            mock_cls.return_value.post_review.assert_called_once()

    def test_suggestion_findings_exit_one(self) -> None:
        """A posted suggestion now blocks the PR — any comment fails the job."""
        result = FakeResult(summary="s", findings=[SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                posted=[SUGGESTION_FINDING],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1
            mock_cls.return_value.post_review.assert_called_once()

    def test_no_findings_exit_zero(self) -> None:
        result = FakeResult(summary="clean", findings=[])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result()
            review_pr_mod.main()  # should not raise

    def test_bug_findings_all_filtered_exit_zero(self) -> None:
        """When all bug findings are filtered (tracked by issues), exit 0."""
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                filtered=[BUG_FINDING],
            )
            review_pr_mod.main()  # should not raise — all bugs were filtered


class TestAnyPostedFindingBlocks:
    """Any newly posted or reopened finding blocks the PR, regardless of severity.

    The exit gate keys off ``post_result.posted + reopened`` — the freshly
    produced comments — not the finding severity. So a bug, a suggestion, a nit,
    or even a hallucinated severity value like 'warning' all exit 1 once posted.
    Findings that were filtered/skipped (already tracked or threaded) are not in
    that set and do not block (covered by TestNormalFlow / TestPostReviewLogBranches).
    """

    def test_hallucinated_severity_posted_exits_one_normal_flow(self) -> None:
        """A posted finding with a hallucinated severity blocks the PR."""
        hallucinated = FakeFinding(severity="warning")
        result = FakeResult(summary="s", findings=[hallucinated])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                posted=[hallucinated],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1

    def test_hallucinated_severity_posted_exits_one_dry_run(self) -> None:
        """Dry-run with an actioned hallucinated-severity finding also exits 1."""
        hallucinated = FakeFinding(severity="critical")
        result = FakeResult(summary="s", findings=[hallucinated])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            reviewer = mock_cls.return_value
            reviewer._filter_tracked_findings.return_value = [hallucinated]
            reviewer._categorize_by_thread_state.return_value = (
                [hallucinated], [], [],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1
            reviewer.post_review.assert_not_called()

    def test_nit_posted_exits_one(self) -> None:
        """A posted nit — the lowest severity — still blocks the PR."""
        nit = FakeFinding(severity="nit")
        result = FakeResult(summary="s", findings=[nit])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                posted=[nit],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1

    def test_reopened_finding_exits_one(self) -> None:
        """A reopened finding (defect persists) also blocks the PR."""
        bug = FakeFinding(severity="bug")
        result = FakeResult(summary="s", findings=[bug])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                reopened=[bug],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1


class TestDryRun:
    """Dry-run mode prints findings and never calls post_review."""

    def test_dry_run_bug_exits_one(self) -> None:
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1
            mock_cls.return_value.post_review.assert_not_called()

    def test_dry_run_suggestion_exits_one(self) -> None:
        """A suggestion that would be posted blocks the PR in dry-run too."""
        result = FakeResult(summary="s", findings=[SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            reviewer = mock_cls.return_value
            reviewer._filter_tracked_findings.return_value = [SUGGESTION_FINDING]
            reviewer._categorize_by_thread_state.return_value = (
                [SUGGESTION_FINDING], [], [],
            )

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            assert exc.value.code == 1
            reviewer.post_review.assert_not_called()

    def test_dry_run_filter_with_findings_uses_categorize_result(self) -> None:
        """Covers the success path of dry-run filtering (cli.py:116): the
        filtered list is non-empty so _categorize_by_thread_state is invoked
        and its ``new`` + ``reopen`` findings become the actioned set."""
        result = FakeResult(summary="s", findings=[BUG_FINDING, SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            reviewer = mock_cls.return_value
            # Real list return types so tuple-unpacking in cli.py:113 works.
            reviewer._filter_tracked_findings.return_value = [BUG_FINDING]
            reviewer._categorize_by_thread_state.return_value = ([BUG_FINDING], [], [])

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            # A bug survived filtering → exit 1.
            assert exc.value.code == 1
            reviewer._filter_tracked_findings.assert_called_once_with(
                result.findings,
            )
            reviewer._categorize_by_thread_state.assert_called_once()
            reviewer.post_review.assert_not_called()

    def test_dry_run_filter_empty_skips_categorize(self) -> None:
        """Covers the else branch of dry-run filtering (cli.py:117-118): when
        _filter_tracked_findings returns an empty list, the code takes the
        ``actioned = []`` path and never calls _categorize_by_thread_state."""
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            reviewer = mock_cls.return_value
            reviewer._filter_tracked_findings.return_value = []

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            # No findings actioned → exit 0 even though the raw result had a bug.
            assert exc.value.code == 0
            reviewer._filter_tracked_findings.assert_called_once_with(
                result.findings,
            )
            reviewer._categorize_by_thread_state.assert_not_called()


class TestPostReviewLogBranches:
    """Cover post_result.disputed and post_result.skipped logging."""

    def test_disputed_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        result = FakeResult(summary="s", findings=[SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
            caplog.at_level(logging.INFO, logger="maestro_review.cli"),
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                posted=[SUGGESTION_FINDING],
            )
            mock_cls.return_value.post_review.return_value.disputed = [SUGGESTION_FINDING]

            # The disputed escalation is logged before the exit gate fires; the
            # posted suggestion then fails the job.
            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

        assert exc.value.code == 1
        assert "Escalated 1 disputed finding" in caplog.text

    def test_skipped_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        result = FakeResult(summary="s", findings=[SUGGESTION_FINDING])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
            caplog.at_level(logging.INFO, logger="maestro_review.cli"),
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result(
                skipped=[SUGGESTION_FINDING],
            )
            review_pr_mod.main()

        assert "Skipped 1 finding" in caplog.text


class TestDryRunFilteringFallback:
    """Cover dry-run filtering exception fallback (lines 116-118)."""

    def test_filter_exception_falls_back_to_raw(self) -> None:
        result = FakeResult(summary="s", findings=[BUG_FINDING])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--dry-run"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value._filter_tracked_findings.side_effect = RuntimeError("API fail")

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

            # Bug finding present → exit 1 (fallback to raw findings)
            assert exc.value.code == 1


class TestTomlConfigParsingException:
    """Cover TOML parse exception (lines 57-58)."""

    def test_malformed_toml_returns_empty(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "cwd" / "maestro.toml"
        toml_file.parent.mkdir()
        toml_file.write_text("{{{invalid toml content")

        from maestro_review.cli import _load_review_defaults

        original_file = review_pr_mod.__file__
        try:
            review_pr_mod.__file__ = str(tmp_path / "nonexistent" / "maestro_review" / "cli.py")
            with patch("maestro_review.cli.Path.cwd", return_value=tmp_path / "cwd"):
                result = _load_review_defaults()
        finally:
            review_pr_mod.__file__ = original_file

        assert result == {}


class TestTomlConfigIntegration:
    """Verify maestro_review.cli loads defaults from maestro.toml."""

    def test_defaults_are_sensible(self) -> None:
        """CLI module-level defaults must have expected values."""
        assert review_pr_mod._DEFAULT_MAX_DIFF_CHARS == 80_000
        assert review_pr_mod._DEFAULT_MAX_FILES == 50
        assert review_pr_mod._DEFAULT_GH_TIMEOUT == 30

    def test_gh_timeout_passed_to_reviewer(self) -> None:
        """PRReviewer must receive hardcoded defaults when TOML has no [review] section."""
        result = FakeResult(summary="clean", findings=[])

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli._load_review_defaults", return_value={}),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result()
            review_pr_mod.main()

            mock_cls.assert_called_once_with(
                github_repo="owner/repo",
                max_diff_chars=review_pr_mod._DEFAULT_MAX_DIFF_CHARS,
                max_files=review_pr_mod._DEFAULT_MAX_FILES,
                gh_timeout=review_pr_mod._DEFAULT_GH_TIMEOUT,
            )

    def test_gh_timeout_cli_override(self) -> None:
        """--gh-timeout CLI arg must override the TOML default."""
        result = FakeResult(summary="clean", findings=[])

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--gh-timeout", "60"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result()
            review_pr_mod.main()

            mock_cls.assert_called_once_with(
                github_repo="owner/repo",
                max_diff_chars=review_pr_mod._DEFAULT_MAX_DIFF_CHARS,
                max_files=review_pr_mod._DEFAULT_MAX_FILES,
                gh_timeout=60,
            )


class TestLoadReviewDefaults:
    """Verify _load_review_defaults handles edge cases correctly."""

    def test_empty_review_section_skipped(self, tmp_path: Path) -> None:
        """A maestro.toml with an empty [review] section should not stop the search.

        The first candidate has maestro.toml without [review]; the second has it.
        The function should skip the first and return values from the second.
        """
        # First candidate: maestro.toml exists but [review] is absent
        toml_no_review = tmp_path / "cwd" / "maestro.toml"
        toml_no_review.parent.mkdir()
        toml_no_review.write_text("[timeouts]\ngit = 30\n")

        # Second candidate: maestro.toml with [review] section
        toml_with_review = tmp_path / "pkg" / "maestro.toml"
        toml_with_review.parent.mkdir()
        toml_with_review.write_text(
            "[review]\nmax_diff_chars = 50000\nmax_files = 25\ngh_timeout = 15\n"
        )

        from maestro_review.cli import _load_review_defaults

        original_file = review_pr_mod.__file__
        try:
            # Monkeypatch __file__ so parent.parent resolves to our tmp dir
            review_pr_mod.__file__ = str(tmp_path / "pkg" / "maestro_review" / "cli.py")
            with patch("maestro_review.cli.Path.cwd", return_value=tmp_path / "cwd"):
                result = _load_review_defaults()
        finally:
            review_pr_mod.__file__ = original_file

        assert result["max_diff_chars"] == 50000
        assert result["max_files"] == 25
        assert result["gh_timeout"] == 15

    def test_no_toml_returns_empty(self, tmp_path: Path) -> None:
        """When no maestro.toml exists, return empty dict."""
        from maestro_review.cli import _load_review_defaults

        original_file = review_pr_mod.__file__
        try:
            review_pr_mod.__file__ = str(tmp_path / "nonexistent" / "maestro_review" / "cli.py")
            with patch("maestro_review.cli.Path.cwd", return_value=tmp_path / "also_nonexistent"):
                result = _load_review_defaults()
        finally:
            review_pr_mod.__file__ = original_file

        assert result == {}

    def test_review_section_with_values(self, tmp_path: Path) -> None:
        """When maestro.toml has a populated [review] section, values are returned."""
        from maestro_review.cli import _load_review_defaults

        toml_file = tmp_path / "cwd" / "maestro.toml"
        toml_file.parent.mkdir()
        toml_file.write_text(
            "[review]\nmax_diff_chars = 100000\nmax_files = 75\ngh_timeout = 45\n"
        )

        original_file = review_pr_mod.__file__
        try:
            review_pr_mod.__file__ = str(tmp_path / "nonexistent" / "maestro_review" / "cli.py")
            with patch("maestro_review.cli.Path.cwd", return_value=tmp_path / "cwd"):
                result = _load_review_defaults()
        finally:
            review_pr_mod.__file__ = original_file

        assert result["max_diff_chars"] == 100000
        assert result["max_files"] == 75
        assert result["gh_timeout"] == 45


class TestAdvisorySeveritiesRemoved:
    """_ADVISORY_SEVERITIES must not exist as dead code in cli.py."""

    def test_advisory_severities_constant_is_removed(self) -> None:
        """Verify _ADVISORY_SEVERITIES has been removed from cli.py (was dead code)."""
        import ast
        import pathlib

        source = pathlib.Path(review_pr_mod.__file__).read_text()
        tree = ast.parse(source)

        usages = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "_ADVISORY_SEVERITIES"
        ]
        definitions = [n for n in usages if isinstance(n.ctx, ast.Store)]

        assert len(definitions) == 0, (
            "_ADVISORY_SEVERITIES is dead code and should have been removed from cli.py"
        )


class TestRequireCommentReplies:
    """--require-comment-replies check-only gate (proposal 083, step 4).

    The flag turns the run into a read-only reply-coverage audit: the diff
    review and posting are skipped entirely (reviewer.review / asyncio.run are
    never called), and the exit code reflects whether any of the reviewer's own
    findings still lack a reply.
    """

    @staticmethod
    def _thread(body: str) -> ReviewThread:
        return ReviewThread(
            thread_id="t1",
            comment_id=1,
            path="a.py",
            body=body,
            all_bodies=body,
            reply_bodies=[],
        )

    @staticmethod
    def _overflow() -> OverflowFinding:
        return OverflowFinding(
            fingerprint="of-abc12345",
            file="b.py",
            line=10,
            severity="bug",
            title="Overflow finding",
            description="d",
        )

    def test_unreplied_findings_exit_one_and_skip_review(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """≥1 unreplied finding → exit 1; review/asyncio.run never called; logged."""
        report = UnrepliedComments(
            inline=[self._thread("🐛 **Bare finding**\nDetails")],
            overflow=[self._overflow()],
        )

        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--require-comment-replies"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
            caplog.at_level(logging.INFO, logger="maestro_review.cli"),
        ):
            reviewer = mock_cls.return_value
            reviewer.find_unreplied_comments.return_value = report

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

        assert exc.value.code == 1
        reviewer.find_unreplied_comments.assert_called_once_with(1)
        reviewer.review.assert_not_called()
        mock_asyncio.run.assert_not_called()
        assert "Unreplied inline comment: a.py — Bare finding" in caplog.text
        assert "Unreplied overflow comment: of-abc12345 b.py:10" in caplog.text
        assert "Failing review: 2 unreplied comment(s)" in caplog.text

    def test_no_unreplied_findings_exit_zero(self) -> None:
        """A genuinely empty report → exit 0; review/asyncio.run never called.

        This is the all-replied case only. A gh/GraphQL outage no longer maps to
        an empty report: ``find_unreplied_comments`` probes connectivity first
        and raises on a failure, so the gate fails closed (exit 1) — covered by
        ``test_check_failure_exits_one`` here and by
        ``test_find_unreplied_comments_fails_closed_on_connectivity_outage`` in
        test_review.py — instead of being mistaken for "all comments replied".
        """
        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--require-comment-replies"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            reviewer = mock_cls.return_value
            reviewer.find_unreplied_comments.return_value = UnrepliedComments()

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

        assert exc.value.code == 0
        reviewer.find_unreplied_comments.assert_called_once_with(1)
        reviewer.review.assert_not_called()
        mock_asyncio.run.assert_not_called()

    def test_check_failure_exits_one(self) -> None:
        """An exception from find_unreplied_comments surfaces as exit 1."""
        with (
            patch.object(sys, "argv", [*BASE_ARGV, "--require-comment-replies"]),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            reviewer = mock_cls.return_value
            reviewer.find_unreplied_comments.side_effect = RuntimeError("gh down")

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

        assert exc.value.code == 1
        mock_asyncio.run.assert_not_called()

    def test_toml_default_honored_without_flag(self) -> None:
        """A truthy [review] require_comment_replies default enables the gate."""
        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch(
                "maestro_review.cli._load_review_defaults",
                return_value={"require_comment_replies": 1},
            ),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            reviewer = mock_cls.return_value
            reviewer.find_unreplied_comments.return_value = UnrepliedComments()

            with pytest.raises(SystemExit) as exc:
                review_pr_mod.main()

        assert exc.value.code == 0
        reviewer.find_unreplied_comments.assert_called_once_with(1)
        mock_asyncio.run.assert_not_called()


class TestTomlDefaultsNotFrozenAtImport:
    """Defaults must be resolved at main() call time, not import time."""

    def test_main_calls_load_review_defaults_at_invocation(self) -> None:
        """Patching _load_review_defaults must affect the next main() call."""
        result = FakeResult(summary="clean", findings=[])
        custom_defaults = {"max_diff_chars": 12345, "max_files": 7, "gh_timeout": 999}

        with (
            patch.object(sys, "argv", BASE_ARGV),
            patch("maestro_review.cli._load_review_defaults", return_value=custom_defaults),
            patch("maestro_review.cli.PRReviewer") as mock_cls,
            patch("maestro_review.cli.asyncio") as mock_asyncio,
        ):
            mock_asyncio.run.return_value = result
            mock_cls.return_value.post_review.return_value = _post_result()
            review_pr_mod.main()

            mock_cls.assert_called_once_with(
                github_repo="owner/repo",
                max_diff_chars=12345,
                max_files=7,
                gh_timeout=999,
            )
