"""Tests for maestro_review.github — GitHubClient with subprocess mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from maestro_review.github import GitHubClient, GitHubError


class TestListIssuesUsesRun:
    """Verify list_issues delegates to _run instead of calling subprocess directly."""

    def test_list_issues_delegates_to_run(self) -> None:
        """list_issues must call self._run, not subprocess.run directly."""
        client = GitHubClient(repo="owner/repo")
        issues_json = json.dumps([{"number": 1, "title": "Bug"}])
        mock_result = MagicMock()
        mock_result.stdout = issues_json

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.list_issues(state="open")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "gh" in cmd
        assert "issue" in cmd
        assert "list" in cmd
        assert result == [{"number": 1, "title": "Bug"}]

    def test_list_issues_catches_github_error(self) -> None:
        """list_issues returns None when _run raises GitHubError."""
        client = GitHubClient(repo="owner/repo")

        with patch.object(client, "_run", side_effect=GitHubError("timeout")):
            result = client.list_issues()

        assert result is None

    def test_list_issues_no_repo_returns_none(self) -> None:
        """list_issues returns None when no repo is configured."""
        client = GitHubClient(repo=None)
        result = client.list_issues()
        assert result is None

    def test_list_issues_invalid_json_returns_none(self) -> None:
        """list_issues returns None when _run returns invalid JSON."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "not-json"

        with patch.object(client, "_run", return_value=mock_result):
            result = client.list_issues()

        assert result is None

    def test_list_issues_passes_labels(self) -> None:
        """list_issues includes --label flags for each label."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.list_issues(labels=["bug", "critical"])

        cmd = mock_run.call_args[0][0]
        assert "--label" in cmd
        label_indices = [i for i, arg in enumerate(cmd) if arg == "--label"]
        assert len(label_indices) == 2
        assert cmd[label_indices[0] + 1] == "bug"
        assert cmd[label_indices[1] + 1] == "critical"

    def test_list_issues_does_not_call_subprocess_directly(self) -> None:
        """Ensure subprocess.run is NOT called directly by list_issues.

        This is the key regression test: list_issues must route through _run.
        """
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with (
            patch.object(client, "_run", return_value=mock_result),
            patch("maestro_review.github.subprocess.run") as mock_subprocess,
        ):
            client.list_issues()

        mock_subprocess.assert_not_called()


class TestGetPrDiff:
    """Tests for GitHubClient.get_pr_diff."""

    def test_get_pr_diff_success(self) -> None:
        """get_pr_diff returns the full branch diff via gh pr diff."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "diff --git a/file.py b/file.py\n+new line\n"

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_diff(42)

        assert result == "diff --git a/file.py b/file.py\n+new line\n"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "diff", "42", "--repo", "owner/repo"]

    def test_get_pr_diff_github_error_returns_empty(self) -> None:
        """get_pr_diff returns '' when _run raises GitHubError."""
        client = GitHubClient(repo="owner/repo")

        with patch.object(client, "_run", side_effect=GitHubError("timeout")):
            result = client.get_pr_diff(42)

        assert result == ""

    def test_get_pr_diff_no_repo_omits_flag(self) -> None:
        """get_pr_diff omits --repo when no repo is configured."""
        client = GitHubClient(repo=None)
        mock_result = MagicMock()
        mock_result.stdout = "some diff"

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_diff(1)

        assert result == "some diff"
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "diff", "1"]
        assert "--repo" not in cmd


class TestGetPrReviewThreads:
    """Tests for GitHubClient.get_pr_review_threads."""

    def test_uses_cli_flags_not_stdin(self) -> None:
        """get_pr_review_threads must use -f/-F flags, not --input -."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}},
        })

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.get_pr_review_threads(42)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0:3] == ["gh", "api", "graphql"]
        assert "--input" not in cmd
        # Verify -f and -F flags are used
        f_indices = [i for i, v in enumerate(cmd) if v == "-f"]
        assert len(f_indices) == 3  # query, owner, repo
        big_f_indices = [i for i, v in enumerate(cmd) if v == "-F"]
        assert len(big_f_indices) == 1  # pr (integer)
        assert cmd[big_f_indices[0] + 1] == "pr=42"
        # input_data should not be passed
        assert "input_data" not in (mock_run.call_args[1] or {})

    def test_graphql_query_includes_is_outdated(self) -> None:
        """GraphQL query must request isOutdated so the reviewer can skip
        threads where the commented code has changed."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}},
        })

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.get_pr_review_threads(1)

        cmd = mock_run.call_args[0][0]
        # The query string is passed as "-f" "query=..."
        query_arg = next(
            cmd[i + 1] for i, v in enumerate(cmd)
            if v == "-f" and cmd[i + 1].startswith("query=")
        )
        assert "isOutdated" in query_arg

    def test_success_returns_nodes(self) -> None:
        """get_pr_review_threads returns thread nodes on success."""
        client = GitHubClient(repo="owner/repo")
        nodes = [{"id": "T1", "isResolved": False, "comments": {"nodes": []}}]
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": nodes}}}},
        })

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_threads(1)

        assert result == nodes

    def test_no_repo_returns_empty(self) -> None:
        """get_pr_review_threads returns [] when no repo is configured."""
        client = GitHubClient(repo=None)
        assert client.get_pr_review_threads(1) == []

    def test_github_error_returns_empty(self) -> None:
        """get_pr_review_threads returns [] when gh CLI fails."""
        client = GitHubClient(repo="owner/repo")

        with patch.object(client, "_run", side_effect=GitHubError("fail")):
            result = client.get_pr_review_threads(1)

        assert result == []

    def test_graphql_errors_returns_empty(self) -> None:
        """get_pr_review_threads returns [] when GraphQL returns null data with errors."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "data": None,
            "errors": [{"message": "Something went wrong"}],
        })

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_threads(1)

        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        """get_pr_review_threads returns [] on malformed JSON response."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "not-json"

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_threads(1)

        assert result == []


class TestGetPrReviewBodies:
    """Tests for GitHubClient.get_pr_review_bodies."""

    def test_success_returns_bodies_in_order(self) -> None:
        """get_pr_review_bodies returns review body strings in order."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "reviews": [{"body": "first"}, {"body": "second"}],
        })

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_review_bodies(42)

        assert result == ["first", "second"]
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "gh", "pr", "view", "42", "--repo", "owner/repo",
            "--json", "reviews",
        ]

    def test_missing_or_null_reviews_returns_empty(self) -> None:
        """A null/absent reviews key yields an empty list, not a crash."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"reviews": None})

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_bodies(42)

        assert result == []

    def test_non_dict_entries_skipped_and_missing_body_empty(self) -> None:
        """Non-dict entries are filtered; a dict missing body yields ''."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "reviews": ["not-a-dict", {"body": None}, {"body": "kept"}],
        })

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_bodies(42)

        assert result == ["", "kept"]

    def test_github_error_returns_empty(self) -> None:
        """get_pr_review_bodies returns [] when _run raises GitHubError."""
        client = GitHubClient(repo="owner/repo")

        with patch.object(client, "_run", side_effect=GitHubError("timeout")):
            result = client.get_pr_review_bodies(42)

        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        """get_pr_review_bodies returns [] on malformed JSON."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "not-json"

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_review_bodies(42)

        assert result == []

    def test_no_repo_omits_flag(self) -> None:
        """get_pr_review_bodies omits --repo when no repo is configured."""
        client = GitHubClient(repo=None)
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"reviews": []})

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_review_bodies(1)

        assert result == []
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "view", "1", "--json", "reviews"]
        assert "--repo" not in cmd


class TestGetPrCommentBodies:
    """Tests for GitHubClient.get_pr_comment_bodies."""

    def test_success_returns_bodies_in_order(self) -> None:
        """get_pr_comment_bodies returns comment body strings in order."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "comments": [{"body": "alpha"}, {"body": "beta"}],
        })

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_comment_bodies(42)

        assert result == ["alpha", "beta"]
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "gh", "pr", "view", "42", "--repo", "owner/repo",
            "--json", "comments",
        ]

    def test_missing_or_null_comments_returns_empty(self) -> None:
        """A null/absent comments key yields an empty list, not a crash."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"comments": None})

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_comment_bodies(42)

        assert result == []

    def test_non_dict_entries_skipped_and_missing_body_empty(self) -> None:
        """Non-dict entries are filtered; a dict missing body yields ''."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "comments": [42, {"body": None}, {"body": "kept"}],
        })

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_comment_bodies(42)

        assert result == ["", "kept"]

    def test_github_error_returns_empty(self) -> None:
        """get_pr_comment_bodies returns [] when _run raises GitHubError."""
        client = GitHubClient(repo="owner/repo")

        with patch.object(client, "_run", side_effect=GitHubError("timeout")):
            result = client.get_pr_comment_bodies(42)

        assert result == []

    def test_invalid_json_returns_empty(self) -> None:
        """get_pr_comment_bodies returns [] on malformed JSON."""
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "not-json"

        with patch.object(client, "_run", return_value=mock_result):
            result = client.get_pr_comment_bodies(42)

        assert result == []

    def test_no_repo_omits_flag(self) -> None:
        """get_pr_comment_bodies omits --repo when no repo is configured."""
        client = GitHubClient(repo=None)
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"comments": []})

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.get_pr_comment_bodies(1)

        assert result == []
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "view", "1", "--json", "comments"]
        assert "--repo" not in cmd


class TestCheckConnectivity:
    """Tests for GitHubClient.check_connectivity (reply-coverage fail-closed probe)."""

    def test_success_is_silent_and_probes_both_surfaces(self) -> None:
        """A reachable PR returns None and probes BOTH the REST and GraphQL APIs.

        The inline-thread data source is GraphQL while the overflow bodies are
        REST; a REST-only probe would miss a GraphQL-only outage, so both
        endpoints are exercised (see check_connectivity docstring).
        """
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"number": 42})

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.check_connectivity(42)

        assert result is None
        assert mock_run.call_count == 2
        rest_cmd = mock_run.call_args_list[0][0][0]
        assert rest_cmd == [
            "gh", "pr", "view", "42", "--repo", "owner/repo",
            "--json", "number",
        ]
        gql_cmd = mock_run.call_args_list[1][0][0]
        assert gql_cmd[:3] == ["gh", "api", "graphql"]
        assert "owner=owner" in gql_cmd
        assert "repo=repo" in gql_cmd
        assert "pr=42" in gql_cmd

    def test_rest_outage_propagates_github_error(self) -> None:
        """A REST-domain outage on the first probe is NOT swallowed — it raises."""
        client = GitHubClient(repo="owner/repo")

        with (
            patch.object(client, "_run", side_effect=GitHubError("rate limited")),
            pytest.raises(GitHubError),
        ):
            client.check_connectivity(42)

    def test_graphql_outage_propagates_github_error(self) -> None:
        """REST up but GraphQL down (separate failure domain) still fails closed.

        Regression guard for the fail-closed hole: without the GraphQL probe a
        GraphQL-only outage passed the REST probe, get_pr_review_threads
        swallowed it to ``[]``, and the gate exited 0 (false all-clear).
        """
        client = GitHubClient(repo="owner/repo")
        rest_ok = MagicMock()
        rest_ok.stdout = json.dumps({"number": 42})

        with (
            patch.object(
                client, "_run",
                side_effect=[rest_ok, GitHubError("graphql rate limited")],
            ) as mock_run,
            pytest.raises(GitHubError),
        ):
            client.check_connectivity(42)

        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1][0][0][:3] == ["gh", "api", "graphql"]

    def test_no_repo_omits_flag_and_skips_graphql(self) -> None:
        """No repo: only the REST probe runs; the GraphQL probe is skipped.

        get_pr_review_threads makes no GraphQL call without a valid repo
        (returns ``[]`` unconditionally), so there is nothing to fail closed on.
        """
        client = GitHubClient(repo=None)
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"number": 1})

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.check_connectivity(1)

        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "view", "1", "--json", "number"]
        assert "--repo" not in cmd


class TestListIssuesLimit:
    """Verify list_issues passes --limit when provided."""

    def test_list_issues_passes_limit(self) -> None:
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.list_issues(limit=500)

        cmd = mock_run.call_args[0][0]
        limit_idx = cmd.index("--limit")
        assert cmd[limit_idx + 1] == "500"

    def test_list_issues_omits_limit_when_none(self) -> None:
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "[]"

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            client.list_issues()

        cmd = mock_run.call_args[0][0]
        assert "--limit" not in cmd


class TestCloseIssue:
    """Tests for GitHubClient.close_issue."""

    def test_close_issue_delegates_to_run(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as mock_run:
            client.close_issue(42)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["gh", "issue", "close", "42"]
        assert "--repo" in cmd

    def test_close_issue_with_comment(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as mock_run:
            client.close_issue(42, comment="Duplicate of #1")

        cmd = mock_run.call_args[0][0]
        comment_idx = cmd.index("--comment")
        assert cmd[comment_idx + 1] == "Duplicate of #1"

    def test_close_issue_without_comment_omits_flag(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as mock_run:
            client.close_issue(42)

        cmd = mock_run.call_args[0][0]
        assert "--comment" not in cmd

    def test_close_issue_no_repo_returns_early(self) -> None:
        client = GitHubClient(repo=None)
        with patch.object(client, "_run") as mock_run:
            client.close_issue(42)
        mock_run.assert_not_called()

    def test_close_issue_catches_github_error(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("fail")):
            client.close_issue(42)  # should not raise


class TestCommentOnIssue:
    """Tests for GitHubClient.comment_on_issue."""

    def test_comment_on_issue_delegates_to_run(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run") as mock_run:
            client.comment_on_issue(42, "Hello")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["gh", "issue", "comment", "42"]
        body_idx = cmd.index("--body")
        assert cmd[body_idx + 1] == "Hello"

    def test_comment_on_issue_no_repo_returns_early(self) -> None:
        client = GitHubClient(repo=None)
        with patch.object(client, "_run") as mock_run:
            client.comment_on_issue(42, "Hello")
        mock_run.assert_not_called()

    def test_comment_on_issue_catches_github_error(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("fail")):
            client.comment_on_issue(42, "Hello")  # should not raise


class TestRepoWithMultipleSlashes:
    """Repos with multiple slashes (e.g. 'org/repo/extra') pass the
    '/' check but split(/, 1) gives wrong repo_name for GraphQL."""

    def test_review_threads_multi_slash_returns_empty(self) -> None:
        client = GitHubClient(repo="org/repo/extra")
        result = client.get_pr_review_threads(1)
        assert result == []

