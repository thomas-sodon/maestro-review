"""Tests for create_issue in GitHubClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from maestro_review.github import GitHubClient, GitHubError


class TestCreateIssue:
    """Tests for GitHubClient.create_issue (REST API via ``gh api``)."""

    def test_create_issue_returns_number(self) -> None:
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = '{"number": 42, "html_url": "https://github.com/owner/repo/issues/42"}'

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.create_issue("Bug title", "Bug body")

        assert result == 42
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["gh", "api"]
        assert "repos/owner/repo/issues" in cmd
        assert "--method" in cmd
        assert "POST" in cmd
        # Verify payload is passed via --input -
        assert "--input" in cmd
        assert "-" in cmd
        # Verify input_data contains title and body
        input_data = mock_run.call_args[1]["input_data"]
        import json
        payload = json.loads(input_data)
        assert payload["title"] == "Bug title"
        assert payload["body"] == "Bug body"
        assert "labels" not in payload

    def test_create_issue_with_labels(self) -> None:
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = '{"number": 7}'

        with patch.object(client, "_run", return_value=mock_result) as mock_run:
            result = client.create_issue("title", "body", labels=["bug", "ai"])

        assert result == 7
        import json
        input_data = mock_run.call_args[1]["input_data"]
        payload = json.loads(input_data)
        assert payload["labels"] == ["bug", "ai"]

    def test_create_issue_no_repo_returns_none(self) -> None:
        client = GitHubClient(repo=None)
        result = client.create_issue("title", "body")
        assert result is None

    def test_create_issue_github_error_returns_none(self) -> None:
        client = GitHubClient(repo="owner/repo")
        with patch.object(client, "_run", side_effect=GitHubError("fail")):
            result = client.create_issue("title", "body")
        assert result is None

    def test_create_issue_unparseable_response_returns_none(self) -> None:
        client = GitHubClient(repo="owner/repo")
        mock_result = MagicMock()
        mock_result.stdout = "unexpected output"

        with patch.object(client, "_run", return_value=mock_result):
            result = client.create_issue("title", "body")

        assert result is None
