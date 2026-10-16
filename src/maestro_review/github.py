"""GitHub client for code review operations.

Wraps the ``gh`` CLI for PR review interactions: fetching diffs, posting
inline review comments, managing review threads, and listing issues.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("maestro_review")


class GitHubError(Exception):
    """Raised when a GitHub CLI operation fails."""


class CodeHost(Protocol):
    """Protocol for GitHub review operations — allows test doubles."""

    def get_pr_diff(self, pr_number: int) -> str: ...

    def get_pr_info(self, pr_number: int) -> dict[str, str]: ...

    def post_pr_review(
        self, pr_number: int, payload: dict[str, Any],
    ) -> None: ...

    def get_pr_review_threads(
        self, pr_number: int,
    ) -> list[dict[str, Any]]: ...

    def get_pr_review_bodies(self, pr_number: int) -> list[str]: ...

    def get_pr_comment_bodies(self, pr_number: int) -> list[str]: ...

    def check_connectivity(self, pr_number: int) -> None: ...

    def reply_to_review_comment(
        self, pr_number: int, comment_id: int, body: str,
    ) -> None: ...

    def list_issues(
        self,
        *,
        state: str = "open",
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None: ...

    def create_issue(
        self, title: str, body: str, labels: list[str] | None = None,
    ) -> int | None: ...

    def close_issue(
        self, number: int, *, comment: str | None = None,
    ) -> None: ...

    def comment_on_issue(self, number: int, body: str) -> None: ...

    def get_pr_base_branch(self, pr_number: int) -> str | None: ...


class GitHubClient:
    """GitHub CLI (``gh``) client for code review operations."""

    def __init__(
        self,
        repo: str | None = None,
        *,
        workspace: Path | None = None,
        timeout: int = 30,
    ) -> None:
        self._repo = repo
        self._cwd = str(workspace) if workspace else None
        self._timeout = timeout

    def _repo_args(self) -> list[str]:
        """Return ``['--repo', repo]`` when a repo is configured, else ``[]``."""
        return ["--repo", self._repo] if self._repo else []

    def _run(
        self,
        cmd: list[str],
        *,
        timeout: int | None = None,
        input_data: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a gh command, raising :class:`GitHubError` on failure."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self._timeout if timeout is None else timeout,
                cwd=self._cwd,
                input=input_data,
            )
        except subprocess.CalledProcessError as exc:
            parts = [f"gh command failed (exit {exc.returncode}): {exc.cmd}"]
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            if stderr.strip():
                parts.append(f"stderr: {stderr.strip()}")
            if stdout.strip():
                parts.append(f"stdout: {stdout.strip()}")
            raise GitHubError("\n".join(parts)) from exc
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise GitHubError(str(exc)) from exc

    # -- Must-succeed methods -----------------------------------------------

    def post_pr_review(
        self, pr_number: int, payload: dict[str, Any],
    ) -> None:
        if not self._repo:
            raise GitHubError("Cannot post review: no repo configured")
        self._run(
            [
                "gh", "api",
                f"repos/{self._repo}/pulls/{pr_number}/reviews",
                "--method", "POST",
                "--input", "-",
            ],
            input_data=json.dumps(payload),
        )

    def check_connectivity(self, pr_number: int) -> None:
        """Probe that the PR is reachable, raising on a gh/GraphQL outage.

        The best-effort read methods below swallow every failure and return an
        empty result, so a rate-limited, unauthenticated, or network-failed
        ``gh`` invocation is indistinguishable from "nothing to report".  The
        reply-coverage gate (:meth:`PRReviewer.find_unreplied_comments`) calls
        this first so a genuine outage surfaces as a raised
        :class:`GitHubError` — failing the gate closed — instead of a false
        all-clear.  Succeeds silently when the PR is reachable.

        The audit reads two independent GitHub API surfaces with separate
        rate-limit pools and failure modes, so both are probed: the REST
        endpoint (``gh pr view``, behind the overflow review bodies and the
        ack roll-up) and the GraphQL endpoint (``gh api graphql``, behind the
        inline review threads — the audit's primary data source).  A REST-only
        probe would pass while GraphQL is rate-limited, letting
        :meth:`get_pr_review_threads` swallow that outage and yield a false
        all-clear.  The GraphQL probe is skipped only when no valid repo is
        configured, mirroring :meth:`get_pr_review_threads`, which then makes no
        GraphQL call at all (returns ``[]`` unconditionally).
        """
        self._run([
            "gh", "pr", "view", str(pr_number),
            *self._repo_args(),
            "--json", "number",
        ])
        parts = self._valid_repo_parts()
        if parts is None:
            return
        owner, repo = parts
        self._run([
            "gh", "api", "graphql",
            "-f", (
                "query=query($owner:String!,$repo:String!,$pr:Int!){"
                "repository(owner:$owner,name:$repo){"
                "pullRequest(number:$pr){number}}}"
            ),
            "-f", f"owner={owner}",
            "-f", f"repo={repo}",
            "-F", f"pr={pr_number}",
        ])

    # -- Best-effort methods ------------------------------------------------

    def get_pr_diff(self, pr_number: int) -> str:
        """Fetch the full branch diff for the PR (base branch to HEAD)."""
        try:
            result = self._run([
                "gh", "pr", "diff", str(pr_number),
                *self._repo_args(),
            ])
            return result.stdout
        except GitHubError:
            logger.warning("Failed to fetch PR diff", exc_info=True)
            return ""

    def get_pr_info(self, pr_number: int) -> dict[str, str]:
        try:
            result = self._run([
                "gh", "pr", "view", str(pr_number),
                *self._repo_args(),
                "--json", "title,body",
            ])
            data = json.loads(result.stdout)
            return {
                "title": data.get("title", ""),
                "body": data.get("body", ""),
            }
        except (GitHubError, json.JSONDecodeError):
            logger.warning("Failed to fetch PR info", exc_info=True)
            return {"title": "", "body": ""}

    def get_pr_base_branch(self, pr_number: int) -> str | None:
        """Return the base branch name for a PR, or ``None`` on failure."""
        try:
            result = self._run([
                "gh", "pr", "view", str(pr_number),
                *self._repo_args(),
                "--json", "baseRefName",
            ])
            data = json.loads(result.stdout)
            branch = str(data.get("baseRefName", ""))
            if branch:
                return branch
        except (GitHubError, json.JSONDecodeError):
            logger.warning("Failed to fetch PR base branch", exc_info=True)
        return None

    def _valid_repo_parts(self) -> tuple[str, str] | None:
        """Split repo into (owner, name). Returns None if format is invalid."""
        if not self._repo or self._repo.count("/") != 1:
            return None
        owner, name = self._repo.split("/")
        if not owner or not name:
            return None
        return owner, name

    def get_pr_review_threads(
        self, pr_number: int,
    ) -> list[dict[str, Any]]:
        parts = self._valid_repo_parts()
        if parts is None:
            return []
        owner, repo = parts
        gql_query = (
            "query($owner:String!,$repo:String!,$pr:Int!){"
            "repository(owner:$owner,name:$repo){"
            "pullRequest(number:$pr){"
            "reviewThreads(first:100){nodes{"
            "id isResolved isOutdated "
            "comments(first:100){nodes{databaseId path body}}"
            "}}"
            "}}}"
        )
        raw_stdout = ""
        try:
            result = self._run([
                "gh", "api", "graphql",
                "-f", f"query={gql_query}",
                "-f", f"owner={owner}",
                "-f", f"repo={repo}",
                "-F", f"pr={pr_number}",
            ])
            raw_stdout = result.stdout
            data = json.loads(raw_stdout)
            if data.get("data") is None:
                logger.warning(
                    "GraphQL returned null data for PR #%d%s: %.500s",
                    pr_number,
                    f" (errors: {data['errors']})" if "errors" in data else "",
                    raw_stdout,
                )
                return []
            nodes: list[dict[str, Any]] = (
                data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
            )
            return nodes
        except GitHubError as exc:
            logger.warning(
                "gh CLI failed fetching review threads for PR #%d: %s",
                pr_number, exc, exc_info=True,
            )
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Failed to parse review threads for PR #%d: %s — response: %.500s",
                pr_number, exc, raw_stdout, exc_info=True,
            )
            return []

    def get_pr_review_bodies(self, pr_number: int) -> list[str]:
        """Fetch the bodies of a PR's submitted reviews, oldest first.

        The reply-coverage gate scans these for the reviewer's
        overflow-findings block.  Best-effort — logs a warning and returns
        ``[]`` on failure (consistent with :meth:`get_pr_review_threads`).
        """
        try:
            result = self._run([
                "gh", "pr", "view", str(pr_number),
                *self._repo_args(),
                "--json", "reviews",
            ])
            data = json.loads(result.stdout)
            reviews = data.get("reviews") or []
            return [
                str(review.get("body") or "")
                for review in reviews
                if isinstance(review, dict)
            ]
        except (GitHubError, json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Failed to fetch PR review bodies", exc_info=True)
            return []

    def get_pr_comment_bodies(self, pr_number: int) -> list[str]:
        """Fetch the bodies of a PR's issue comments, oldest first.

        Used to locate the roll-up overflow acknowledgement comment.
        Best-effort — logs a warning and returns ``[]`` on failure (consistent
        with :meth:`get_pr_review_threads`).
        """
        try:
            result = self._run([
                "gh", "pr", "view", str(pr_number),
                *self._repo_args(),
                "--json", "comments",
            ])
            data = json.loads(result.stdout)
            comments = data.get("comments") or []
            return [
                str(comment.get("body") or "")
                for comment in comments
                if isinstance(comment, dict)
            ]
        except (GitHubError, json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Failed to fetch PR comment bodies", exc_info=True)
            return []

    def reply_to_review_comment(
        self, pr_number: int, comment_id: int, body: str,
    ) -> None:
        if not self._repo:
            return
        try:
            self._run(
                [
                    "gh", "api",
                    f"repos/{self._repo}/pulls/{pr_number}/comments",
                    "--method", "POST",
                    "--input", "-",
                ],
                input_data=json.dumps({
                    "body": body,
                    "in_reply_to": comment_id,
                }),
            )
        except GitHubError:
            logger.warning(
                "Failed to reply to review comment %d", comment_id,
                exc_info=True,
            )

    def list_issues(
        self,
        *,
        state: str = "open",
        labels: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        if not self._repo:
            return None

        cmd: list[str] = [
            "gh", "issue", "list",
            "--repo", self._repo,
            "--state", state,
            "--json", "number,title,labels,body,createdAt",
        ]
        if labels:
            for label in labels:
                cmd.extend(["--label", label])
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

        try:
            result = self._run(cmd)
        except GitHubError:
            logger.warning("Failed to list issues", exc_info=True)
            return None

        try:
            raw: list[dict[str, Any]] = json.loads(result.stdout)
            return raw
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from gh issue list")
            return None

    def create_issue(
        self, title: str, body: str, labels: list[str] | None = None,
    ) -> int | None:
        """Create a GitHub issue. Returns the issue number, or ``None`` on failure.

        Uses the REST API (``gh api``) instead of ``gh issue create`` because
        the latter uses GraphQL internally, which fails with the automatic
        ``GITHUB_TOKEN`` in GitHub Actions (``Resource not accessible by
        integration``).  The REST endpoint respects ``issues: write``.

        Best-effort — logs a warning and returns ``None`` on any error.
        """
        if not self._repo:
            return None

        logger.info(
            "Creating issue: title=%r, labels=%s, repo=%s",
            title, labels or [], self._repo,
        )

        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels

        try:
            result = self._run(
                [
                    "gh", "api",
                    f"repos/{self._repo}/issues",
                    "--method", "POST",
                    "--input", "-",
                ],
                input_data=json.dumps(payload),
            )
        except GitHubError:
            logger.warning(
                "Failed to create issue (title=%r, labels=%s)",
                title, labels or [], exc_info=True,
            )
            return None

        try:
            data = json.loads(result.stdout)
            issue_number = int(data["number"])
            logger.info("Created issue #%d", issue_number)
            return issue_number
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Could not parse issue number from response: %.500s",
                result.stdout,
            )
            return None

    def close_issue(
        self, number: int, *, comment: str | None = None,
    ) -> None:
        """Close a GitHub issue. Best-effort — logs a warning on failure."""
        if not self._repo:
            return

        cmd: list[str] = [
            "gh", "issue", "close", str(number),
            "--repo", self._repo,
        ]
        if comment:
            cmd.extend(["--comment", comment])

        try:
            self._run(cmd)
        except GitHubError:
            logger.warning("Failed to close issue #%d", number, exc_info=True)

    def comment_on_issue(self, number: int, body: str) -> None:
        """Add a comment to a GitHub issue. Best-effort — logs a warning on failure."""
        if not self._repo:
            return

        try:
            self._run([
                "gh", "issue", "comment", str(number),
                "--repo", self._repo,
                "--body", body,
            ])
        except GitHubError:
            logger.warning(
                "Failed to comment on issue #%d", number, exc_info=True,
            )
