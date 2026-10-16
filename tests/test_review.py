"""Tests for PRReviewer — CodeHost and SDK calls are mocked."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maestro_review.github import GitHubError
from maestro_review.markers import _FIXED_LOCALLY_MARKER
from maestro_review.review import (
    _AI_FINDING_RE,
    _OVERFLOW_ACK_SENTINEL,
    _OVERFLOW_LEGACY_HEADER,
    _OVERFLOW_SENTINEL_CLOSE,
    _OVERFLOW_SENTINEL_OPEN,
    OverflowFinding,
    PostReviewResult,
    PRReviewer,
    ReviewFinding,
    ReviewThread,
    UnrepliedComments,
    _extract_thread_title,
    _fuzzy_title_match,
    _legacy_overflow_fingerprint,
    _normalize_title,
    _overflow_fingerprint,
    _parse_overflow_ack,
    _parse_overflow_findings,
)
from maestro_review.sdk import QueryResult, QueryUsage

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/maestro/review.py b/maestro/review.py
--- a/maestro/review.py
+++ b/maestro/review.py
@@ -10,6 +10,8 @@ import logging
 import json
 import re
 import subprocess
+import os
+import sys
 from dataclasses import dataclass
"""

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

MULTI_FILE_DIFF_WITH_METADATA = """\
diff --git a/a.py b/a.py
index abc1234..def5678 100644
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 line1
+added
 line3
 line4
diff --git a/b.py b/b.py
similarity index 90%
rename from old_b.py
rename to b.py
index 1111111..2222222 100644
new mode 100755
--- a/b.py
+++ b/b.py
@@ -5,3 +5,4 @@
 existing
+new_line
 more
"""

LOCKFILE_DIFF = """\
diff --git a/package-lock.json b/package-lock.json
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,4 @@
 {
+  "new": true
 }
diff --git a/maestro/real.py b/maestro/real.py
--- a/maestro/real.py
+++ b/maestro/real.py
@@ -1,3 +1,4 @@
 code
+new_code
 more
"""


def _reviewer(host: MagicMock | None = None) -> PRReviewer:
    if host is None:
        host = MagicMock()
        # Defaults so filtering is a no-op unless a test configures them.
        host.list_issues.return_value = []
        host.get_pr_review_threads.return_value = []
    return PRReviewer("owner/repo", code_host=host)


def _sdk_result(response_data: dict[str, object]) -> QueryResult:
    """Create a QueryResult with JSON response."""
    return QueryResult(
        response=json.dumps(response_data),
        usage=QueryUsage(total_tokens=1000, total_cost_usd=0.05),
    )


# ------------------------------------------------------------------
# get_pr_diff
# ------------------------------------------------------------------


def test_get_pr_diff_success() -> None:
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    reviewer = _reviewer(host)

    diff = reviewer.get_pr_diff(42)

    assert "review.py" in diff
    host.get_pr_diff.assert_called_once_with(42)


def test_get_pr_diff_failure() -> None:
    host = MagicMock()
    host.get_pr_diff.return_value = ""
    reviewer = _reviewer(host)

    diff = reviewer.get_pr_diff(42)

    assert diff == ""


# ------------------------------------------------------------------
# get_pr_info
# ------------------------------------------------------------------


def test_get_pr_info_success() -> None:
    host = MagicMock()
    host.get_pr_info.return_value = {"title": "Fix bug", "body": "Details"}
    reviewer = _reviewer(host)

    info = reviewer.get_pr_info(10)

    assert info["title"] == "Fix bug"
    assert info["body"] == "Details"


def test_get_pr_info_failure() -> None:
    host = MagicMock()
    host.get_pr_info.return_value = {"title": "", "body": ""}
    reviewer = _reviewer(host)

    info = reviewer.get_pr_info(10)

    assert info == {"title": "", "body": ""}


# ------------------------------------------------------------------
# _truncate_diff
# ------------------------------------------------------------------


def test_truncate_diff_small() -> None:
    """Diffs under the limit are returned unchanged."""
    reviewer = PRReviewer("owner/repo", max_diff_chars=100_000)
    result = reviewer._truncate_diff(SAMPLE_DIFF)
    assert result == SAMPLE_DIFF


def test_truncate_diff_empty() -> None:
    reviewer = PRReviewer("owner/repo")
    assert reviewer._truncate_diff("") == ""


def test_truncate_diff_filters_lockfiles() -> None:
    """Lock files are removed from the diff."""
    reviewer = PRReviewer("owner/repo", max_diff_chars=100_000)
    result = reviewer._truncate_diff(LOCKFILE_DIFF)
    assert "package-lock.json" not in result
    assert "maestro/real.py" in result


def test_truncate_diff_large() -> None:
    """Diffs over the limit are truncated at file boundaries."""
    reviewer = PRReviewer("owner/repo", max_diff_chars=50)
    result = reviewer._truncate_diff(MULTI_FILE_DIFF)
    assert "omitted" in result


def test_truncate_diff_max_files() -> None:
    """Files beyond max_files are dropped."""
    reviewer = PRReviewer("owner/repo", max_files=1, max_diff_chars=100_000)
    result = reviewer._truncate_diff(MULTI_FILE_DIFF)
    assert "omitted" in result


def test_truncate_diff_zero_budget() -> None:
    """A non-positive max_diff_chars produces only an omission marker."""
    reviewer = PRReviewer("owner/repo", max_diff_chars=0)
    result = reviewer._truncate_diff(MULTI_FILE_DIFF)
    assert "2 file(s) omitted" in result


# ------------------------------------------------------------------
# _parse_diff_lines
# ------------------------------------------------------------------


def test_parse_diff_lines_added() -> None:
    """Added lines are identified as commentable."""
    lines = PRReviewer._parse_diff_lines(SAMPLE_DIFF)
    assert "maestro/review.py" in lines
    # Lines 13 and 14 are the added lines (+import os, +import sys)
    assert 13 in lines["maestro/review.py"]
    assert 14 in lines["maestro/review.py"]


def test_parse_diff_lines_context() -> None:
    """Context lines (unchanged) are also commentable."""
    lines = PRReviewer._parse_diff_lines(SAMPLE_DIFF)
    # Context lines around the additions should also be present
    assert 10 in lines["maestro/review.py"]


def test_parse_diff_lines_multi_file() -> None:
    """Multiple files are parsed correctly with exact line sets."""
    lines = PRReviewer._parse_diff_lines(MULTI_FILE_DIFF)
    assert lines["a.py"] == {1, 2, 3, 4}
    assert lines["b.py"] == {5, 6, 7}


def test_parse_diff_lines_inter_file_metadata_not_counted() -> None:
    """Inter-file diff metadata (index, similarity, rename, mode) must not inflate line numbers."""
    lines = PRReviewer._parse_diff_lines(MULTI_FILE_DIFF_WITH_METADATA)
    # a.py has 4 lines in hunk: context(1), +added(2), context(3), context(4)
    assert lines["a.py"] == {1, 2, 3, 4}
    # b.py has 3 lines in hunk: context(5), +new_line(6), context(7)
    assert lines["b.py"] == {5, 6, 7}


def test_parse_diff_lines_empty() -> None:
    """Empty diff returns empty dict."""
    assert PRReviewer._parse_diff_lines("") == {}


def test_parse_diff_lines_deleted_file_not_attributed_to_previous() -> None:
    """Deleted files (+++ /dev/null) must not pollute the previous file's line set.

    When a file is deleted, the diff shows '+++ /dev/null' which doesn't
    start with '+++ b/'.  The old parser left current_file pointing at the
    previous file, causing the deletion's hunk header to increment line
    numbers for the wrong file.
    """
    diff = (
        "diff --git a/keep.py b/keep.py\n"
        "--- a/keep.py\n"
        "+++ b/keep.py\n"
        "@@ -1,3 +1,4 @@\n"
        " existing line\n"
        "+new line\n"
        " another line\n"
        " end\n"
        "diff --git a/removed.py b/removed.py\n"
        "--- a/removed.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-old content\n"
        "-more old\n"
    )
    lines = PRReviewer._parse_diff_lines(diff)
    # keep.py should have its own lines only
    assert "keep.py" in lines
    assert lines["keep.py"] == {1, 2, 3, 4}
    # The deleted file's hunk must NOT add phantom lines to keep.py
    # (old bug: deleted file hunk was attributed to keep.py, adding line 5)
    assert 5 not in lines.get("keep.py", set()), (
        "Deleted file's hunk lines leaked into keep.py"
    )


def test_parse_diff_lines_new_file_mode_not_counted() -> None:
    """'new file mode' metadata must not inflate line numbers of the previous file.

    Git emits 'new file mode 100644' for newly created files.  This is
    distinct from 'new mode 100755' (chmod) and must be skipped by the
    metadata filter so it doesn't add a phantom line to the preceding
    file's commentable set.
    """
    diff = (
        "diff --git a/existing.py b/existing.py\n"
        "--- a/existing.py\n"
        "+++ b/existing.py\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        "+added\n"
        " line3\n"
        " line4\n"
        "diff --git a/brand_new.py b/brand_new.py\n"
        "new file mode 100644\n"
        "index 0000000..abc1234\n"
        "--- /dev/null\n"
        "+++ b/brand_new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+first\n"
        "+second\n"
    )
    lines = PRReviewer._parse_diff_lines(diff)
    # existing.py should have exactly its own hunk lines
    assert lines["existing.py"] == {1, 2, 3, 4}, (
        f"expected {{1, 2, 3, 4}} but got {lines['existing.py']}"
    )
    # brand_new.py should have lines 1 and 2
    assert lines["brand_new.py"] == {1, 2}


# ------------------------------------------------------------------
# review (async)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_success() -> None:
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "Fix", "body": ""}
    reviewer = _reviewer(host)

    response_data = {
        "summary": "Looks good overall",
        "findings": [
            {
                "file": "maestro/review.py",
                "line": 13,
                "severity": "suggestion",
                "title": "Unused import",
                "description": "os is imported but never used",
                "fix_plan": "Remove the import",
                "suggested_code": None,
            },
        ],
    }

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result(response_data)
        result = await reviewer.review(42)

    assert len(result.findings) == 1
    assert result.findings[0].title == "Unused import"
    assert result.summary == "Looks good overall"
    assert result.tokens_used == 1000
    assert result.diff == SAMPLE_DIFF


@pytest.mark.asyncio
async def test_review_empty_diff() -> None:
    host = MagicMock()
    host.get_pr_diff.return_value = ""
    reviewer = _reviewer(host)

    result = await reviewer.review(42)

    assert result.summary == "No changes to review."
    assert result.findings == []


@pytest.mark.asyncio
async def test_review_json_parse_fallback() -> None:
    """When SDK returns non-JSON, fall back to raw response as summary."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    raw_result = QueryResult(
        response="This is not JSON, just text feedback",
        usage=QueryUsage(total_tokens=500, total_cost_usd=0.02),
    )

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = raw_result
        result = await reviewer.review(42)

    assert "not JSON" in result.summary
    assert result.findings == []
    assert result.diff == SAMPLE_DIFF


@pytest.mark.asyncio
async def test_review_malformed_line_values() -> None:
    """Findings with null or non-numeric line values should default to 0, not crash."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    response_data = {
        "summary": "Issues found",
        "findings": [
            {"file": "a.py", "line": None, "severity": "bug",
             "title": "Null line", "description": "d", "fix_plan": "f"},
            {"file": "b.py", "line": "N/A", "severity": "suggestion",
             "title": "String line", "description": "d", "fix_plan": "f"},
            {"file": "c.py", "severity": "nit",
             "title": "Missing line", "description": "d", "fix_plan": "f"},
        ],
    }

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result(response_data)
        result = await reviewer.review(42)

    assert len(result.findings) == 3
    assert result.findings[0].line == 0  # None → 0
    assert result.findings[1].line == 0  # "N/A" → 0
    assert result.findings[2].line == 0  # missing → 0


@pytest.mark.asyncio
async def test_review_null_string_fields_coerced_to_empty() -> None:
    """When the AI returns null for string fields, they must be coerced to "".

    dict.get("key", "") returns None when the key exists with a null value,
    not the default "".  This caused TypeError in _format_finding_body because
    str.join() cannot handle None elements.
    """
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    response_data = {
        "summary": "Issues found",
        "findings": [
            {
                "file": "a.py",
                "line": 10,
                "severity": None,
                "title": None,
                "description": None,
                "fix_plan": None,
                "suggested_code": None,
            },
        ],
    }

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result(response_data)
        result = await reviewer.review(42)

    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.file == "a.py"
    assert f.severity == "suggestion"
    assert f.title == ""
    assert f.description == ""
    assert f.fix_plan == ""
    assert f.suggested_code is None


@pytest.mark.asyncio
async def test_review_drops_finding_without_a_file_path() -> None:
    """A finding carrying no file can never be placed inline, so it would route
    to overflow — where its identity is not re-derivable and the fingerprint
    raises, which would lose every finding in the review.  Drop just that one."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    response_data = {
        "summary": "Issues found",
        "findings": [
            {"file": None, "line": 10, "severity": "bug",
             "title": "Nowhere", "description": "d", "fix_plan": "f"},
            {"file": "a.py", "line": 10, "severity": "bug",
             "title": "Somewhere", "description": "d", "fix_plan": "f"},
        ],
    }

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result(response_data)
        result = await reviewer.review(42)

    assert [f.title for f in result.findings] == ["Somewhere"]


@pytest.mark.asyncio
async def test_review_data_is_bare_list() -> None:
    """When the model returns a bare list (not a dict), it is wrapped."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    bare_list_response = QueryResult(
        response=json.dumps([
            {"file": "a.py", "line": 1, "severity": "bug", "title": "X",
             "description": "d", "fix_plan": "f"},
        ]),
        usage=QueryUsage(total_tokens=10, total_cost_usd=0.0),
    )
    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = bare_list_response
        result = await reviewer.review(42)

    assert len(result.findings) == 1
    assert result.findings[0].file == "a.py"


@pytest.mark.asyncio
async def test_review_skips_non_dict_finding_items_and_normalises_values() -> None:
    """Non-dict items in findings are skipped; negative lines and bad severities normalised."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    response_data = {
        "summary": "mixed",
        "findings": [
            "not a dict",  # skipped
            42,  # skipped
            {"file": "a.py", "line": -5, "severity": "BOGUS",
             "title": "neg", "description": "d", "fix_plan": "f"},
        ],
    }

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result(response_data)
        result = await reviewer.review(42)

    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.line == 0  # negative clamped to 0
    assert f.severity == "suggestion"  # invalid severity normalised


@pytest.mark.asyncio
async def test_review_uses_read_only_tools() -> None:
    """SDK is called with only read-only tools."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result({"summary": "ok", "findings": []})
        await reviewer.review(42)

    options = mock_query.call_args[1]["options"]
    assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}
    # No write tools
    for tool in ("Write", "Edit", "Bash"):
        assert tool not in options.allowed_tools


@pytest.mark.asyncio
async def test_review_uses_opus_5_5_at_high_effort() -> None:
    """The review call runs on claude-opus-5-5 at effort high."""
    host = MagicMock()
    host.get_pr_diff.return_value = SAMPLE_DIFF
    host.get_pr_info.return_value = {"title": "T", "body": ""}
    reviewer = _reviewer(host)

    with patch("maestro_review.review.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = _sdk_result({"summary": "ok", "findings": []})
        await reviewer.review(42)

    options = mock_query.call_args[1]["options"]
    assert options.model == "claude-opus-5-5"
    assert options.effort == "high"


# ------------------------------------------------------------------
# _format_finding_body
# ------------------------------------------------------------------


def test_format_finding_with_suggestion() -> None:
    finding = ReviewFinding(
        file="a.py", line=10, severity="bug", title="Null check",
        description="x may be None", fix_plan="Add guard",
        suggested_code="if x is None:\n    return",
    )
    body = PRReviewer._format_finding_body(finding)
    assert "```suggestion" in body
    assert "if x is None" in body
    assert "\U0001f41b" in body  # bug icon


def test_format_finding_with_fix_plan() -> None:
    finding = ReviewFinding(
        file="a.py", line=10, severity="suggestion", title="Race condition",
        description="Concurrent writes", fix_plan="Use file_lock()",
    )
    body = PRReviewer._format_finding_body(finding)
    assert "**Fix:**" in body
    assert "file_lock()" in body
    assert "suggestion" not in body.lower() or "```suggestion" not in body


def test_format_finding_nit_icon() -> None:
    finding = ReviewFinding(
        file="a.py", line=10, severity="nit", title="Naming",
        description="Consider renaming", fix_plan="Rename x to count",
    )
    body = PRReviewer._format_finding_body(finding)
    assert "\u270f\ufe0f" in body  # pencil icon


def test_format_finding_none_description_does_not_crash() -> None:
    """When the AI returns "description": null, _format_finding_body must not crash.

    dict.get("description", "") returns None (not "") when the key exists
    with a null value.  _format_finding_body joins parts with str.join(),
    which raises TypeError if any element is None.
    """
    finding = ReviewFinding(
        file="a.py", line=10, severity="bug", title="Null desc",
        description=None,  # type: ignore[arg-type]  # simulates null JSON value
        fix_plan="Fix it",
    )
    body = PRReviewer._format_finding_body(finding)
    assert "**Null desc**" in body
    assert "**Fix:** Fix it" in body


# ------------------------------------------------------------------
# _build_review_payload
# ------------------------------------------------------------------


def test_build_review_payload_inline() -> None:
    """Findings on valid lines become inline comments."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(
            file="a.py", line=2, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix it",
        ),
    ]
    valid_lines = {"a.py": {1, 2, 3}}
    payload = reviewer._build_review_payload(findings, valid_lines, "Summary")

    assert payload["event"] == "COMMENT"
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["path"] == "a.py"
    assert payload["comments"][0]["line"] == 2


def test_build_review_payload_invalid_line_overflow() -> None:
    """Findings on invalid lines go to the review body."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(
            file="a.py", line=99, severity="bug", title="Missing",
            description="Not in diff", fix_plan="Fix",
        ),
    ]
    valid_lines = {"a.py": {1, 2, 3}}
    payload = reviewer._build_review_payload(findings, valid_lines, "Summary")

    assert len(payload["comments"]) == 0
    assert "Additional findings" in payload["body"]
    assert "a.py:99" in payload["body"]


def test_build_review_payload_mixed() -> None:
    """Mix of valid and invalid findings."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(
            file="a.py", line=2, severity="bug", title="Valid",
            description="In diff", fix_plan="Fix",
        ),
        ReviewFinding(
            file="a.py", line=99, severity="nit", title="Invalid",
            description="Not in diff", fix_plan="Fix",
        ),
    ]
    valid_lines = {"a.py": {1, 2, 3}}
    payload = reviewer._build_review_payload(findings, valid_lines, "Summary")

    assert len(payload["comments"]) == 1
    assert "Additional findings" in payload["body"]


def test_build_review_payload_disclaimer() -> None:
    """Review body always contains the advisory disclaimer."""
    reviewer = _reviewer()
    payload = reviewer._build_review_payload([], {}, "All good")
    assert "advisory only" in payload["body"].lower()


def test_build_review_payload_overflow_sentinel_wrapping() -> None:
    """Overflow findings are wrapped in the stable sentinel pair, each line
    prefixed with a deterministic ``of-<hash>`` handle that is stable across two
    renders of the same finding."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(
            file="a.py", line=0, severity="bug", title="Null line bug",
            description="Off-hunk", fix_plan="Fix",
        ),
        ReviewFinding(
            file="binary.bin", line=5, severity="bug", title="Not in diff",
            description="File absent from diff", fix_plan="Fix",
        ),
    ]
    valid_lines: dict[str, set[int]] = {"a.py": {1, 2, 3}}

    payload = reviewer._build_review_payload(findings, valid_lines, "Summary")
    body = payload["body"]

    # Sentinel pair wraps the overflow block, in order, exactly once.
    assert body.count(_OVERFLOW_SENTINEL_OPEN) == 1
    assert body.count(_OVERFLOW_SENTINEL_CLOSE) == 1
    open_idx = body.index(_OVERFLOW_SENTINEL_OPEN)
    close_idx = body.index(_OVERFLOW_SENTINEL_CLOSE)
    assert open_idx < close_idx

    # Every line inside the block carries an ``of-<hash>`` handle right after the
    # markdown bullet, and the human-readable text is preserved.
    block = body[open_idx + len(_OVERFLOW_SENTINEL_OPEN):close_idx]
    lines = [line for line in block.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("- of-")
    assert "a.py:0" in block
    assert "binary.bin:5" in block

    # Handles are deterministic across a second render of the same findings.
    payload_again = reviewer._build_review_payload(findings, valid_lines, "Summary")
    assert payload_again["body"] == body


def test_overflow_fingerprint_survives_title_rewording() -> None:
    """The ``of-<hash>`` fingerprint is deterministic for an identical finding
    and — per issue #3326 — is unchanged by a reworded LLM-authored title."""
    finding = ReviewFinding(
        file="a.py", line=7, severity="bug", title="Race condition",
        description="d", fix_plan="f",
    )
    handle = _overflow_fingerprint(finding)
    assert handle.startswith("of-")
    assert len(handle) == len("of-") + 8

    # Same identity → same handle (re-derived from the diff each review run).
    same = ReviewFinding(
        file="a.py", line=7, severity="bug", title="Race condition",
        description="other description", fix_plan="other fix",
    )
    assert _overflow_fingerprint(same) == handle

    # Reworded title → SAME handle, so an ack posted under it keeps matching.
    reworded = ReviewFinding(
        file="a.py", line=7, severity="bug", title="Deadlock hazard",
        description="d", fix_plan="f",
    )
    assert _overflow_fingerprint(reworded) == handle

    # Severity, file, and line remain identity-bearing.
    for changed in (
        ReviewFinding(file="a.py", line=7, severity="nit", title="Race condition",
                      description="d", fix_plan="f"),
        ReviewFinding(file="b.py", line=7, severity="bug", title="Race condition",
                      description="d", fix_plan="f"),
        ReviewFinding(file="a.py", line=8, severity="bug", title="Race condition",
                      description="d", fix_plan="f"),
    ):
        assert _overflow_fingerprint(changed) != handle


@pytest.mark.parametrize(("severity", "file"), [("", "a.py"), ("bug", "")])
def test_overflow_fingerprint_rejects_non_derivable_identity(
    severity: str, file: str,
) -> None:
    """An empty severity or file makes the handle un-ackable, so minting one
    raises rather than emitting a handle no acknowledgement can ever match."""
    finding = ReviewFinding(
        file=file, line=7, severity=severity, title="t",
        description="d", fix_plan="f",
    )
    with pytest.raises(ValueError, match="non-empty severity and file"):
        _overflow_fingerprint(finding)


def test_legacy_title_keyed_ack_still_matches() -> None:
    """A roll-up comment carrying only the *old* title-derived handle still
    acknowledges the finding, so historical acks are not resurfaced once."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)

    # A real line number, so the current handle is the coarse site-only one and
    # genuinely differs from the legacy title-derived key.
    finding = ReviewFinding(file="a.py", line=5, severity="bug",
                            title="Race condition", description="d", fix_plan="f")
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([finding], {}, "review")["body"],
    ]
    legacy = _legacy_overflow_fingerprint(finding)
    assert legacy != _overflow_fingerprint(finding)
    host.get_pr_comment_bodies.return_value = [
        f"{_OVERFLOW_ACK_SENTINEL}\n- {legacy} — {_FIXED_LOCALLY_MARKER}",
    ]

    assert reviewer.find_unreplied_comments(7).overflow == []


def test_coarse_keyed_ack_suppresses_legacy_rendered_finding() -> None:
    """An ack posted under the *coarse* handle must also answer the copy of that
    finding surviving the union from a pre-#3326 legacy block, whose embedded
    handle is title-derived — otherwise the answered finding reads as unreplied."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)

    finding = ReviewFinding(file="a.py", line=5, severity="bug",
                            title="Race condition", description="d", fix_plan="f")
    coarse = _overflow_fingerprint(finding)
    legacy = _legacy_overflow_fingerprint(finding)
    assert coarse != legacy
    # Only the legacy block is still on the PR, so the union's representative
    # carries the legacy handle.
    host.get_pr_review_bodies.return_value = [
        f"<!-- maestro:overflow -->\n"
        f"- {legacy} **a.py:5** [bug] — Race condition: d\n"
        f"<!-- /maestro:overflow -->",
    ]
    host.get_pr_comment_bodies.return_value = [
        f"{_OVERFLOW_ACK_SENTINEL}\n- {coarse} — {_FIXED_LOCALLY_MARKER}",
    ]

    assert reviewer.find_unreplied_comments(7).overflow == []


def test_overflow_fingerprint_keeps_file_level_findings_distinct() -> None:
    """A file-level (``line <= 0``) finding folds the title back into the handle:
    the site alone is not an identity there, so two distinct findings in one file
    must not collapse to a single handle that one ack would suppress."""
    first = ReviewFinding(file="a.py", line=0, severity="bug",
                          title="Race condition", description="d", fix_plan="f")
    second = ReviewFinding(file="a.py", line=0, severity="bug",
                           title="Unbounded list growth", description="d",
                           fix_plan="f")
    assert _overflow_fingerprint(first) != _overflow_fingerprint(second)
    # At line 0 the current handle *is* the legacy key, so historical acks match.
    assert _overflow_fingerprint(first) == _legacy_overflow_fingerprint(first)


def test_union_overflow_findings_newest_title_wins_for_one_handle() -> None:
    """When two review bodies carry the *same* handle with drifted titles, the
    union keeps one entry and the newest body's text wins."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    old = ReviewFinding(file="a.py", line=7, severity="bug",
                        title="Race condition", description="d", fix_plan="f")
    new = ReviewFinding(file="a.py", line=7, severity="bug",
                        title="Deadlock hazard", description="d", fix_plan="f")
    assert _overflow_fingerprint(old) == _overflow_fingerprint(new)
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([old], {}, "old review")["body"],
        reviewer._build_review_payload([new], {}, "new review")["body"],
    ]

    overflow = reviewer.find_unreplied_comments(9).overflow

    assert [f.title for f in overflow] == ["Deadlock hazard"]


def test_union_overflow_keeps_two_distinct_findings_at_one_site() -> None:
    """Two different findings the reviewer emitted at the same out-of-diff
    ``severity|file|line`` share the coarse handle, so a fingerprint-keyed merge
    would silently drop one.  Both must survive the union — within one block a
    repeated handle marks distinct findings, and a re-render across blocks must
    still collapse to that same pair rather than duplicating them."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    first = ReviewFinding(file="a.py", line=7, severity="bug",
                          title="Race condition", description="d", fix_plan="f")
    second = ReviewFinding(file="a.py", line=7, severity="bug",
                           title="Unbounded growth", description="d", fix_plan="f")
    assert _overflow_fingerprint(first) == _overflow_fingerprint(second)
    body = reviewer._build_review_payload([first, second], {}, "review")["body"]
    host.get_pr_review_bodies.return_value = [body, body]

    overflow = reviewer.find_unreplied_comments(9).overflow

    assert sorted(f.title for f in overflow) == ["Race condition",
                                                 "Unbounded growth"]


def _overflow_body(*lines: str) -> str:
    """Assemble a review body carrying the sentinel-delimited block of *lines*.

    Renders the block directly so a test can embed an arbitrary ``of-<hash>``
    handle — including a pre-#3326 title-derived one, which nothing emits any
    more and so cannot be produced via ``_build_review_payload``.
    """
    return (
        "Summary.\n\n"
        + _OVERFLOW_SENTINEL_OPEN
        + "\n" + "\n".join(lines) + "\n"
        + _OVERFLOW_SENTINEL_CLOSE
    )


def test_union_overflow_collapses_legacy_title_derived_renderings() -> None:
    """A PR reviewed across the #3326 identity change carries a *different*
    embedded handle per legacy block for the same still-open finding.  Keying the
    union on those raw handles reported the finding once per historical review;
    the canonical key collapses them onto the current coarse handle."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    finding = ReviewFinding(file="a.py", line=7, severity="bug",
                            title="Race condition", description="d", fix_plan="f")
    coarse = _overflow_fingerprint(finding)
    # Two pre-#3326 blocks whose embedded handles are the title-derived keys of
    # two re-wordings of one finding, then the current block's coarse handle.
    legacy_one = _legacy_overflow_fingerprint(finding)
    legacy_two = _legacy_overflow_fingerprint(
        ReviewFinding(file="a.py", line=7, severity="bug",
                      title="Deadlock hazard", description="d", fix_plan="f"),
    )
    assert len({legacy_one, legacy_two, coarse}) == 3
    host.get_pr_review_bodies.return_value = [
        _overflow_body(f"- {legacy_one} **a.py:7** [bug] — Race condition: boom"),
        _overflow_body(f"- {legacy_two} **a.py:7** [bug] — Deadlock hazard: boom"),
        reviewer._build_review_payload([finding], {}, "current")["body"],
    ]

    overflow = reviewer.find_unreplied_comments(9).overflow

    assert [f.fingerprint for f in overflow] == [coarse]
    assert [f.title for f in overflow] == ["Race condition"]


def test_union_overflow_merge_key_keeps_file_level_renderings_distinct() -> None:
    """At ``line <= 0`` there is no coarse identity, so the canonical key falls
    back to the finding's own handle and two distinct file-level findings still
    survive the union separately."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    first = ReviewFinding(file="a.py", line=0, severity="bug",
                          title="Race condition", description="d", fix_plan="f")
    second = ReviewFinding(file="a.py", line=0, severity="bug",
                           title="Unbounded growth", description="d", fix_plan="f")
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([first], {}, "old")["body"],
        reviewer._build_review_payload([second], {}, "new")["body"],
    ]

    overflow = reviewer.find_unreplied_comments(9).overflow

    assert sorted(f.title for f in overflow) == ["Race condition",
                                                 "Unbounded growth"]


def test_union_overflow_keeps_widest_block_for_a_handle() -> None:
    """A newer block that re-renders only one of two same-site findings must not
    shrink the union: the block carrying the most occurrences of a handle wins,
    so the finding the newest review happened to omit stays selectable."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    first = ReviewFinding(file="a.py", line=7, severity="bug",
                          title="Race condition", description="d", fix_plan="f")
    second = ReviewFinding(file="a.py", line=7, severity="bug",
                           title="Unbounded growth", description="d", fix_plan="f")
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([first, second], {}, "old")["body"],
        reviewer._build_review_payload([first], {}, "new")["body"],
    ]

    overflow = reviewer.find_unreplied_comments(9).overflow

    assert sorted(f.title for f in overflow) == ["Race condition",
                                                 "Unbounded growth"]


# ------------------------------------------------------------------
# Overflow parsers (read side) — _parse_overflow_findings / _parse_overflow_ack
# ------------------------------------------------------------------


def test_parse_overflow_findings_round_trips_payload_all_severities() -> None:
    """The reader extracts exactly the overflow lines the writer renders, across
    every severity (the core reader keeps only bugs)."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(file="a.py", line=0, severity="bug",
                      title="Null line bug", description="off-hunk", fix_plan="f"),
        ReviewFinding(file="b.py", line=5, severity="suggestion",
                      title="Rename for clarity", description="d", fix_plan="f"),
    ]
    # Only a.py has valid diff lines, so both findings land in overflow.
    payload = reviewer._build_review_payload(findings, {"a.py": {1, 2}}, "Summary")

    parsed = _parse_overflow_findings(payload["body"])

    assert len(parsed) == 2
    assert {f.severity for f in parsed} == {"bug", "suggestion"}
    by_file = {f.file: f for f in parsed}
    assert by_file["a.py"].fingerprint == _overflow_fingerprint(findings[0])
    assert by_file["a.py"].line == 0
    assert by_file["a.py"].title == "Null line bug"
    assert by_file["a.py"].description == "off-hunk"
    assert by_file["b.py"].fingerprint == _overflow_fingerprint(findings[1])


def test_parse_overflow_findings_acceptance_two_findings_one_ack() -> None:
    """Step 2 acceptance: two overflow lines parse to two findings; an ack comment
    acking one fingerprint yields an ack set of size one."""
    reviewer = _reviewer()
    findings = [
        ReviewFinding(file="a.py", line=0, severity="bug",
                      title="First defect", description="d1", fix_plan="f"),
        ReviewFinding(file="b.py", line=0, severity="nit",
                      title="Second defect", description="d2", fix_plan="f"),
    ]
    payload = reviewer._build_review_payload(findings, {}, "Summary")

    parsed = _parse_overflow_findings(payload["body"])
    assert len(parsed) == 2

    acked_fp = _overflow_fingerprint(findings[0])
    ack_comment = f"{_OVERFLOW_ACK_SENTINEL}\n{acked_fp} — Fixed automatically"
    acked = _parse_overflow_ack(["unrelated chatter", ack_comment])

    assert acked == {acked_fp}


def test_parse_overflow_findings_skips_malformed_line() -> None:
    """A bullet that does not match the rendered shape is skipped, not fatal."""
    body = (
        f"{_OVERFLOW_SENTINEL_OPEN}\n"
        "- of-aaaaaaaa **a.py:3** [bug] — Good line: works\n"
        "- this is not a valid overflow bullet at all\n"
        f"{_OVERFLOW_SENTINEL_CLOSE}"
    )
    parsed = _parse_overflow_findings(body)
    assert len(parsed) == 1
    assert parsed[0].fingerprint == "of-aaaaaaaa"
    assert parsed[0].file == "a.py"
    assert parsed[0].line == 3


def test_parse_overflow_findings_empty_and_absent() -> None:
    """An empty body and a body with no overflow block both yield no findings."""
    assert _parse_overflow_findings("") == []
    assert _parse_overflow_findings("Just a summary, no overflow here.") == []


def test_parse_overflow_findings_legacy_header_fallback() -> None:
    """A body carrying the legacy header but no sentinels is still parsed, and a
    handle-less legacy line gets a recomputed fingerprint."""
    body = (
        f"{_OVERFLOW_LEGACY_HEADER}\n"
        "- **c.py:7** [nit] — Tweak spacing: minor\n"
        "\n"
        "Some later section that must not be parsed.\n"
        "- **d.py:1** [bug] — Not part of the block: ignore\n"
    )
    parsed = _parse_overflow_findings(body)

    assert len(parsed) == 1
    assert parsed[0].file == "c.py"
    assert parsed[0].severity == "nit"
    # No embedded handle → recomputed from the parsed fields.
    assert parsed[0].fingerprint == _overflow_fingerprint(
        ReviewFinding(file="c.py", line=7, severity="nit",
                      title="Tweak spacing", description="", fix_plan=""),
    )


def test_parse_overflow_findings_empty_description_form() -> None:
    """A rendered line whose description is empty parses with a blank description
    and the title closed at the trailing colon."""
    body = (
        f"{_OVERFLOW_SENTINEL_OPEN}\n"
        "- of-bbbbbbbb **e.py:2** [suggestion] — Lone title:\n"
        f"{_OVERFLOW_SENTINEL_CLOSE}"
    )
    parsed = _parse_overflow_findings(body)
    assert len(parsed) == 1
    assert parsed[0].title == "Lone title"
    assert parsed[0].description == ""


def test_parse_overflow_ack_mixed_lines() -> None:
    """Every of-<hash> handle on a terminal-verdict line under the ack sentinel is
    collected, across comments, and comments without the sentinel are ignored."""
    ack_one = (
        f"prose\n{_OVERFLOW_ACK_SENTINEL}\n"
        "of-11111111 — Fixed automatically\n"
        "of-22222222 — Could not prove this defect\n"
    )
    ack_two = f"{_OVERFLOW_ACK_SENTINEL}\nof-33333333 — Tracked in #42"
    non_ack = "of-99999999 mentioned in an unrelated comment"

    acked = _parse_overflow_ack([ack_one, non_ack, ack_two])

    assert acked == {"of-11111111", "of-22222222", "of-33333333"}


def test_parse_overflow_ack_requires_terminal_marker() -> None:
    """A handle listed under the sentinel without a terminal verdict on its line is
    NOT treated as answered — only the terminal-marked handle is collected.

    Guards the leniency the reviewer flagged: were every co-occurring handle
    acked, the planned reply-coverage gate would pass an overflow finding that was
    never closed (a quoted, in-progress, or prose handle)."""
    body = (
        f"Re-running remediation.\n{_OVERFLOW_ACK_SENTINEL}\n"
        "of-aaaaaaaa — Fixed automatically\n"
        "of-bbbbbbbb — still in progress, no verdict yet\n"
        "(see of-cccccccc quoted in the discussion above)\n"
    )

    acked = _parse_overflow_ack([body])

    assert acked == {"of-aaaaaaaa"}


def test_parse_overflow_ack_collects_only_first_handle_per_line() -> None:
    """A terminal-verdict line carrying more than one of-<hash> handle acks only the
    FIRST handle — the one the roll-up keyed the verdict to — not a handle merely
    quoted inside the verdict text.

    Guards parity with the core consumer ``_acknowledged_overflow_fingerprints``,
    which uses ``.search()`` (first match only).  Collecting every handle is the
    unsafe direction: the quoted ``of-bbbbbbbb`` was never closed, so treating it as
    answered would let the planned reply-coverage gate pass an unaddressed finding."""
    body = (
        f"{_OVERFLOW_ACK_SENTINEL}\n"
        "of-aaaaaaaa — Fixed automatically (was a dup of of-bbbbbbbb)\n"
    )

    acked = _parse_overflow_ack([body])

    assert acked == {"of-aaaaaaaa"}


def test_parse_overflow_ack_empty() -> None:
    """No comments and no ack sentinel both yield an empty set."""
    assert _parse_overflow_ack([]) == set()
    assert _parse_overflow_ack(["just a normal PR comment"]) == set()


def test_ai_finding_re_matches_each_icon() -> None:
    """The finding regex matches each severity icon the reviewer emits and
    rejects a human (non-icon) root comment."""
    for finding in (
        ReviewFinding(file="a.py", line=1, severity="bug",
                      title="A bug", description="d", fix_plan="f"),
        ReviewFinding(file="a.py", line=1, severity="suggestion",
                      title="A suggestion", description="d", fix_plan="f"),
        ReviewFinding(file="a.py", line=1, severity="nit",
                      title="A nit", description="d", fix_plan="f"),
    ):
        body = PRReviewer._format_finding_body(finding)
        match = _AI_FINDING_RE.match(body)
        assert match is not None
        assert match.group(1) == finding.title

    assert _AI_FINDING_RE.match("Just a human review comment") is None


def test_overflow_finding_is_frozen() -> None:
    """OverflowFinding is an immutable value object."""
    finding = OverflowFinding(
        fingerprint="of-12345678", file="a.py", line=1,
        severity="bug", title="t", description="d",
    )
    with pytest.raises(FrozenInstanceError):
        finding.line = 2  # type: ignore[misc]


# ------------------------------------------------------------------
# post_review
# ------------------------------------------------------------------


def test_post_review_success() -> None:
    host = MagicMock()
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="a.py", line=2, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix",
        ),
    ]

    result = reviewer.post_review(10, findings, "Summary", diff=MULTI_FILE_DIFF)

    host.post_pr_review.assert_called_once()
    pr_number, payload = host.post_pr_review.call_args[0]
    assert pr_number == 10
    assert payload["event"] == "COMMENT"
    assert len(payload["comments"]) == 1
    assert isinstance(result, PostReviewResult)
    assert result.posted == findings
    assert result.reopened == []
    assert result.filtered == []
    assert result.skipped == []


def test_post_review_failure_raises() -> None:
    host = MagicMock()
    host.list_issues.return_value = []
    host.post_pr_review.side_effect = GitHubError("failed")
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="a.py", line=2, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix",
        ),
    ]

    with pytest.raises(GitHubError):
        reviewer.post_review(10, findings, "Summary")


def test_post_review_no_diff_all_findings_posted() -> None:
    """When no diff is provided, all findings are posted as inline."""
    host = MagicMock()
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="a.py", line=5, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix",
        ),
    ]

    result = reviewer.post_review(10, findings, "Summary")  # no diff

    _, payload = host.post_pr_review.call_args[0]
    assert len(payload["comments"]) == 1
    assert result.posted == findings


def test_post_review_unparseable_diff_findings_go_to_overflow() -> None:
    """When a diff is provided but parses to empty (e.g. binary diff),
    findings must go to the review body (overflow), not as inline comments.

    The old code treated empty parse results the same as no-diff-provided,
    making all findings appear as inline comments — even when their line
    numbers are invalid for the actual diff.
    """
    host = MagicMock()
    host.list_issues.return_value = []
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="a.py", line=5, severity="bug", title="Bug",
            description="Issue", fix_plan="Fix",
        ),
    ]

    # Pass a diff that has content but no parseable hunks
    result = reviewer.post_review(10, findings, "Summary", diff="Binary files differ\n")

    _, payload = host.post_pr_review.call_args[0]
    # Finding should be in the overflow body, NOT as an inline comment
    assert len(payload["comments"]) == 0, (
        "Findings should be in overflow when diff is provided but unparseable"
    )
    assert "a.py:5" in payload["body"]
    assert result.posted == findings


def test_post_review_filters_issue_tracked_bugs() -> None:
    """Bug findings tracked by an open issue are excluded from the posted review."""
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unused import in maestro/review.py", "body": "unused import detected"},
    ]
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="maestro/review.py", line=13, severity="bug",
            title="Unused import detected",
            description="os is imported but never used", fix_plan="Remove",
        ),
        ReviewFinding(
            file="maestro/agent.py", line=5, severity="bug",
            title="Missing null check",
            description="x may be None", fix_plan="Add guard",
        ),
    ]

    result = reviewer.post_review(10, findings, "Summary", diff=MULTI_FILE_DIFF)

    _, payload = host.post_pr_review.call_args[0]
    # Only the unmatched finding should be posted
    bodies = [c["body"] for c in payload["comments"]]
    assert any("null check" in b.lower() for b in bodies) or "agent.py" in payload["body"]
    host.list_issues.assert_called_once_with(state="open")
    assert len(result.filtered) == 1
    assert result.filtered[0].title == "Unused import detected"
    assert len(result.posted) == 1
    assert result.posted[0].title == "Missing null check"
    # Review body should note the filtered finding
    assert "already covered by open issues" in payload["body"]
    assert "Unused import detected" in payload["body"]


def test_post_review_skips_when_all_bugs_filtered() -> None:
    """When all bug findings are tracked by issues, no review is posted."""
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unused import in maestro/review.py", "body": "unused import detected"},
    ]
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)
    findings = [
        ReviewFinding(
            file="maestro/review.py", line=13, severity="bug",
            title="Unused import detected",
            description="os is imported but never used", fix_plan="Remove",
        ),
    ]

    result = reviewer.post_review(10, findings, "Summary", diff=MULTI_FILE_DIFF)

    host.post_pr_review.assert_not_called()
    assert result.posted == []
    assert result.reopened == []
    assert result.filtered == findings
    assert result.skipped == []


# ------------------------------------------------------------------
# _extract_keywords
# ------------------------------------------------------------------


def test_extract_keywords_basic() -> None:
    result = PRReviewer._extract_keywords("Unused import os")
    assert "unused" in result
    assert "import" in result
    # "os" is only 2 chars — excluded
    assert "os" not in result


def test_extract_keywords_underscore_identifiers() -> None:
    result = PRReviewer._extract_keywords("Missing file_lock usage")
    assert "file_lock" in result
    assert "missing" in result
    assert "usage" in result


def test_extract_keywords_stops_removed() -> None:
    result = PRReviewer._extract_keywords("the function has and was not for this")
    assert "function" in result
    for stop in ("the", "has", "and", "was", "not", "for", "this"):
        assert stop not in result


def test_extract_keywords_empty() -> None:
    assert PRReviewer._extract_keywords("") == set()


# ------------------------------------------------------------------
# _is_finding_tracked
# ------------------------------------------------------------------


def _make_finding(
    file: str = "maestro/review.py",
    title: str = "Unused import detected",
) -> ReviewFinding:
    return ReviewFinding(
        file=file, line=10, severity="suggestion",
        title=title, description="desc", fix_plan="fix",
    )


def test_is_finding_tracked_match() -> None:
    issues = [
        {"title": "review.py: unused import and dead code",
         "body": "maestro/review.py has unused imports that should be cleaned up"},
    ]
    assert PRReviewer._is_finding_tracked(_make_finding(), issues) is True


def test_is_finding_tracked_no_file_match() -> None:
    """Issue mentions a different file — no match."""
    issues = [
        {"title": "Unused import in maestro/agent.py",
         "body": "maestro/agent.py unused import detected"},
    ]
    assert PRReviewer._is_finding_tracked(_make_finding(), issues) is False


def test_is_finding_tracked_no_keyword_overlap() -> None:
    """File matches but title keywords do not overlap enough."""
    issues = [
        {"title": "Race condition in maestro/review.py",
         "body": "maestro/review.py has a race condition in the writer"},
    ]
    assert PRReviewer._is_finding_tracked(_make_finding(), issues) is False


def test_is_finding_tracked_empty_issues() -> None:
    assert PRReviewer._is_finding_tracked(_make_finding(), []) is False


def test_is_finding_tracked_short_title() -> None:
    """Title with fewer than 2 extractable keywords is never considered tracked."""
    finding = _make_finding(title="Bug")
    issues = [
        {"title": "Bug in maestro/review.py",
         "body": "maestro/review.py has a bug"},
    ]
    assert PRReviewer._is_finding_tracked(finding, issues) is False


def test_is_finding_tracked_empty_file_path_not_matched() -> None:
    """A finding with an empty file path must never match an issue.

    In Python, ``"" in some_string`` is always ``True``, so an empty file
    path would bypass the file-match guard and match any issue that shares
    enough keywords — incorrectly filtering out the finding.
    """
    finding = _make_finding(file="", title="Unused import detected")
    issues = [
        {"title": "Unused import in maestro/review.py",
         "body": "unused import detected in maestro/review.py"},
    ]
    assert PRReviewer._is_finding_tracked(finding, issues) is False


def test_is_finding_tracked_no_lookahead_false_positive_pylib() -> None:
    """'a.py' should NOT match an issue that only mentions 'a.pylib'."""
    finding = _make_finding(
        file="a.py",
        title="missing error handling check",
    )
    issues = [
        {
            "title": "Bug in a.pylib error handling check",
            "body": "The a.pylib file has a problem",
        }
    ]
    assert PRReviewer._is_finding_tracked(finding, issues) is False


def test_is_finding_tracked_no_lookahead_false_positive_bak() -> None:
    """'a.py' should NOT match an issue that only mentions 'a.py.bak'."""
    finding = _make_finding(
        file="a.py",
        title="missing error handling check",
    )
    issues = [
        {
            "title": "Bug in a.py.bak error handling check",
            "body": "The a.py.bak file has a problem",
        }
    ]
    assert PRReviewer._is_finding_tracked(finding, issues) is False


# ------------------------------------------------------------------
# _filter_tracked_findings
# ------------------------------------------------------------------


def test_filter_tracked_findings_removes_matched_bug() -> None:
    """Bug findings tracked by an open issue are filtered out."""
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unused import in maestro/review.py",
         "body": "unused import detected in maestro/review.py"},
    ]
    reviewer = _reviewer(host)

    bug_finding = ReviewFinding(
        file="maestro/review.py", line=10, severity="bug",
        title="Unused import detected", description="desc", fix_plan="fix",
    )
    findings = [
        bug_finding,
        _make_finding(file="maestro/agent.py", title="Missing null check"),
    ]
    result = reviewer._filter_tracked_findings(findings)

    assert len(result) == 1
    assert result[0].file == "maestro/agent.py"


def test_filter_tracked_findings_all_severities_filtered() -> None:
    """Findings of any severity tracked by an open issue are filtered out."""
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unused import in maestro/review.py",
         "body": "unused import detected in maestro/review.py"},
    ]
    reviewer = _reviewer(host)

    # _make_finding defaults to severity="suggestion"
    findings = [_make_finding()]
    result = reviewer._filter_tracked_findings(findings)

    assert result == []  # suggestions are also filtered when tracked


def test_filter_tracked_findings_all_bugs_tracked() -> None:
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unused import in maestro/review.py",
         "body": "unused import detected in maestro/review.py"},
    ]
    reviewer = _reviewer(host)

    bug = ReviewFinding(
        file="maestro/review.py", line=10, severity="bug",
        title="Unused import detected", description="desc", fix_plan="fix",
    )
    result = reviewer._filter_tracked_findings([bug])

    assert result == []


def test_filter_tracked_findings_none_tracked() -> None:
    host = MagicMock()
    host.list_issues.return_value = [
        {"title": "Unrelated issue", "body": "Nothing relevant"},
    ]
    reviewer = _reviewer(host)

    findings = [_make_finding()]
    result = reviewer._filter_tracked_findings(findings)

    assert result == findings


def test_filter_tracked_findings_api_failure() -> None:
    """When list_issues returns None (API error), all findings pass through."""
    host = MagicMock()
    host.list_issues.return_value = None
    reviewer = _reviewer(host)

    findings = [_make_finding()]
    result = reviewer._filter_tracked_findings(findings)

    assert result == findings


def test_filter_tracked_findings_no_issues() -> None:
    """When there are no open issues, all findings pass through."""
    host = MagicMock()
    host.list_issues.return_value = []
    reviewer = _reviewer(host)

    findings = [_make_finding()]
    result = reviewer._filter_tracked_findings(findings)

    assert result == findings


# ------------------------------------------------------------------
# _match_finding_to_thread
# ------------------------------------------------------------------


def _make_thread(
    *,
    path: str = "maestro/review.py",
    body: str = "\U0001f41b **Unused import detected**\n\nos is unused",
    is_resolved: bool = False,
    thread_id: str = "PRRT_1",
    comment_id: int = 100,
    all_bodies: str = "",
) -> ReviewThread:
    return ReviewThread(
        thread_id=thread_id,
        comment_id=comment_id,
        path=path,
        body=body,
        all_bodies=all_bodies or body,
        is_resolved=is_resolved,
    )


def test_parse_review_threads_null_comments() -> None:
    """GraphQL can return comments: null — should not crash."""
    nodes = [
        {"id": "PRRT_1", "isResolved": False, "comments": None},
        {"id": "PRRT_2", "isResolved": False},  # missing key entirely
        {
            "id": "PRRT_3",
            "isResolved": False,
            "comments": {"nodes": [None, 42, "bad"]},  # non-dict comment nodes
        },
    ]
    threads = PRReviewer._parse_review_threads(nodes)
    # All three nodes should be skipped gracefully (no usable comments)
    assert threads == []


def test_parse_review_threads_null_database_id() -> None:
    """GraphQL can return databaseId: null — should default to 0."""
    nodes = [{
        "id": "PRRT_1",
        "isResolved": False,
        "comments": {"nodes": [
            {"databaseId": None, "path": "a.py", "body": "Bug here"},
        ]},
    }]
    threads = PRReviewer._parse_review_threads(nodes)
    assert len(threads) == 1
    assert threads[0].comment_id == 0


def test_is_finding_tracked_null_issue_body() -> None:
    """GitHub API can return body: null — must not crash on string concatenation."""
    finding = _make_finding()
    issues = [
        {"title": "unused import in maestro/review.py", "body": None},
    ]
    # Should not raise TypeError
    PRReviewer._is_finding_tracked(finding, issues)


def test_match_finding_to_thread_empty_title_returns_none() -> None:
    """A finding with an empty title cannot match any thread."""
    thread = _make_thread()
    finding = _make_finding(title="")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_match() -> None:
    thread = _make_thread()
    finding = _make_finding()
    result = PRReviewer._match_finding_to_thread(finding, [thread])
    assert result is thread


def test_match_finding_to_thread_no_match() -> None:
    thread = _make_thread(path="maestro/agent.py")
    finding = _make_finding()
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_case_insensitive() -> None:
    thread = _make_thread(body="**UNUSED IMPORT DETECTED**")
    finding = _make_finding()
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is not None


def test_match_finding_to_thread_different_title() -> None:
    thread = _make_thread(body="**Race condition**\nConcurrent writes")
    finding = _make_finding()
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_variant_wording() -> None:
    """Wording drift on the same defect still matches the existing thread."""
    thread = _make_thread(body="\U0001f41b **Unused import detected**\n\nos is unused")
    finding = _make_finding(title="Unused import")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is thread


def test_match_finding_to_thread_fuzzy_reordered_tokens() -> None:
    """Token order in the title does not matter for fuzzy match."""
    thread = _make_thread(body="\U0001f41b **Unused import**")
    finding = _make_finding(title="Import is unused")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is thread


def test_match_finding_to_thread_fuzzy_filler_only_difference() -> None:
    """Differences confined to filler/stop words still fuzzy-match."""
    thread = _make_thread(body="\U0001f41b **Race condition detected**")
    finding = _make_finding(title="Possible race condition")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is thread


def test_match_finding_to_thread_fuzzy_distinct_findings_no_match() -> None:
    """Semantically distinct findings on the same file do not fuzzy-match."""
    thread = _make_thread(body="\U0001f41b **Race condition in shared cache**")
    finding = _make_finding(title="Unused import")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_single_significant_token_rejected() -> None:
    """A single-token finding never fuzzy-matches another single-token thread."""
    thread = _make_thread(body="\U0001f41b **Memory leak**")
    finding = _make_finding(title="Leak")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_partial_overlap_below_threshold() -> None:
    """Token overlap below the 0.6 coefficient threshold is treated as distinct."""
    thread = _make_thread(body="\U0001f41b **Unannotated return type**")
    finding = _make_finding(title="Missing type hint on parameter")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_requires_path_match() -> None:
    """Fuzzy match never crosses file paths."""
    thread = _make_thread(
        path="maestro/agent.py",
        body="\U0001f41b **Unused import detected**",
    )
    finding = _make_finding(file="maestro/review.py", title="Unused import")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_skips_thread_without_bold_block() -> None:
    """A thread body with no bold block is ignored by the fuzzy pass."""
    thread = _make_thread(body="plain text with no bold marker about an unused import")
    finding = _make_finding(title="Unused import in module")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_match_finding_to_thread_fuzzy_empty_normalized_set() -> None:
    """A title made entirely of stop-words yields no fuzzy match."""
    thread = _make_thread(body="\U0001f41b **Unused import detected**")
    finding = _make_finding(title="The possible issue")
    assert PRReviewer._match_finding_to_thread(finding, [thread]) is None


def test_normalize_title_strips_punctuation_and_stopwords() -> None:
    """``_normalize_title`` drops punctuation, stop-words, and short tokens."""
    assert _normalize_title("The unused import, detected!") == frozenset({"unused", "import"})


def test_normalize_title_empty_returns_empty_set() -> None:
    """Empty input normalizes to the empty set."""
    assert _normalize_title("") == frozenset()


def test_fuzzy_title_match_below_min_tokens() -> None:
    """Both sides must clear ``_FUZZY_MIN_SIGNIFICANT_TOKENS``."""
    assert _fuzzy_title_match(frozenset({"leak"}), frozenset({"memory", "leak"})) is False


def test_substantive_defect_nouns_preserved_for_fuzzy_match() -> None:
    """Substantive defect nouns (``error``) survive stop-word stripping so that
    short titles like "Missing error handling" still clear the significant-token
    floor and fuzzy-match equivalent wordings like "No error handling".
    """
    a = _normalize_title("Missing error handling")
    b = _normalize_title("No error handling")
    # Both sides should retain enough significant tokens for fuzzy matching.
    assert len(a) >= 2, f"expected >=2 significant tokens, got {a}"
    assert len(b) >= 2, f"expected >=2 significant tokens, got {b}"
    assert _fuzzy_title_match(a, b), (
        f"expected {a!r} and {b!r} to fuzzy-match as the same finding"
    )


def test_extract_thread_title_handles_icon_prefix() -> None:
    """The leading severity icon does not interfere with title extraction."""
    assert _extract_thread_title("\U0001f41b **Unused import**\n\nos is unused") == "Unused import"


def test_extract_thread_title_returns_none_when_no_bold() -> None:
    """Bodies without a bold block return ``None``."""
    assert _extract_thread_title("plain comment text") is None


def test_extract_thread_title_returns_none_when_bold_is_blank() -> None:
    """A bold block containing only whitespace returns ``None``."""
    assert _extract_thread_title("**   **") is None


# ------------------------------------------------------------------
# _thread_has_issue_link
# ------------------------------------------------------------------


def test_thread_has_issue_link_url() -> None:
    thread = _make_thread(
        all_bodies="Fixed via https://github.com/owner/repo/issues/42",
    )
    assert PRReviewer._thread_has_issue_link(thread) is True


def test_thread_has_issue_link_hash_ref() -> None:
    thread = _make_thread(all_bodies="Tracked in #42")
    assert PRReviewer._thread_has_issue_link(thread) is True


def test_thread_has_issue_link_no_link() -> None:
    thread = _make_thread(all_bodies="Just a regular comment")
    assert PRReviewer._thread_has_issue_link(thread) is False


# ------------------------------------------------------------------
# _categorize_by_thread_state
# ------------------------------------------------------------------


def _graphql_thread_node(
    *,
    thread_id: str = "PRRT_1",
    is_resolved: bool = False,
    is_outdated: bool = False,
    comment_id: int = 100,
    path: str = "maestro/review.py",
    body: str = "\U0001f41b **Unused import detected**\n\nos is unused",
    extra_comments: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a raw GraphQL thread node dict."""
    comments: list[dict[str, object]] = [
        {"databaseId": comment_id, "path": path, "body": body},
    ]
    if extra_comments:
        comments.extend(extra_comments)
    return {
        "id": thread_id,
        "isResolved": is_resolved,
        "isOutdated": is_outdated,
        "comments": {"nodes": comments},
    }


def test_categorize_new_finding() -> None:
    """Finding with no matching thread goes to 'new'."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)

    new, skip, reopen = reviewer._categorize_by_thread_state(
        10, [_make_finding()],
    )
    assert len(new) == 1
    assert skip == []
    assert reopen == []


def test_categorize_skip_unresolved() -> None:
    """Finding matching an unresolved thread goes to 'skip'."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(is_resolved=False),
    ]
    reviewer = _reviewer(host)

    new, skip, reopen = reviewer._categorize_by_thread_state(
        10, [_make_finding()],
    )
    assert new == []
    assert len(skip) == 1
    assert reopen == []


def test_categorize_reopen_resolved() -> None:
    """Bug matching a resolved thread without issue link goes to 'reopen'."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(is_resolved=True),
    ]
    reviewer = _reviewer(host)

    bug = ReviewFinding(
        file="maestro/review.py", line=10, severity="bug",
        title="Unused import detected", description="desc", fix_plan="fix",
    )
    new, skip, reopen = reviewer._categorize_by_thread_state(10, [bug])
    assert new == []
    assert skip == []
    assert len(reopen) == 1
    assert reopen[0][0].title == "Unused import detected"


def test_categorize_skip_resolved_with_issue_link() -> None:
    """Resolved thread with issue link in replies goes to 'skip'."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            is_resolved=True,
            extra_comments=[
                {"databaseId": 101, "path": "maestro/review.py",
                 "body": "Created https://github.com/owner/repo/issues/42"},
            ],
        ),
    ]
    reviewer = _reviewer(host)

    new, skip, reopen = reviewer._categorize_by_thread_state(
        10, [_make_finding()],
    )
    assert new == []
    assert len(skip) == 1
    assert reopen == []


def test_categorize_mixed() -> None:
    """Multiple findings: new, skipped, and reopened."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(thread_id="PRRT_1", is_resolved=False),
        _graphql_thread_node(
            thread_id="PRRT_2", is_resolved=True,
            path="maestro/agent.py", comment_id=200,
            body="\U0001f41b **Missing null check**\nx may be None",
        ),
    ]
    reviewer = _reviewer(host)

    bug_finding = ReviewFinding(
        file="maestro/agent.py", line=10, severity="bug",
        title="Missing null check", description="desc", fix_plan="fix",
    )
    findings = [
        _make_finding(),  # matches PRRT_1 (unresolved) → skip
        bug_finding,  # matches PRRT_2 (resolved, bug) → reopen
        _make_finding(file="maestro/cli.py", title="Broken argument parsing"),  # no match → new
    ]
    new, skip, reopen = reviewer._categorize_by_thread_state(10, findings)

    assert len(new) == 1
    assert new[0].file == "maestro/cli.py"
    assert len(skip) == 1
    assert skip[0].file == "maestro/review.py"
    assert len(reopen) == 1
    assert reopen[0][0].file == "maestro/agent.py"


def test_categorize_api_failure() -> None:
    """When get_pr_review_threads returns empty, all findings are new."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    reviewer = _reviewer(host)

    findings = [_make_finding()]
    new, skip, reopen = reviewer._categorize_by_thread_state(10, findings)

    assert len(new) == 1
    assert skip == []
    assert reopen == []


# ------------------------------------------------------------------
# post_review — thread management integration
# ------------------------------------------------------------------


def test_post_review_unresolves_and_replies() -> None:
    """Resolved bug thread without issue link is unreolved + replied to."""
    host = MagicMock()
    host.list_issues.return_value = []
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            thread_id="PRRT_99", comment_id=555, is_resolved=True,
        ),
    ]
    reviewer = _reviewer(host)
    bug = ReviewFinding(
        file="maestro/review.py", line=10, severity="bug",
        title="Unused import detected", description="desc", fix_plan="fix",
    )

    result = reviewer.post_review(
        10, [bug], "Summary", diff=MULTI_FILE_DIFF,
    )

    host.reply_to_review_comment.assert_called_once()
    _, kwargs = host.reply_to_review_comment.call_args
    if not kwargs:
        args = host.reply_to_review_comment.call_args[0]
        assert args[0] == 10
        assert args[1] == 555
        assert "defect persists" in args[2].lower()
    # No new review posted (all findings went to reopen)
    host.post_pr_review.assert_not_called()
    assert result.reopened == [bug]
    assert result.posted == []


def test_post_review_skips_unresolved_thread() -> None:
    """Finding with unresolved thread is skipped — no new comment or reply."""
    host = MagicMock()
    host.list_issues.return_value = []
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(is_resolved=False),
    ]
    reviewer = _reviewer(host)
    finding = _make_finding()

    result = reviewer.post_review(
        10, [finding], "Summary", diff=MULTI_FILE_DIFF,
    )

    host.reply_to_review_comment.assert_not_called()
    host.post_pr_review.assert_not_called()
    assert result.skipped == [finding]
    assert result.posted == []


def test_post_review_skips_resolved_with_issue() -> None:
    """Resolved thread with issue link is skipped entirely."""
    host = MagicMock()
    host.list_issues.return_value = []
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            is_resolved=True,
            extra_comments=[
                {"databaseId": 101, "path": "maestro/review.py",
                 "body": "Tracked in #42"},
            ],
        ),
    ]
    reviewer = _reviewer(host)
    finding = _make_finding()

    result = reviewer.post_review(
        10, [finding], "Summary", diff=MULTI_FILE_DIFF,
    )

    host.reply_to_review_comment.assert_not_called()
    host.post_pr_review.assert_not_called()
    assert result.skipped == [finding]
    assert result.posted == []


def test_post_review_mixed_new_and_reopen() -> None:
    """New findings are posted; resolved bug threads get a reply."""
    host = MagicMock()
    host.list_issues.return_value = []
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            thread_id="PRRT_1", is_resolved=True,
            path="maestro/agent.py", comment_id=200,
            body="\U0001f41b **Missing null check**\nx may be None",
        ),
    ]
    reviewer = _reviewer(host)

    new_finding = _make_finding()
    reopen_finding = ReviewFinding(
        file="maestro/agent.py", line=10, severity="bug",
        title="Missing null check", description="desc", fix_plan="fix",
    )
    findings = [new_finding, reopen_finding]
    result = reviewer.post_review(10, findings, "Summary", diff=MULTI_FILE_DIFF)

    # Reopen path — reply posted for defect-persists thread
    # reply_to_review_comment called: once for defect-persists reply, possibly for issue linking
    assert host.reply_to_review_comment.call_count >= 1
    # New findings path
    host.post_pr_review.assert_called_once()
    assert result.posted == [new_finding]
    assert result.reopened == [reopen_finding]


# ------------------------------------------------------------------
# find_unreplied_comments (Step 3 — reply-coverage audit)
# ------------------------------------------------------------------


def test_find_unreplied_comments_acceptance_matrix() -> None:
    """Step 3 acceptance: across a replied AI thread, a bare AI thread, a resolved
    bare thread, and a non-AI root — plus one acked and one un-acked overflow
    finding — the report contains exactly the bare AI thread and the un-acked
    finding."""
    host = MagicMock()
    bare = ReviewFinding(file="a.py", line=1, severity="bug",
                         title="Bare unreplied bug", description="d", fix_plan="f")
    replied = ReviewFinding(file="b.py", line=2, severity="suggestion",
                            title="Replied suggestion", description="d", fix_plan="f")
    resolved = ReviewFinding(file="c.py", line=3, severity="nit",
                             title="Resolved nit", description="d", fix_plan="f")
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            thread_id="T_bare", path="a.py", comment_id=10,
            body=PRReviewer._format_finding_body(bare),
        ),
        _graphql_thread_node(
            thread_id="T_replied", path="b.py", comment_id=20,
            body=PRReviewer._format_finding_body(replied),
            extra_comments=[
                {"databaseId": 21, "path": "b.py", "body": "Thanks, fixed."},
            ],
        ),
        _graphql_thread_node(
            thread_id="T_resolved", path="c.py", comment_id=30,
            is_resolved=True,
            body=PRReviewer._format_finding_body(resolved),
        ),
        _graphql_thread_node(
            thread_id="T_human", path="d.py", comment_id=40,
            body="A human review note, no icon.",
        ),
    ]
    reviewer = _reviewer(host)

    # Two overflow findings; the first is acked, the second is not.
    of_acked = ReviewFinding(file="e.py", line=0, severity="bug",
                             title="Acked overflow", description="d", fix_plan="f")
    of_open = ReviewFinding(file="f.py", line=0, severity="suggestion",
                            title="Open overflow", description="d", fix_plan="f")
    payload = reviewer._build_review_payload([of_acked, of_open], {}, "Summary")
    host.get_pr_review_bodies.return_value = [payload["body"]]
    acked_fp = _overflow_fingerprint(of_acked)
    host.get_pr_comment_bodies.return_value = [
        f"{_OVERFLOW_ACK_SENTINEL}\n{acked_fp} — Fixed automatically",
    ]

    report = reviewer.find_unreplied_comments(7)

    assert [t.thread_id for t in report.inline] == ["T_bare"]
    assert report.inline[0].path == "a.py"
    assert [f.fingerprint for f in report.overflow] == [_overflow_fingerprint(of_open)]
    assert report.overflow[0].file == "f.py"


def test_find_unreplied_comments_outdated_thread_exempt() -> None:
    """An outdated bare AI thread is exempt (matches human-paced comment-remediation
    eligibility), so it is not reported."""
    host = MagicMock()
    finding = ReviewFinding(file="a.py", line=1, severity="bug",
                            title="Outdated bug", description="d", fix_plan="f")
    host.get_pr_review_threads.return_value = [
        _graphql_thread_node(
            is_outdated=True, body=PRReviewer._format_finding_body(finding),
        ),
    ]
    host.get_pr_review_bodies.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    report = reviewer.find_unreplied_comments(1)

    assert report.inline == []
    assert report.overflow == []


def test_find_unreplied_comments_empty_when_no_comments() -> None:
    """No review threads and no overflow review bodies yield an empty report —
    the no-threads / empty-data case stays deterministic and does not crash."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_review_bodies.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    report = reviewer.find_unreplied_comments(3)

    assert report.inline == []
    assert report.overflow == []


def test_find_unreplied_comments_fails_closed_on_connectivity_outage() -> None:
    """A gh/GraphQL outage raises instead of being mistaken for "all replied".

    The host's read methods swallow failures and return empty, so the audit
    probes connectivity first; when that probe raises, find_unreplied_comments
    propagates the error (the CLI gate then exits 1 — fail closed) rather than
    returning an empty report and passing the gate falsely.
    """
    host = MagicMock()
    host.check_connectivity.side_effect = GitHubError("rate limited")
    reviewer = _reviewer(host)

    with pytest.raises(GitHubError):
        reviewer.find_unreplied_comments(5)

    host.check_connectivity.assert_called_once_with(5)
    # The probe short-circuits before any best-effort fetch is trusted.
    host.get_pr_review_threads.assert_not_called()


def test_find_unreplied_comments_unions_overflow_bodies() -> None:
    """When several reviews carry an overflow block, every one contributes — the
    block is non-cumulative, so an older body's finding is still open (#3326)."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    old = ReviewFinding(file="old.py", line=0, severity="bug",
                        title="Stale overflow", description="d", fix_plan="f")
    new = ReviewFinding(file="new.py", line=0, severity="bug",
                        title="Current overflow", description="d", fix_plan="f")
    # Oldest-first: the most recent body carries only ``new``.
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([old], {}, "old review")["body"],
        reviewer._build_review_payload([new], {}, "new review")["body"],
    ]

    report = reviewer.find_unreplied_comments(9)

    assert sorted(f.file for f in report.overflow) == ["new.py", "old.py"]


def test_find_unreplied_comments_skips_newer_body_without_overflow_block() -> None:
    """The block-less-body skip path: a review body carrying no overflow block
    contributes nothing to :meth:`_union_overflow_findings`, while the older body
    that does carry one still reports its still-open finding."""
    host = MagicMock()
    host.get_pr_review_threads.return_value = []
    host.get_pr_comment_bodies.return_value = []
    reviewer = _reviewer(host)

    old = ReviewFinding(file="old.py", line=0, severity="suggestion",
                        title="Earlier overflow", description="d", fix_plan="f")
    # Oldest-first: an older review carrying the overflow block, then a newer
    # review that is a plain summary with no findings (no overflow block).
    host.get_pr_review_bodies.return_value = [
        reviewer._build_review_payload([old], {}, "old review")["body"],
        reviewer._build_review_payload([], {}, "newer review — no findings")["body"],
    ]

    report = reviewer.find_unreplied_comments(11)

    assert [f.file for f in report.overflow] == ["old.py"]
    assert [f.fingerprint for f in report.overflow] == [_overflow_fingerprint(old)]


def test_unreplied_comments_defaults_to_empty_lists() -> None:
    """UnrepliedComments defaults both lists to empty (independent instances)."""
    a = UnrepliedComments()
    b = UnrepliedComments()
    assert a.inline == []
    assert a.overflow == []
    a.inline.append(MagicMock())
    assert b.inline == []
