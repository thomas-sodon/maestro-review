"""Tests for the code review prompt template."""

from __future__ import annotations

from maestro_review.prompt import CODE_REVIEW_PROMPT


def test_prompt_is_repo_neutral() -> None:
    """The prompt must not frame every PR as belonging to one project."""
    assert "Maestro" not in CODE_REVIEW_PROMPT


def test_prompt_formats_with_review_placeholders() -> None:
    """The template formats with exactly the placeholders ``review`` passes."""
    rendered = CODE_REVIEW_PROMPT.format(pr_title="T", pr_body="B", diff="D")
    assert "Title: T" in rendered
