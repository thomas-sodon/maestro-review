"""TDD bug hunt tests for maestro_review/sdk.py edge cases."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from maestro_review.sdk import QueryResult, _truncate


class TestTruncateEdgeCases:
    """_truncate must not crash on max_len <= 0."""

    def test_truncate_zero_max_len(self) -> None:
        """max_len=0 should return empty string, not crash."""
        assert _truncate("hello", max_len=0) == ""

    def test_truncate_negative_max_len(self) -> None:
        """Negative max_len should return empty string."""
        assert _truncate("hello", max_len=-5) == ""


class TestParseJsonFencedBlockFallthrough:
    """parse_json falls through when a fenced code block fails to parse."""

    def test_fenced_block_invalid_json_falls_through_to_bracket(self) -> None:
        """If fenced block has invalid JSON, should fall through to bracket extraction."""
        text = '```json\nnot valid json\n```\nHere is the real answer: {"key": "value"}'
        result = QueryResult(response=text)
        parsed = result.parse_json()
        assert parsed == {"key": "value"}

    def test_fenced_block_invalid_json_no_fallback_raises(self) -> None:
        """If fenced block is invalid and no bracket extraction possible, should raise."""
        text = "```json\nnot valid json\n```\nno other JSON here"
        result = QueryResult(response=text)
        with pytest.raises(json.JSONDecodeError):
            result.parse_json()


class TestCostDisplayZero:
    """cost_str must not show 'N/A' when total_cost_usd is 0.0."""

    @pytest.mark.asyncio
    async def test_zero_cost_logged_correctly(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A cost of $0.00 should log as '$0.0000', not 'N/A'."""
        from claude_agent_sdk.types import ResultMessage

        mock_result_msg = ResultMessage(
            subtype="result",
            result="test response",
            total_cost_usd=0.0,
            num_turns=1,
            duration_ms=100,
            duration_api_ms=90,
            session_id="test-session",
            usage=None,
            is_error=False,
        )

        async def mock_query_gen(**kwargs):  # type: ignore[no-untyped-def]
            yield mock_result_msg

        with (
            patch("maestro_review.sdk._query", side_effect=mock_query_gen),
            caplog.at_level(logging.INFO, logger="maestro_review"),
        ):
            from maestro_review.sdk import query
            await query("test prompt")

        # Check the log message contains $0.0000, not N/A
        cost_logs = [r for r in caplog.records if "cost=" in r.message]
        assert cost_logs, "Expected a log message with cost info"
        assert "$0.0000" in cost_logs[0].message, (
            f"Expected '$0.0000' in log, got: {cost_logs[0].message}"
        )
        assert "N/A" not in cost_logs[0].message
