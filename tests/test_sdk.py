"""Tests for maestro_review.sdk — query(), retry, and rate limit handling."""

from __future__ import annotations

import asyncio
import errno
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import (
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
)

from maestro_review.sdk import QueryOptions, _extract_error_detail, _is_transient, query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_msg(result: str = "done", **overrides: Any) -> ResultMessage:
    """Create a ResultMessage with sensible defaults."""
    defaults: dict[str, Any] = {
        "subtype": "result",
        "duration_ms": 100,
        "duration_api_ms": 80,
        "total_cost_usd": 0.001,
        "num_turns": 1,
        "is_error": False,
        "result": result,
        "session_id": "s1",
    }
    defaults.update(overrides)
    return ResultMessage(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRateLimitHandling:
    """Verify query() handles RateLimitEvent defensively."""

    @pytest.mark.asyncio
    async def test_rate_limit_valid_info(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Happy path: valid rate_limit_info with status attribute."""
        event = RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="allowed_warning"),
            uuid="u1",
            session_id="s1",
        )

        async def gen(**kwargs: Any) -> Any:
            yield event
            yield _result_msg()

        with (
            caplog.at_level(logging.WARNING, logger="maestro_review"),
            patch("maestro_review.sdk._query", side_effect=gen),
        ):
            result = await query(prompt="test")

        assert "SDK rate limit: allowed_warning" in caplog.text
        assert result.rate_limit_events == ["allowed_warning"]

    @pytest.mark.asyncio
    async def test_rate_limit_none_info_no_crash(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """rate_limit_info=None must not crash; falls back to 'unknown'."""
        event = MagicMock(spec=RateLimitEvent)
        event.rate_limit_info = None

        async def gen(**kwargs: Any) -> Any:
            yield event
            yield _result_msg()

        with (
            caplog.at_level(logging.WARNING, logger="maestro_review"),
            patch("maestro_review.sdk._query", side_effect=gen),
        ):
            result = await query(prompt="test")

        assert "SDK rate limit: unknown" in caplog.text
        assert result.rate_limit_events == ["unknown"]

    @pytest.mark.asyncio
    async def test_rate_limit_missing_status_attr(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """rate_limit_info without .status falls back to str(rate_limit_info)."""

        class _NoStatus:
            def __str__(self) -> str:
                return "custom_info_repr"

        event = MagicMock(spec=RateLimitEvent)
        event.rate_limit_info = _NoStatus()

        async def gen(**kwargs: Any) -> Any:
            yield event
            yield _result_msg()

        with (
            caplog.at_level(logging.WARNING, logger="maestro_review"),
            patch("maestro_review.sdk._query", side_effect=gen),
        ):
            result = await query(prompt="test")

        assert "SDK rate limit: custom_info_repr" in caplog.text
        assert result.rate_limit_events == ["custom_info_repr"]


class TestStructuredOutput:
    """Verify structured_output propagation from ResultMessage to QueryResult."""

    @pytest.mark.asyncio
    async def test_structured_output_propagated(self) -> None:
        """structured_output from ResultMessage should appear in QueryResult."""
        data = {"valid": True}

        async def gen(**kwargs: Any) -> Any:
            yield _result_msg("", structured_output=data)

        with patch("maestro_review.sdk._query", side_effect=gen):
            result = await query(prompt="test")

        assert result.structured_output == data


class TestResultUsageGuard:
    """Verify a non-dict truthy ResultMessage.usage degrades gracefully."""

    @pytest.mark.asyncio
    async def test_non_dict_result_usage_degrades_gracefully(self) -> None:
        """A truthy non-dict usage on the terminal ResultMessage must not crash.

        The ResultMessage.usage block mirrors the TaskProgressMessage guard: a
        malformed usage payload (here a scalar) is ignored, so total_tokens and
        tool_uses stay at their pre-ResultMessage QueryUsage defaults instead of
        raising AttributeError on `.get`.
        """

        async def gen(**kwargs: Any) -> Any:
            yield _result_msg("", usage=5)

        with patch("maestro_review.sdk._query", side_effect=gen):
            result = await query(prompt="test")

        assert result.usage.total_tokens == 0
        assert result.usage.tool_uses == 0


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------


class TestIsTransient:
    @pytest.mark.parametrize(
        "exc, expected",
        [
            (RuntimeError("Command failed with exit code 1"), True),
            (Exception("Command failed with exit code 1 (exit code: 1)"), True),
            (ConnectionError("connection reset"), True),
            (BrokenPipeError(), True),
            (Exception("Control request timeout: initialize"), True),
            (Exception("Control request timeout: interrupt"), True),
            (Exception("Claude Code returned an error result: success"), True),
            (RuntimeError("authentication failed exit code 1"), False),
            (RuntimeError("invalid api key exit code 1"), False),
            (RuntimeError("permission denied exit code 1"), False),
            (ValueError("bad json"), False),
            (RuntimeError("something else"), False),
            (TypeError("wrong type"), False),
        ],
        ids=[
            "exit_code_runtime",
            "exit_code_base_exception",
            "connection_error",
            "broken_pipe",
            "control_request_timeout_initialize",
            "control_request_timeout_interrupt",
            "error_result_success_contradiction",
            "authentication_failed_keyword",
            "invalid_api_key_keyword",
            "permission_denied_keyword",
            "value_error",
            "other_runtime",
            "type_error",
        ],
    )
    def test_classification(self, exc: Exception, expected: bool) -> None:
        assert _is_transient(exc) is expected

    def test_oserror_with_connection_errno(self) -> None:
        exc = OSError(errno.ECONNREFUSED, "Connection refused")
        assert _is_transient(exc) is True

    def test_oserror_with_pipe_errno(self) -> None:
        exc = OSError(errno.EPIPE, "Broken pipe")
        assert _is_transient(exc) is True

    def test_oserror_with_unrelated_errno(self) -> None:
        exc = OSError(errno.ENOENT, "No such file")
        assert _is_transient(exc) is False

    def test_structured_exit_code_1_is_transient(self) -> None:
        from claude_agent_sdk._errors import ProcessError

        exc = ProcessError("subprocess died", exit_code=1)
        assert _is_transient(exc) is True

    @pytest.mark.parametrize("code", [2, 126, 127, 137])
    def test_structured_non_transient_exit_codes(self, code: int) -> None:
        from claude_agent_sdk._errors import ProcessError

        exc = ProcessError("cli failed", exit_code=code)
        assert _is_transient(exc) is False


# ---------------------------------------------------------------------------
# _extract_error_detail
# ---------------------------------------------------------------------------


class TestExtractErrorDetail:
    def test_plain_exception(self) -> None:
        exc = ValueError("bad input")
        result = _extract_error_detail(exc)
        assert result == "ValueError: bad input"

    def test_exception_with_stderr_bytes(self) -> None:
        exc = RuntimeError("failed")
        exc.stderr = b"some error output\n"  # type: ignore[attr-defined]
        result = _extract_error_detail(exc)
        assert "RuntimeError: failed" in result
        assert "stderr: some error output" in result

    def test_exception_with_stderr_string(self) -> None:
        exc = RuntimeError("failed")
        exc.stderr = "string stderr"  # type: ignore[attr-defined]
        result = _extract_error_detail(exc)
        assert "stderr: string stderr" in result

    def test_exception_with_empty_stderr(self) -> None:
        exc = RuntimeError("failed")
        exc.stderr = ""  # type: ignore[attr-defined]
        result = _extract_error_detail(exc)
        assert result == "RuntimeError: failed"
        assert "stderr" not in result

    def test_exception_with_none_stderr(self) -> None:
        exc = RuntimeError("failed")
        exc.stderr = None  # type: ignore[attr-defined]
        result = _extract_error_detail(exc)
        assert result == "RuntimeError: failed"


# ---------------------------------------------------------------------------
# query() retry logic
# ---------------------------------------------------------------------------


class TestQueryRetry:
    @pytest.mark.asyncio
    async def test_retry_on_transient_error_succeeds(self) -> None:
        """First call fails with transient error, second succeeds."""
        call_count = 0

        async def gen(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Command failed with exit code 1")
            yield _result_msg("success")

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await query(prompt="test")

        assert result.response == "success"
        assert call_count == 2
        mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self) -> None:
        """All attempts fail — exception propagates after retries."""
        async def gen(**kwargs: Any) -> Any:
            raise RuntimeError("Command failed with exit code 1")
            yield  # make it an async generator  # pragma: no cover

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             pytest.raises(RuntimeError, match="exit code 1"):
            await query(prompt="test")

        # 3 retries = 3 sleep calls with exponential backoff
        assert mock_sleep.await_count == 3
        mock_sleep.assert_any_await(1.0)
        mock_sleep.assert_any_await(2.0)
        mock_sleep.assert_any_await(4.0)

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self) -> None:
        """Non-transient errors propagate immediately without retry."""
        async def gen(**kwargs: Any) -> Any:
            raise ValueError("bad input")
            yield  # pragma: no cover

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             pytest.raises(ValueError, match="bad input"):
            await query(prompt="test")

        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self) -> None:
        """Auth-related RuntimeError not retried despite containing 'exit code'."""
        async def gen(**kwargs: Any) -> Any:
            raise RuntimeError("authentication failed exit code 1")
            yield  # pragma: no cover

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             pytest.raises(RuntimeError, match="authentication"):
            await query(prompt="test")

        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connection_error_is_transient(self) -> None:
        """ConnectionError triggers retry."""
        call_count = 0

        async def gen(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection reset")
            yield _result_msg("recovered")

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await query(prompt="test")

        assert result.response == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_backoff_delays(self) -> None:
        """Verify exact exponential backoff sequence: 1.0, 2.0, 4.0."""
        call_count = 0

        async def gen(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("Command failed with exit code 1")
            yield _result_msg("finally")

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await query(prompt="test")

        assert result.response == "finally"
        assert mock_sleep.await_count == 3
        delays = [call.args[0] for call in mock_sleep.await_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_retry_sleep_cancelled_error_is_handled(self) -> None:
        """CancelledError during retry sleep (stale anyio scope) doesn't crash."""
        call_count = 0

        async def gen(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Command failed with exit code 1")
            yield _result_msg("recovered")

        sleep_count = 0

        async def flaky_sleep(delay: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count == 1:
                raise asyncio.CancelledError(
                    "Cancelled via cancel scope abc; reason: deadline exceeded"
                )

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", AsyncMock(side_effect=flaky_sleep)):
            result = await query(prompt="test")

        assert result.response == "recovered"
        assert call_count == 3


# ---------------------------------------------------------------------------
# stderr capture
# ---------------------------------------------------------------------------


class TestStderrCapture:
    @pytest.mark.asyncio
    async def test_stderr_lines_attached_to_exception(self) -> None:
        """stderr callback lines are joined and attached to the exception."""
        async def gen(**kwargs: Any) -> Any:
            # Invoke the stderr callback before raising
            opts = kwargs.get("options")
            if opts and opts.stderr:
                opts.stderr("error line 1")
                opts.stderr("error line 2")
            raise ValueError("non-transient crash")
            yield  # pragma: no cover

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             pytest.raises(ValueError, match="non-transient crash") as exc_info:
            await query(prompt="test")

        assert hasattr(exc_info.value, "stderr")
        assert "error line 1\nerror line 2" in exc_info.value.stderr  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_stderr_lines_cleared_between_retries(self) -> None:
        """stderr lines from a failed attempt are cleared before the next retry."""
        call_count = 0

        async def gen(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            opts = kwargs.get("options")
            if opts and opts.stderr:
                opts.stderr(f"attempt {call_count}")
            if call_count == 1:
                raise RuntimeError("Command failed with exit code 1")
            yield _result_msg("ok")

        with patch("maestro_review.sdk._query", side_effect=gen), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await query(prompt="test")

        assert result.response == "ok"
        assert call_count == 2


# ---------------------------------------------------------------------------
# options construction
# ---------------------------------------------------------------------------


class TestQueryOptionsConstruction:
    @pytest.mark.asyncio
    async def test_options_always_constructed_with_stderr(self) -> None:
        """ClaudeAgentOptions is built even with no system prompt and no options.

        The stderr callback is wired unconditionally, so ``sdk_opts`` is never
        empty and ``ClaudeAgentOptions(**sdk_opts)`` always runs. This locks the
        always-constructed invariant against a future regression that reintroduces
        an options-skipping path.
        """
        captured: dict[str, Any] = {}

        async def gen(**kwargs: Any) -> Any:
            captured["options"] = kwargs.get("options")
            yield _result_msg("ok")

        with patch("maestro_review.sdk._query", side_effect=gen):
            result = await query(prompt="test")

        assert result.response == "ok"
        options = captured["options"]
        assert isinstance(options, ClaudeAgentOptions)
        assert callable(options.stderr)

    @pytest.mark.asyncio
    async def test_model_and_effort_forwarded_to_sdk_options(self) -> None:
        """QueryOptions.model and .effort reach ClaudeAgentOptions."""
        captured: dict[str, Any] = {}

        async def gen(**kwargs: Any) -> Any:
            captured["options"] = kwargs.get("options")
            yield _result_msg("ok")

        with patch("maestro_review.sdk._query", side_effect=gen):
            await query(
                prompt="t",
                options=QueryOptions(model="claude-opus-5-5", effort="high"),
            )

        options = captured["options"]
        assert isinstance(options, ClaudeAgentOptions)
        assert options.model == "claude-opus-5-5"
        assert options.effort == "high"
