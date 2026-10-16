"""Wrapper around claude_agent_sdk.query for the code review agent.

Slim copy of the Maestro SDK layer — keeps QueryOptions, QueryResult, and
the streaming query() function but drops session file logging (not needed
in CI).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import query as _query
from claude_agent_sdk.types import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

logger = logging.getLogger("maestro_review")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_BACKOFF_FACTOR = 2.0

# The reviewer's single model/effort setting: every review call uses this pair.
REVIEW_MODEL = "claude-opus-5-5"
# Mirrors the SDK's ClaudeAgentOptions.effort Literal; options reach the SDK
# through a dict[str, Any], so this alias is what lets mypy catch a typo.
Effort = Literal["low", "medium", "high", "xhigh", "max"]

REVIEW_EFFORT: Final[Effort] = "high"


def _patch_sdk_stderr_drain() -> None:
    """Monkey-patch the SDK subprocess transport to drain stderr before raising.

    The SDK's ``SubprocessCLITransport._read_messages`` raises a ``ProcessError``
    immediately when the CLI exits non-zero, with a hardcoded message
    "Check stderr output for details".  But ``close()`` cancels the async stderr
    reader task before the ``stderr`` callback fires, so the actual error is lost.

    This patch replaces the post-exit error block to first let the stderr task
    finish (with a short timeout), then include any captured output in the
    ``ProcessError.stderr`` field.
    """
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (  # type: ignore[import-untyped]
            SubprocessCLITransport,
        )
    except ImportError:  # pragma: no cover — SDK always installed
        return

    from claude_agent_sdk._errors import ProcessError  # type: ignore[import-untyped]

    _original = SubprocessCLITransport._read_messages_impl  # type: ignore[attr-defined]

    async def _patched_read_messages_impl(self: Any) -> Any:
        """Wrap _read_messages_impl to drain stderr before raising ProcessError."""
        try:
            async for msg in _original(self):
                yield msg
        except ProcessError:
            # Give the stderr reader a moment to finish before we lose it.
            # The SDK spawns the stderr reader as a detached ``TaskHandle``
            # (``self._stderr_task``); ``wait()`` lets it drain pipe-buffered
            # bytes through our callback. Uses asyncio.wait_for so a hung
            # reader cannot block the re-raise.
            if self._stderr_task is not None:
                with contextlib.suppress(TimeoutError, Exception):
                    await asyncio.wait_for(
                        self._stderr_task.wait(),
                        timeout=2.0,
                    )
            raise
        except Exception:
            raise

    SubprocessCLITransport._read_messages_impl = _patched_read_messages_impl  # type: ignore[attr-defined,method-assign]


_patch_sdk_stderr_drain()


# Subprocess exit codes that indicate permanent failure — never worth retrying.
# 2    — CLI usage error / invalid arguments
# 126  — command found but not executable
# 127  — command not found
# 137  — SIGKILL (OOM killer or forced termination)
_NON_TRANSIENT_EXIT_CODES: frozenset[int] = frozenset({2, 126, 127, 137})


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception is likely transient and worth retrying."""
    msg = str(exc).lower()
    # Never retry auth-related errors.
    if any(kw in msg for kw in ("invalid api key", "unauthorized", "authentication failed", "permission denied")):
        return False
    # Structured subprocess exit code — prefer over fragile string matching.
    # Permanent failures (usage errors, OOM kills, missing CLI) skip retries.
    exit_code = getattr(exc, "exit_code", None)
    if isinstance(exit_code, int):
        return exit_code not in _NON_TRANSIENT_EXIT_CODES
    # Subprocess crashes / exit code failures (string fallback).
    if "exit code" in msg:
        return True
    # SDK <-> bundled CLI handshake stall: the CLI subprocess started but
    # never replied to a control request before the deadline.  Raised as a
    # bare ``Exception("Control request timeout: <subtype>")`` from
    # claude_agent_sdk._internal.query, so it has no exit_code and isn't a
    # ConnectionError — needs its own string match.
    if "control request timeout" in msg:
        return True
    # Contradictory CLI result: the bundled CLI flagged a result as
    # ``is_error`` while reporting ``subtype: success`` and no ``errors``, so
    # the SDK rewraps the trailing non-zero exit as a bare
    # ``Exception("Claude Code returned an error result: success")``
    # (claude_agent_sdk._internal.query). An error-that-is-also-success is a
    # spurious failure — typically rate-limit-adjacent — so re-run it.
    if "error result: success" in msg:
        return True
    if isinstance(exc, ConnectionError):
        return True
    return isinstance(exc, OSError) and exc.errno in (
        errno.ECONNRESET, errno.EPIPE, errno.ECONNREFUSED,
    )


def _extract_error_detail(exc: Exception) -> str:
    """Build a detailed error string, including subprocess stderr if available."""
    base = f"{type(exc).__name__}: {exc}"
    stderr = getattr(exc, "stderr", None)
    if stderr is None:
        return base
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    if stderr:
        return f"{base}\nstderr: {stderr}"
    return base


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate text for log output, appending total length if shortened."""
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text)} chars total]"


@dataclass
class ToolCall:
    """Record of a tool invocation during a query."""

    name: str
    input: dict[str, Any]
    result: str | None = None
    is_error: bool = False


@dataclass
class QueryUsage:
    """Usage statistics from the query."""

    total_tokens: int = 0
    tool_uses: int = 0
    duration_ms: int = 0
    total_cost_usd: float | None = None
    num_turns: int = 0


@dataclass
class QueryOptions:
    """Vendor-neutral options for an SDK query."""

    allowed_tools: list[str] | None = None
    cwd: str | None = None
    max_turns: int | None = None
    model: str | None = None
    effort: Effort | None = None
    output_format: dict[str, Any] | None = None
    max_budget_usd: float | None = None


@dataclass
class QueryResult:
    """Simplified result from a query call."""

    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: QueryUsage = field(default_factory=QueryUsage)
    rate_limit_events: list[str] = field(default_factory=list)
    structured_output: Any = None

    def parse_json(self) -> Any:
        """Extract and parse JSON from the response.

        Handles responses where the model wraps JSON in markdown code blocks
        (e.g. ```json ... ```) or includes surrounding text.

        Raises ``json.JSONDecodeError`` if no valid JSON is found.
        """
        if self.structured_output is not None:
            return self.structured_output

        text = self.response.strip()

        # Try parsing the raw response first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from a fenced code block.
        m = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass  # Fall through to bracket extraction

        # Last resort: find the outermost { ... } or [ ... ].
        best: tuple[int, Any] | None = None
        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
                span = end - start
                if best is None or span > best[0]:
                    best = (span, parsed)
        if best is not None:
            return best[1]

        raise json.JSONDecodeError("No JSON found in response", text, 0)


def _process_content_blocks(
    blocks: list[Any],
    tool_calls: list[ToolCall],
    tool_call_by_id: dict[str, ToolCall],
) -> None:
    """Process content blocks from assistant or user messages."""
    for block in blocks:
        if isinstance(block, ThinkingBlock):
            logger.debug("SDK thinking: %s", _truncate(block.thinking, 200))
        elif isinstance(block, TextBlock):
            logger.debug("SDK text: %s", _truncate(block.text, 300))
        elif isinstance(block, ToolUseBlock):
            logger.info("SDK tool call: %s", block.name)
            logger.debug(
                "SDK tool input [%s]: %s",
                block.name,
                _truncate(str(block.input), 500),
            )
            tc = ToolCall(name=block.name, input=block.input)
            tool_call_by_id[block.id] = tc
            tool_calls.append(tc)
        elif isinstance(block, ToolResultBlock):
            content_str = str(block.content) if block.content else ""
            is_error = bool(block.is_error)
            matched = tool_call_by_id.get(block.tool_use_id)
            tool_name = matched.name if matched else block.tool_use_id
            logger.info(
                "SDK tool result [%s]: %s%s",
                tool_name,
                "ERROR " if is_error else "",
                _truncate(content_str, 300),
            )
            if matched:
                matched.result = content_str
                matched.is_error = is_error


async def query(
    prompt: str,
    *,
    options: QueryOptions | None = None,
    system: str | None = None,
) -> QueryResult:
    """Run a claude_agent_sdk query and return the final result."""
    sdk_kwargs: dict[str, Any] = {"prompt": prompt}
    sdk_opts: dict[str, Any] = {}
    if system is not None:
        sdk_opts["system_prompt"] = system
    if options is not None:
        if options.allowed_tools is not None:
            sdk_opts["allowed_tools"] = options.allowed_tools
        if options.cwd is not None:
            sdk_opts["cwd"] = options.cwd
        if options.max_turns is not None:
            sdk_opts["max_turns"] = options.max_turns
        if options.model is not None:
            sdk_opts["model"] = options.model
        if options.effort is not None:
            sdk_opts["effort"] = options.effort
        if options.output_format is not None:
            sdk_opts["output_format"] = options.output_format
        if options.max_budget_usd is not None:
            sdk_opts["max_budget_usd"] = options.max_budget_usd

    stderr_lines: list[str] = []

    def _stderr_callback(line: str) -> None:
        stderr_lines.append(line)
        logger.warning("SDK CLI stderr: %s", line)

    sdk_opts["stderr"] = _stderr_callback

    sdk_kwargs["options"] = ClaudeAgentOptions(**sdk_opts)

    last_exc: Exception | None = None
    attempt = 0
    while True:
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (_RETRY_BACKOFF_FACTOR ** (attempt - 1))
            logger.warning(
                "SDK query retry %d/%d after %.1fs: %s",
                attempt, _MAX_RETRIES, delay,
                _extract_error_detail(last_exc) if last_exc else "unknown",
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.warning("Retry sleep interrupted by stale cancel scope, continuing")

        try:
            last: ResultMessage | None = None
            tool_calls: list[ToolCall] = []
            tool_call_by_id: dict[str, ToolCall] = {}
            usage = QueryUsage()
            rate_limits: list[str] = []

            async for message in _query(**sdk_kwargs):
                if isinstance(message, AssistantMessage):
                    _process_content_blocks(
                        message.content, tool_calls, tool_call_by_id,
                    )
                elif isinstance(message, UserMessage):
                    if isinstance(message.content, list):
                        _process_content_blocks(
                            message.content, tool_calls, tool_call_by_id,
                        )
                elif isinstance(message, TaskStartedMessage):
                    logger.debug("SDK task started: %s", message.description)
                elif isinstance(message, TaskProgressMessage):
                    msg_usage = message.usage or {}
                    total_tokens = msg_usage.get("total_tokens", 0) if isinstance(msg_usage, dict) else 0
                    tool_uses = msg_usage.get("tool_uses", 0) if isinstance(msg_usage, dict) else 0
                    duration_ms = msg_usage.get("duration_ms", 0) if isinstance(msg_usage, dict) else 0
                    logger.info(
                        "SDK progress: %d tokens, %d tool uses, %dms%s",
                        total_tokens,
                        tool_uses,
                        duration_ms,
                        f", last tool: {message.last_tool_name}"
                        if message.last_tool_name
                        else "",
                    )
                    usage.total_tokens = total_tokens
                    usage.tool_uses = tool_uses
                    usage.duration_ms = duration_ms
                elif isinstance(message, TaskNotificationMessage):
                    logger.info("SDK task %s: %s", message.status, message.summary)
                elif isinstance(message, RateLimitEvent):
                    info = message.rate_limit_info
                    status = getattr(info, "status", None) if info is not None else None
                    status_str = status if status is not None else (str(info) if info is not None else "unknown")
                    logger.warning("SDK rate limit: %s", status_str)
                    rate_limits.append(status_str)
                elif isinstance(message, StreamEvent):
                    logger.debug("SDK stream event")
                elif isinstance(message, ResultMessage):
                    last = message

            if last is None:
                return QueryResult(
                    response="", tool_calls=tool_calls, usage=usage,
                    rate_limit_events=rate_limits,
                )

            usage.total_cost_usd = last.total_cost_usd
            usage.num_turns = last.num_turns
            usage.duration_ms = last.duration_ms
            if last.usage and isinstance(last.usage, dict):
                usage.total_tokens = last.usage.get("total_tokens", usage.total_tokens)
                usage.tool_uses = last.usage.get("tool_uses", usage.tool_uses)

            cost_str = (
                f"${last.total_cost_usd:.4f}"
                if last.total_cost_usd is not None
                else "N/A"
            )
            logger.info(
                "SDK query complete: %d turns, %dms, cost=%s",
                last.num_turns,
                last.duration_ms,
                cost_str,
            )

            return QueryResult(
                response=last.result or "",
                tool_calls=tool_calls,
                usage=usage,
                rate_limit_events=rate_limits,
                structured_output=last.structured_output,
            )
        except Exception as exc:
            last_exc = exc
            if stderr_lines:
                joined = "\n".join(stderr_lines)
                logger.error(
                    "SDK CLI stderr captured (%d lines):\n%s",
                    len(stderr_lines), joined,
                )
                # Always overwrite: the SDK sets ProcessError.stderr to
                # the useless placeholder "Check stderr output for
                # details" which is truthy but carries no diagnostic
                # value.  Callback-captured lines are always more useful.
                exc.stderr = joined  # type: ignore[attr-defined]
                stderr_lines.clear()
            if not _is_transient(exc) or attempt == _MAX_RETRIES:
                raise
            attempt += 1
