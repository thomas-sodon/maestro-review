"""AI code review for pull requests — posts inline comments via GitHub API."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maestro_review.github import GitHubClient, GitHubError
from maestro_review.markers import (
    _ALL_COMMENT_TERMINAL_MARKERS,
    _DISAGREEMENT_MARKER,
)
from maestro_review.prompt import CODE_REVIEW_PROMPT
from maestro_review.sdk import REVIEW_EFFORT, REVIEW_MODEL, QueryOptions, query

if TYPE_CHECKING:
    from maestro_review.github import CodeHost

logger = logging.getLogger("maestro_review")

# File patterns to skip when truncating large diffs.
_SKIP_PATTERNS = re.compile(
    r"(^package-lock\.json$|^yarn\.lock$|^Pipfile\.lock$|^poetry\.lock$"
    r"|\.min\.js$|\.min\.css$|\.generated\.|\.lock$)",
)

# Words too common to be meaningful for matching findings to issues.
_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "from",
    "not", "but", "are", "was", "has", "have", "can",
})

# Matches GitHub issue references: full URLs or #NNN shorthand.
_ISSUE_LINK_RE = re.compile(
    r"(?:https?://github\.com/[^/]+/[^/]+/issues/\d+|#\d+)",
)

# Filler words stripped from finding titles before fuzzy duplicate detection.
# Drops only meta-vocabulary (articles, prepositions, review filler), never
# domain nouns/verbs, so semantically distinct defects keep disjoint token sets.
# Substantive defect nouns (``error``, ``warning``, ``problem``, ``issue``,
# ``unnecessary``) are intentionally preserved — they are domain-meaningful and
# stripping them degrades short titles like "Missing error handling" below the
# :data:`_FUZZY_MIN_SIGNIFICANT_TOKENS` floor, preventing pairs that should
# match (e.g. "No error handling") from ever fuzzy-matching.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "of", "in", "on", "to", "for",
    "with", "and", "or",
    "detected", "found", "missing", "potential", "possible",
})

# Extracts the first ``**...**`` bold block from a comment body. ``DOTALL`` so
# titles that wrap onto a second line are still captured in one piece.
_THREAD_TITLE_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

# Overlap-coefficient threshold for treating two normalized titles as the
# same finding (|A ∩ B| / min(|A|, |B|) >= threshold).
_FUZZY_THRESHOLD = 0.6

# Minimum number of significant tokens on each side required to attempt a
# fuzzy match — guards against single-token titles like "Leak" swallowing
# unrelated findings like "Memory leak".
_FUZZY_MIN_SIGNIFICANT_TOKENS = 2


def _normalize_title(title: str) -> frozenset[str]:
    """Return the set of significant lowercase tokens in *title*."""
    lowered = title.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return frozenset(
        token
        for token in cleaned.split()
        if len(token) >= 2 and token not in _TITLE_STOPWORDS
    )


# Stable delimiter pair wrapping the overflow-findings list in the review body.
# Overflow findings have no inline thread, so downstream consumers (for example
# the Maestro agent, thomas-sodon/maestro-agent) parse the finding lines out of
# the body between these sentinels.  The pair and the ``of-`` fingerprint scheme
# below are a wire contract those consumers match verbatim, so changing either
# is a breaking change.
_OVERFLOW_SENTINEL_OPEN = "<!-- maestro:overflow -->"
_OVERFLOW_SENTINEL_CLOSE = "<!-- /maestro:overflow -->"

# Header the overflow list is rendered under (see :meth:`_build_review_payload`).
# Predates the sentinel pair; the read-side parser falls back to it so an
# interrupted upgrade that left a body with the header but no sentinels is still
# parsed.  Part of the same wire contract as the sentinel pair.
_OVERFLOW_LEGACY_HEADER = "**Additional findings** (could not be placed inline):"

# Roll-up acknowledgement comment header.  An overflow finding has no inline
# thread to reply to, so a comment-remediation tool records closure as
# ``of-<hash> — <verdict>`` lines under this sentinel on one PR comment.  A
# fingerprint listed there is treated as answered by the reply-coverage gate.
# Part of the same wire contract: downstream consumers match it verbatim.
_OVERFLOW_ACK_SENTINEL = "<!-- maestro:overflow-ack -->"


def _overflow_fingerprint(finding: ReviewFinding) -> str:
    """Return a compact, stable ``of-<hash8>`` handle for an overflow *finding*.

    Derived from ``severity|file|line`` when *line* is a real diff line — every
    one of which the reviewer re-derives deterministically from the diff.  The
    title is deliberately *not* part of that identity: it is LLM-authored prose
    that gets re-worded on nearly every run, so folding it in minted a fresh
    handle for the same logical finding each review and no acknowledgement could
    ever stick.  The legacy title-derived key stays computable via
    :func:`_legacy_overflow_fingerprint` so acks already posted against it keep
    matching.

    When ``line <= 0`` the site alone is *not* an identity: a file-level finding
    carries no line, so every finding of one severity in that file would collapse
    to a single handle — acking one would suppress the rest and the union merge
    would drop them outright.  There the sorted normalized title is folded back
    in (making the handle identical to the legacy key), trading ack stability for
    distinctness, which is the safe direction.

    Raises ``ValueError`` when *severity* or *file* is empty: without them the
    identity is not re-derivable, so the handle would be unackable — fail loudly
    at mint time rather than emit one.  Callers must drop such a finding before
    rendering it (see :meth:`PRReviewer._parse_review_response`).
    """
    if not finding.severity or not finding.file:
        raise ValueError(
            "overflow fingerprint needs a non-empty severity and file; "
            f"got severity={finding.severity!r} file={finding.file!r}",
        )
    if finding.line <= 0:
        return _legacy_overflow_fingerprint(finding)
    raw = f"{finding.severity}|{finding.file}|{finding.line}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"of-{digest}"


def _legacy_overflow_fingerprint(finding: ReviewFinding) -> str:
    """Return the legacy ``severity|file|line|sorted-normalized-title`` handle.

    Retained only so an acknowledgement posted before the identity moved still
    matches the finding it answered; nothing emits this handle any more.
    """
    tokens = " ".join(sorted(_normalize_title(finding.title)))
    raw = f"{finding.severity}|{finding.file}|{finding.line}|{tokens}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"of-{digest}"


def _acking_handles(finding: OverflowFinding) -> set[str]:
    """Return every handle that may legitimately acknowledge *finding*.

    The finding's own embedded handle, the canonical coarse
    ``severity|file|line`` handle, and the legacy title-derived one, all
    recomputed from the parsed block line's own fields — so an ack posted under
    any of those keys still answers the finding.

    The canonical handle is required in its own right: a finding parsed out of a
    block rendered before the move from title-derived to site-derived identity
    carries the *title*-derived handle as its ``fingerprint``, so without it the
    set would be the legacy key twice and an ack recorded under the coarse
    handle would fail to answer a legacy-rendered survivor of
    :meth:`PRReviewer._union_overflow_findings`.

    Same-site collapse is deliberate and *asymmetric with the union*: for
    ``line > 0`` two genuinely-different findings at one ``severity|file|line``
    share a handle, so :meth:`PRReviewer._union_overflow_findings` keeps both as
    separate entries while an ack for either handle answers both.  That is
    intentional — the handle must stay derivable from facts the reviewer
    re-derives identically on every run, and folding the title back in only when
    a sibling finding happens to share the site would make one finding's handle
    depend on whether the *other* was emitted that review, flipping it between
    the coarse and title-derived keys and resurrecting the re-minted-handle
    treadmill the coarse handle exists to end.  So distinct same-site findings
    cannot be acknowledged independently; the union still preserves both
    because dropping one loses it even from an *unacked* audit, which is the
    strictly worse failure.
    """
    probe = ReviewFinding(
        file=finding.file, line=finding.line, severity=finding.severity,
        title=finding.title, description="", fix_plan="",
    )
    return {
        finding.fingerprint,
        _overflow_merge_key(finding),
        _legacy_overflow_fingerprint(probe),
    }


def _overflow_merge_key(finding: OverflowFinding) -> str:
    """Return the canonical cross-block dedupe key for a *parsed* finding.

    The union merge must not key on the raw embedded ``of-<hash>`` handle: a
    block rendered before the move from title-derived to site-derived identity
    embedded the *title*-derived handle, which the reviewer re-minted on every
    reworded review, so one still-open finding carries a different handle in
    each historical block.  Keying on those raw handles would survive the union
    once per legacy review instead of collapsing to a single entry —
    over-reporting exactly the PRs reviewed across the identity move.
    Recomputing the coarse ``severity|file|line`` handle
    canonicalizes every such rendering onto one key.

    A file-level finding (``line <= 0``) has no coarse identity — there the
    handle *is* the title-derived key — so its own fingerprint is the key,
    keeping two distinct file-level findings apart.
    """
    if finding.line <= 0:
        return finding.fingerprint
    return _overflow_fingerprint(ReviewFinding(
        file=finding.file, line=finding.line, severity=finding.severity,
        title=finding.title, description="", fix_plan="",
    ))


# ----------------------------------------------------------------------
# Overflow-findings parsing (read side)
# ----------------------------------------------------------------------

# Parses one rendered overflow line:
#   ``- [of-<hash> ]**<file>:<line>** [<severity>] — <title>[: <description>]``
# The ``of-<hash>`` handle is present in the hardened block and absent in the
# legacy block (recomputed via :func:`_overflow_fingerprint` when missing).
# This reader keeps every severity; a downstream consumer may keep only ``bug``.
_OVERFLOW_LINE_RE = re.compile(
    r"^-\s+"
    r"(?:(of-[0-9a-f]{8})\s+)?"
    r"\*\*(.+?):(-?\d+)\*\*\s+"
    r"\[([a-z]+)\]\s*"
    r"—\s*"
    r"(.+?)"
    # Title/description boundary is the reviewer's ``": "`` (colon + space); a
    # bare ``A:B`` colon stays in the title.  A lone trailing colon (the
    # stripped ``"Title: "`` empty-description form) closes the title.
    r"(?::\s+(.*)|:)?$",
)

# Matches an ``of-<hash>`` handle anywhere in an acknowledgement-comment line.
_OVERFLOW_ACK_LINE_RE = re.compile(r"of-[0-9a-f]{8}")

# Matches an AI reviewer finding root: one of the severity icons emitted by
# :meth:`_format_finding_body` (🐛 bug / 💡 suggestion / ✏️ nit) followed by the
# bold title, across every severity the reviewer posts.  The icons are part of
# the wire contract downstream consumers match verbatim.
_AI_FINDING_RE = re.compile(
    r"^(?:\U0001f41b|\U0001f4a1|✏️)\s+\*\*(.+?)\*\*",
)


@dataclass(frozen=True)
class OverflowFinding:
    """An overflow finding parsed back out of a review body.

    Keeps every severity (not just ``bug``) so the reply-coverage gate can audit
    suggestions and nits too.  ``fingerprint`` is the stable ``of-<hash8>``
    handle — taken from the hardened block, or recomputed from the parsed fields
    for a legacy line.
    """

    fingerprint: str
    file: str
    line: int
    severity: str
    title: str
    description: str


def _extract_overflow_block(body: str) -> str | None:
    """Return the raw overflow-list text from a review *body*, or ``None``.

    Prefers the sentinel-delimited block; falls back to the legacy
    ``**Additional findings**`` header, collecting the bullet lines up to the
    next blank-line section boundary.
    """
    if not body:
        return None
    open_idx = body.find(_OVERFLOW_SENTINEL_OPEN)
    if open_idx != -1:
        close_idx = body.find(_OVERFLOW_SENTINEL_CLOSE, open_idx)
        if close_idx != -1:
            return body[open_idx + len(_OVERFLOW_SENTINEL_OPEN):close_idx]
    header_idx = body.find(_OVERFLOW_LEGACY_HEADER)
    if header_idx == -1:
        return None
    rest = body[header_idx + len(_OVERFLOW_LEGACY_HEADER):]
    chunk = rest.split("\n\n", 1)[0]
    lines = [ln for ln in chunk.splitlines() if ln.strip().startswith("- ")]
    return "\n".join(lines) if lines else None


def _parse_overflow_findings(body: str) -> list[OverflowFinding]:
    """Parse every overflow finding rendered in a review *body*.

    Reads the sentinel-delimited block (legacy-header fallback), parsing each
    bullet across all severities.  Uses the embedded ``of-<hash>`` handle when
    present, otherwise recomputes it via :func:`_overflow_fingerprint`.  A line
    that does not match the rendered shape is skipped without raising.
    """
    block = _extract_overflow_block(body)
    if block is None:
        return []
    findings: list[OverflowFinding] = []
    for line in block.splitlines():
        match = _OVERFLOW_LINE_RE.match(line.strip())
        if match is None:
            continue
        handle, file, line_str, severity, title, description = match.groups()
        title = title.strip()
        line_no = int(line_str)
        fingerprint = handle or _overflow_fingerprint(
            ReviewFinding(
                file=file, line=line_no, severity=severity,
                title=title, description="", fix_plan="",
            ),
        )
        findings.append(OverflowFinding(
            fingerprint=fingerprint,
            file=file,
            line=line_no,
            severity=severity,
            title=title,
            description=(description or "").strip(),
        ))
    return findings


def _parse_overflow_ack(comment_bodies: list[str]) -> set[str]:
    """Return the ``of-<hash>`` handles marked terminal on the roll-up comment.

    Scans *comment_bodies* for the one(s) carrying :data:`_OVERFLOW_ACK_SENTINEL`
    and collects an ``of-<hash>`` handle only from a line that *also* carries one
    of :data:`_ALL_COMMENT_TERMINAL_MARKERS` — a fingerprint whose overflow
    finding has an answering verdict.  A handle that merely co-occurs with the
    sentinel (an in-progress line, a quoted finding, or prose) is *not* treated
    as answered, so the reply-coverage gate cannot pass an overflow finding that
    was never actually closed.
    """
    acked: set[str] = set()
    for body in comment_bodies:
        if _OVERFLOW_ACK_SENTINEL not in body:
            continue
        for line in body.splitlines():
            if not any(marker in line for marker in _ALL_COMMENT_TERMINAL_MARKERS):
                continue
            # Collect only the FIRST handle on the line — the one the roll-up keyed
            # the verdict to.  Collecting every handle would also ack a handle
            # merely quoted in the verdict text (e.g.
            # ``of-aaaaaaaa — Fixed automatically (was a dup of of-bbbbbbbb)``),
            # treating a never-closed finding as answered — the unsafe direction.
            acked.update(_OVERFLOW_ACK_LINE_RE.findall(line)[:1])
    return acked


def _extract_thread_title(body: str) -> str | None:
    """Return the first markdown bold block in *body*, or ``None``."""
    match = _THREAD_TITLE_RE.search(body)
    if match is None:
        return None
    return match.group(1).strip() or None


def _fuzzy_title_match(a: frozenset[str], b: frozenset[str]) -> bool:
    """Return True if two normalized title token-sets describe the same finding."""
    if len(a) < _FUZZY_MIN_SIGNIFICANT_TOKENS or len(b) < _FUZZY_MIN_SIGNIFICANT_TOKENS:
        return False
    smaller = min(len(a), len(b))
    return len(a & b) / smaller >= _FUZZY_THRESHOLD


@dataclass
class ReviewFinding:
    """A single issue found during code review."""

    file: str
    line: int
    severity: str  # "bug", "suggestion", "nit"
    title: str
    description: str
    fix_plan: str
    suggested_code: str | None = None


@dataclass
class ReviewResult:
    """Complete review output."""

    findings: list[ReviewFinding] = field(default_factory=list)
    summary: str = ""
    tokens_used: int = 0
    cost_usd: float | None = None
    diff: str = ""


@dataclass
class ReviewThread:
    """A PR review comment thread with resolution state."""

    thread_id: str       # GraphQL node ID for unresolve mutation
    comment_id: int      # REST numeric ID for in_reply_to
    path: str            # File path
    body: str            # Root comment body text
    all_bodies: str      # Concatenated text of all comments in the thread
    reply_bodies: list[str] = field(default_factory=list)  # Individual reply bodies (excludes root)
    is_resolved: bool = False
    is_outdated: bool = False  # True when the commented code has changed


@dataclass
class PostReviewResult:
    """Outcome of posting a review."""

    posted: list[ReviewFinding]     # New findings posted as inline comments
    reopened: list[ReviewFinding]   # Bug findings flagged as persisting on resolved threads
    filtered: list[ReviewFinding]   # Findings skipped (tracked by issues)
    skipped: list[ReviewFinding]    # Findings skipped (existing unresolved threads)
    disputed: list[ReviewFinding] = field(default_factory=list)  # Findings escalated to issues after a disputing reply


@dataclass
class UnrepliedComments:
    """The reviewer's own findings still lacking a reply.

    Produced by :meth:`PRReviewer.find_unreplied_comments`.  ``inline`` holds open
    AI review threads (root matches :data:`_AI_FINDING_RE`, neither resolved nor
    outdated, no reply); ``overflow`` holds overflow findings whose ``of-<hash>``
    fingerprint is absent from the acknowledgement roll-up.  Both lists empty
    means every posted finding has been answered.
    """

    inline: list[ReviewThread] = field(default_factory=list)
    overflow: list[OverflowFinding] = field(default_factory=list)


def _safe_int(value: object, default: int = 0) -> int:
    """Convert *value* to ``int``, returning *default* on failure."""
    try:
        return int(value)  # type: ignore[call-overload, no-any-return]
    except (TypeError, ValueError):
        return default


class PRReviewer:
    """Reviews pull requests and posts inline comments."""

    def __init__(
        self,
        github_repo: str,
        *,
        gh_timeout: int = 30,
        max_diff_chars: int = 80_000,
        max_files: int = 50,
        code_host: CodeHost | None = None,
    ) -> None:
        self._repo = github_repo
        self._max_diff_chars = max_diff_chars
        self._max_files = max_files
        self._host = code_host or GitHubClient(
            github_repo, timeout=gh_timeout,
        )

    # ------------------------------------------------------------------
    # Data fetching (delegate to CodeHost)
    # ------------------------------------------------------------------

    def get_pr_diff(self, pr_number: int) -> str:
        """Fetch the PR diff."""
        return self._host.get_pr_diff(pr_number)

    def get_pr_info(self, pr_number: int) -> dict[str, str]:
        """Fetch PR title and body."""
        return self._host.get_pr_info(pr_number)

    # ------------------------------------------------------------------
    # Issue-aware filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract non-trivial lowercase keywords from *text*."""
        words = re.findall(r"[a-z_][a-z0-9_]*", text.lower())
        return {w for w in words if len(w) >= 3 and w not in _STOP_WORDS}

    @staticmethod
    def _find_matching_issue(
        finding: ReviewFinding,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the first open issue that tracks *finding*, or ``None``."""
        if not issues:
            return None
        finding_keywords = PRReviewer._extract_keywords(finding.title)
        if len(finding_keywords) < 2:
            return None
        file_lower = finding.file.lower()
        if not file_lower:
            return None

        for issue in issues:
            issue_text = (
                ((issue.get("title") or "") + " " + (issue.get("body") or ""))
                .lower()
            )
            # Use word-boundary-aware search so "a.py" does not match "data.py".
            if not re.search(r"(?<![a-z0-9_/\\])" + re.escape(file_lower) + r"(?![a-z0-9_]|\.(?:[a-z0-9]))", issue_text):
                continue
            issue_keywords = PRReviewer._extract_keywords(issue_text)
            if len(finding_keywords & issue_keywords) >= 2:
                return issue
        return None

    @staticmethod
    def _is_finding_tracked(
        finding: ReviewFinding,
        issues: list[dict[str, Any]],
    ) -> bool:
        """Return ``True`` if an open issue already tracks this finding."""
        return PRReviewer._find_matching_issue(finding, issues) is not None

    def _filter_tracked_findings(
        self,
        findings: list[ReviewFinding],
    ) -> list[ReviewFinding]:
        """Remove findings already tracked by open GitHub issues.

        Applies to every severity — once a finding has an `ai-review` issue
        the comment-remediation workflow stops needing the inline reminder.
        """
        issues = self._host.list_issues(state="open")
        if not issues:  # None (API error) or empty list
            return findings

        filtered: list[ReviewFinding] = []
        for finding in findings:
            if self._is_finding_tracked(finding, issues):
                logger.info(
                    "Skipping %s '%s' in %s — tracked by open issue",
                    finding.severity,
                    finding.title,
                    finding.file,
                )
            else:
                filtered.append(finding)
        return filtered

    # ------------------------------------------------------------------
    # Thread-aware deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_review_threads(
        raw_nodes: list[dict[str, Any]],
    ) -> list[ReviewThread]:
        """Convert raw GraphQL thread nodes into :class:`ReviewThread` objects."""
        threads: list[ReviewThread] = []
        for node in raw_nodes:
            comments_obj = node.get("comments") or {}
            comments = comments_obj.get("nodes") or []
            # Filter to dict entries only — GraphQL can return null nodes.
            comments = [c for c in comments if isinstance(c, dict)]
            if not comments:
                continue
            root = comments[0]
            all_bodies = "\n".join(c.get("body") or "" for c in comments)
            reply_bodies = [c.get("body") or "" for c in comments[1:]]
            threads.append(ReviewThread(
                thread_id=node.get("id", ""),
                comment_id=_safe_int(root.get("databaseId")),
                path=root.get("path") or "",
                body=root.get("body") or "",
                all_bodies=all_bodies,
                reply_bodies=reply_bodies,
                is_resolved=bool(node.get("isResolved", False)),
                is_outdated=bool(node.get("isOutdated", False)),
            ))
        return threads

    @staticmethod
    def _match_finding_to_thread(
        finding: ReviewFinding,
        threads: list[ReviewThread],
    ) -> ReviewThread | None:
        """Return the thread that matches *finding*, or ``None``.

        Two passes per same-path thread:

        1. Exact bold-block substring (``**title**``) — preserves the original
           behavior and avoids false positives on short common words.
        2. Normalized token-overlap fallback — absorbs wording drift across
           review runs (e.g. "Unused import" ↔ "Unused import detected"). Both
           sides must yield at least :data:`_FUZZY_MIN_SIGNIFICANT_TOKENS`
           tokens after stop-word removal, so single-token titles never match.
        """
        title_lower = finding.title.lower()
        if not title_lower:
            return None
        same_path = [t for t in threads if t.path == finding.file]
        for thread in same_path:
            body_lower = (thread.body or "").lower()
            if f"**{title_lower}**" in body_lower:
                return thread
        finding_tokens = _normalize_title(finding.title)
        if len(finding_tokens) < _FUZZY_MIN_SIGNIFICANT_TOKENS:
            return None
        for thread in same_path:
            thread_title = _extract_thread_title(thread.body or "")
            if thread_title is None:
                continue
            if _fuzzy_title_match(finding_tokens, _normalize_title(thread_title)):
                return thread
        return None

    @staticmethod
    def _thread_has_issue_link(thread: ReviewThread) -> bool:
        """Return ``True`` if any comment in *thread* references a GitHub issue."""
        return bool(_ISSUE_LINK_RE.search(thread.all_bodies or ""))

    def _categorize_by_thread_state(
        self,
        pr_number: int,
        findings: list[ReviewFinding],
        threads: list[ReviewThread] | None = None,
    ) -> tuple[
        list[ReviewFinding],
        list[ReviewFinding],
        list[tuple[ReviewFinding, ReviewThread]],
    ]:
        """Split *findings* into ``(new, skip, reopen)`` based on thread state.

        * **new** — no matching thread; post as new inline comment.
        * **skip** — matching thread is unresolved, or resolved with an issue link.
        * **reopen** — matching thread is resolved without an issue link
          (defect persists after a code-fix attempt).

        If *threads* is ``None`` the helper fetches them itself; callers that
        already hold a parsed list (e.g. :meth:`post_review`) pass it in to
        avoid a duplicate GraphQL round-trip.
        """
        if threads is None:
            raw = self._host.get_pr_review_threads(pr_number)
            if not raw:
                return (findings, [], [])
            threads = self._parse_review_threads(raw)
        elif not threads:
            return (findings, [], [])
        new: list[ReviewFinding] = []
        skip: list[ReviewFinding] = []
        reopen: list[tuple[ReviewFinding, ReviewThread]] = []

        for finding in findings:
            thread = self._match_finding_to_thread(finding, threads)
            if thread is None:
                new.append(finding)
            elif not thread.is_resolved:
                logger.info(
                    "Skipping finding '%s' in %s — already commented (unresolved)",
                    finding.title, finding.file,
                )
                skip.append(finding)
            elif self._thread_has_issue_link(thread):
                logger.info(
                    "Skipping finding '%s' in %s — resolved with issue link",
                    finding.title, finding.file,
                )
                skip.append(finding)
            elif finding.severity != "bug":
                # Nits/suggestions in resolved threads without an issue link
                # are skipped — only bug-severity findings get a reply on a
                # resolved thread flagging that the defect persists.
                logger.info(
                    "Skipping %s '%s' in %s — resolved thread (not a bug)",
                    finding.severity, finding.title, finding.file,
                )
                skip.append(finding)
            else:
                reopen.append((finding, thread))

        return (new, skip, reopen)

    # ------------------------------------------------------------------
    # Diff processing
    # ------------------------------------------------------------------

    def _truncate_diff(self, diff: str) -> str:
        """Truncate a diff to stay within token budget.

        Splits on ``diff --git`` boundaries, removes known noisy files,
        and drops the largest chunks until under *max_diff_chars*.
        """
        if not diff:
            return diff

        # A budget of zero or less means no diff content should be returned.
        if self._max_diff_chars <= 0:
            raw_parts = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
            count = sum(1 for p in raw_parts if p.strip())
            return f"\n[{count} file(s) omitted due to size limits]\n"

        # Split into per-file chunks (keep the header with each chunk)
        parts = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
        parts = [p for p in parts if p.strip()]

        # Filter out noisy files
        filtered: list[str] = []
        for part in parts:
            first_line = part.split("\n", 1)[0]
            # Extract filename from "diff --git a/path b/path"
            match = re.search(r" b/(.+)$", first_line)
            if match and _SKIP_PATTERNS.search(match.group(1)):
                continue
            filtered.append(part)

        # Limit file count
        omitted = 0
        if len(filtered) > self._max_files:
            omitted = len(filtered) - self._max_files
            filtered = filtered[: self._max_files]

        # Shrink by removing largest chunks until under limit
        while sum(len(p) for p in filtered) > self._max_diff_chars and len(filtered) > 1:
            # Remove the largest chunk
            largest_idx = max(range(len(filtered)), key=lambda i: len(filtered[i]))
            filtered.pop(largest_idx)
            omitted += 1

        # If a single remaining file still exceeds the budget, hard-truncate it.
        if len(filtered) == 1 and len(filtered[0]) > self._max_diff_chars:
            filtered[0] = filtered[0][:self._max_diff_chars] + "\n[truncated]\n"

        result = "".join(filtered)
        if omitted:
            result += f"\n[{omitted} file(s) omitted due to size limits]\n"
        return result

    @staticmethod
    def _parse_diff_lines(diff: str) -> dict[str, set[int]]:
        """Parse a unified diff to find commentable line numbers per file.

        Returns a mapping of file path to the set of line numbers (in the
        new version of the file) that appear in diff hunks — these are the
        lines GitHub allows inline review comments on.
        """
        result: dict[str, set[int]] = {}
        current_file: str | None = None
        new_line = 0

        for raw_line in diff.splitlines():
            # Detect file header: "diff --git a/old b/new" or "+++ b/path"
            if raw_line.startswith("+++ "):
                if raw_line.startswith("+++ b/"):
                    current_file = raw_line[6:]
                    result.setdefault(current_file, set())
                else:
                    # Deleted file (+++ /dev/null) — clear current_file so
                    # subsequent hunk lines are not attributed to a wrong file.
                    current_file = None
                continue

            # Hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
            if hunk_match:
                new_line = int(hunk_match.group(1))
                continue

            # Skip inter-file diff metadata that appears between hunks
            if raw_line.startswith(("diff ", "index ", "old mode", "new mode",
                                     "new file mode", "deleted file mode",
                                     "similarity index", "rename from",
                                     "rename to", "Binary files",
                                     "\\ No newline")):
                continue

            if current_file is None:
                continue

            if raw_line.startswith("+"):
                # Added line — commentable
                result[current_file].add(new_line)
                new_line += 1
            elif raw_line.startswith("-"):
                # Deleted line — doesn't increment new_line
                pass
            else:
                # Context line — also commentable, increments new_line
                result[current_file].add(new_line)
                new_line += 1

        return result

    # ------------------------------------------------------------------
    # Review execution
    # ------------------------------------------------------------------

    async def review(self, pr_number: int) -> ReviewResult:
        """Run the AI code review and return structured findings."""
        diff = self.get_pr_diff(pr_number)
        if not diff.strip():
            return ReviewResult(summary="No changes to review.")

        info = self.get_pr_info(pr_number)
        truncated_diff = self._truncate_diff(diff)

        prompt = CODE_REVIEW_PROMPT.format(
            pr_title=info["title"],
            pr_body=info["body"] or "(no description)",
            diff=truncated_diff,
        )

        sdk_result = await query(
            prompt=prompt,
            options=QueryOptions(
                allowed_tools=["Read", "Glob", "Grep"],
                model=REVIEW_MODEL,
                effort=REVIEW_EFFORT,
            ),
        )

        # Parse structured response
        try:
            data = sdk_result.parse_json()
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse review JSON, using raw response")
            return ReviewResult(
                summary=sdk_result.response[:500],
                tokens_used=sdk_result.usage.total_tokens,
                cost_usd=sdk_result.usage.total_cost_usd,
                diff=diff,
            )

        findings: list[ReviewFinding] = []
        if not isinstance(data, dict):
            # Model returned a non-object (e.g., a bare list).  Wrap it so
            # the downstream code can proceed.
            data = {"findings": data if isinstance(data, list) else [], "summary": ""}
        for item in data.get("findings") or []:
            if not isinstance(item, dict):
                continue
            try:
                line_no = int(item.get("line", 0))
            except (TypeError, ValueError):
                line_no = 0
            if line_no < 0:
                line_no = 0
            raw_severity = item.get("severity") or "suggestion"
            if raw_severity not in ("bug", "suggestion", "nit"):
                raw_severity = "suggestion"
            raw_file = item.get("file") or ""
            if not raw_file:
                # A finding with no file can never be placed inline, so it would
                # route to overflow — where its identity is not re-derivable and
                # `_overflow_fingerprint` raises, losing *every* finding in the
                # review.  Drop just this one, loudly.
                logger.warning(
                    "Dropping review finding with no file path: %r",
                    item.get("title") or "",
                )
                continue
            findings.append(ReviewFinding(
                file=raw_file,
                line=line_no,
                severity=raw_severity,
                title=item.get("title") or "",
                description=item.get("description") or "",
                fix_plan=item.get("fix_plan") or "",
                suggested_code=item.get("suggested_code"),
            ))

        return ReviewResult(
            findings=findings,
            summary=data.get("summary", ""),
            tokens_used=sdk_result.usage.total_tokens,
            cost_usd=sdk_result.usage.total_cost_usd,
            diff=diff,
        )

    # ------------------------------------------------------------------
    # Review posting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_finding_body(finding: ReviewFinding) -> str:
        """Format a single finding as a GitHub review comment body."""
        icon = {"bug": "\U0001f41b", "suggestion": "\U0001f4a1", "nit": "\u270f\ufe0f"}.get(
            finding.severity, "\U0001f4a1",
        )
        parts = [f"{icon} **{finding.title}**"]
        if finding.description:
            parts.extend(["", finding.description])

        if finding.suggested_code and finding.suggested_code.strip():
            parts.extend(["", "```suggestion", finding.suggested_code, "```"])
        elif finding.fix_plan:
            parts.extend(["", f"**Fix:** {finding.fix_plan}"])

        return "\n".join(parts)

    def _build_review_payload(
        self,
        findings: list[ReviewFinding],
        valid_lines: dict[str, set[int]],
        summary: str,
        filtered: list[ReviewFinding] | None = None,
    ) -> dict[str, Any]:
        """Build the GitHub PR Review API payload.

        Findings on valid diff lines become inline comments.
        Findings on invalid lines are appended to the review body.
        If *filtered* is non-empty, a note listing issue-tracked findings
        is appended to the review body.
        """
        comments: list[dict[str, Any]] = []
        overflow: list[str] = []

        for finding in findings:
            file_lines = valid_lines.get(finding.file, set())
            if finding.line > 0 and finding.line in file_lines:
                comments.append({
                    "path": finding.file,
                    "line": finding.line,
                    "body": self._format_finding_body(finding),
                })
            else:
                overflow.append(
                    f"- {_overflow_fingerprint(finding)} "
                    f"**{finding.file}:{finding.line}** [{finding.severity}] "
                    f"— {finding.title}: {finding.description}"
                )

        body_parts = [summary] if summary else []
        if overflow:
            body_parts.append(
                "\n\n"
                + _OVERFLOW_LEGACY_HEADER
                + "\n"
                + _OVERFLOW_SENTINEL_OPEN
                + "\n"
                + "\n".join(overflow)
                + "\n"
                + _OVERFLOW_SENTINEL_CLOSE,
            )
        if filtered:
            items = "\n".join(
                f"- **{f.file}:{f.line}** — {f.title}"
                for f in filtered
            )
            body_parts.append(
                f"\n\n**Findings already covered by open issues** "
                f"({len(filtered)} skipped):\n{items}",
            )
        body_parts.append(
            "\n\n---\n*AI-generated review by maestro-review — new findings "
            "fail this check*"
        )

        return {
            "event": "COMMENT",
            "body": "\n".join(body_parts).lstrip("\n"),
            "comments": comments,
        }

    # ------------------------------------------------------------------
    # Disputed finding → issue escalation
    # ------------------------------------------------------------------

    # Markers ``_thread_disputes_finding`` looks for in the last reply of a
    # review thread.  The active marker is imported from ``maestro_review.markers``
    # so the two sites cannot drift; the legacy ``bug`` phrasing covers threads
    # closed under the older wording before the marker was renamed.
    _DISAGREEMENT_MARKERS: tuple[str, ...] = (
        _DISAGREEMENT_MARKER,
        "Could not prove this bug",
    )

    @classmethod
    def _thread_disputes_finding(cls, thread: ReviewThread) -> bool:
        """Return ``True`` if the thread's last reply is a "Could not prove" verdict."""
        if not thread.reply_bodies:
            return False
        last = thread.reply_bodies[-1]
        return any(marker in last for marker in cls._DISAGREEMENT_MARKERS)

    def _create_issue_for_disputed_finding(
        self,
        pr_number: int,
        finding: ReviewFinding,
    ) -> int | None:
        """Create a GitHub issue for a finding a thread reply disputed."""
        title = f"AI Review: {finding.title}"
        body = (
            f"## Review Finding (PR #{pr_number})\n\n"
            f"- **File:** `{finding.file}`\n"
            f"- **Line:** {finding.line}\n"
            f"- **Severity:** {finding.severity}\n"
            f"- **Description:** {finding.description}\n\n"
        )
        if finding.fix_plan:
            body += f"**Fix plan:** {finding.fix_plan}\n\n"
        if finding.suggested_code:
            body += f"**Suggested code:**\n```\n{finding.suggested_code}\n```\n\n"
        body += (
            "*Escalated by maestro-review: a reply on the review thread "
            "disputed this finding, and a later review raised it again.*"
        )
        return self._host.create_issue(
            title, body, labels=["ai-review", finding.severity],
        )

    def _handle_disagreements(
        self,
        pr_number: int,
        findings: list[ReviewFinding],
        threads: list[ReviewThread] | None = None,
    ) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
        """Escalate findings a thread reply previously disputed.

        For each finding whose matching thread ends with a "Could not prove"
        reply (and has no existing issue link), file an ``ai-review`` issue
        and reply ``Tracked in #N``.  Returns ``(remaining, escalated)`` —
        escalated findings are removed from the in-flight list so they are
        not re-posted as new inline comments.

        If *threads* is ``None`` the helper fetches them itself; callers that
        already hold a parsed list (e.g. :meth:`post_review`) pass it in to
        avoid a duplicate GraphQL round-trip.
        """
        if not findings:
            return findings, []

        if threads is None:
            raw = self._host.get_pr_review_threads(pr_number)
            if not raw:
                return findings, []
            threads = self._parse_review_threads(raw)
        elif not threads:
            return findings, []

        issues = self._host.list_issues(state="open") or []
        escalated: list[ReviewFinding] = []
        remaining: list[ReviewFinding] = []

        for finding in findings:
            thread = self._match_finding_to_thread(finding, threads)
            if (
                thread is None
                or thread.is_resolved
                or thread.is_outdated
                or not self._thread_disputes_finding(thread)
                or self._thread_has_issue_link(thread)
            ):
                remaining.append(finding)
                continue

            # Reuse an existing matching issue when available.
            matching_issue = self._find_matching_issue(finding, issues)
            issue_num: int | None = None
            if matching_issue is not None:
                num = matching_issue.get("number")
                if num is not None:
                    issue_num = int(num)
                    logger.info(
                        "Linking disputed finding '%s' to existing issue #%d",
                        finding.title, issue_num,
                    )
            if issue_num is None:
                issue_num = self._create_issue_for_disputed_finding(
                    pr_number, finding,
                )
                if issue_num is None:
                    logger.warning(
                        "Failed to create issue for disputed finding '%s' — "
                        "leaving in-flight",
                        finding.title,
                    )
                    remaining.append(finding)
                    continue
                logger.info(
                    "Created issue #%d for disputed finding '%s'",
                    issue_num, finding.title,
                )

            reply_body = (
                f"Tracked in #{issue_num} \u2014 this finding was disputed on the "
                f"thread and raised again on a later review, so it is now "
                f"tracked as an issue."
            )
            try:
                self._host.reply_to_review_comment(
                    pr_number, thread.comment_id, reply_body,
                )
            except Exception:
                # The issue was already created/reused above, so the finding is
                # now tracked and must be removed from the in-flight list. A
                # transient reply failure here (network blip, GraphQL 5xx) must
                # not propagate out of post_review and drop the remaining
                # findings. Skip the in-memory mirror: the thread
                # was not actually annotated on GitHub, so the next run may file
                # a duplicate, which is preferable to aborting the whole pass.
                logger.warning(
                    "Failed to annotate thread for disputed finding '%s' with "
                    "Tracked in #%d — issue created but thread not linked; a "
                    "duplicate may be filed on the next run",
                    finding.title, issue_num, exc_info=True,
                )
                escalated.append(finding)
                continue
            # Mirror the just-posted reply into the in-memory thread state so
            # that any subsequent finding matching the same thread short-circuits
            # via `_thread_has_issue_link` instead of creating a duplicate
            # issue and reply (#stale-issue-link guard).
            thread.reply_bodies.append(reply_body)
            thread.all_bodies = (
                f"{thread.all_bodies}\n{reply_body}" if thread.all_bodies
                else reply_body
            )
            escalated.append(finding)

        return remaining, escalated

    # ------------------------------------------------------------------
    # Review posting
    # ------------------------------------------------------------------

    def post_review(
        self,
        pr_number: int,
        findings: list[ReviewFinding],
        summary: str,
        diff: str | None = None,
    ) -> PostReviewResult:
        """Post an inline review on the PR.

        If *diff* is provided it is used to validate line numbers;
        otherwise all findings are posted as-is.

        Returns a :class:`PostReviewResult` describing what was posted,
        flagged as persisting on a resolved thread, filtered (tracked by
        issues), and skipped (existing threads).
        """
        valid_lines: dict[str, set[int]] = (
            self._parse_diff_lines(diff) if diff else {}
        )

        # When no diff was provided at all (None), treat all lines as valid.
        # But if a diff WAS provided (even empty string) and parsed to empty
        # (e.g., binary diff), leave valid_lines empty so findings go to overflow.
        if diff is None and not valid_lines and findings:
            for f in findings:
                valid_lines.setdefault(f.file, set()).add(f.line)

        # Fetch and parse PR review threads once — both the disagreement
        # handler and the thread-state categorizer consume them.
        raw_threads = self._host.get_pr_review_threads(pr_number)
        threads = self._parse_review_threads(raw_threads) if raw_threads else []

        # Escalate findings a thread reply previously disputed BEFORE filtering
        # by issue tracking.  If a finding's matching thread already has a
        # "Could not prove" reply *and* a separate ai-review issue exists, the
        # filter would otherwise drop it silently and leave the disputed
        # thread permanently in its "Could not prove" state.  Running
        # escalation first lets the reviewer post `Tracked in #N` on the
        # thread (reusing the existing issue) before the filter removes it.
        findings, disputed = self._handle_disagreements(
            pr_number, findings, threads=threads,
        )

        # Filter out remaining findings already tracked by open GitHub issues.
        post_disagreement = findings
        findings = self._filter_tracked_findings(findings)
        filtered = [f for f in post_disagreement if f not in findings]

        if not findings:
            logger.info(
                "No findings left to post — %d filtered, %d escalated",
                len(filtered), len(disputed),
            )
            return PostReviewResult(
                posted=[], reopened=[], filtered=filtered, skipped=[],
                disputed=disputed,
            )

        # Categorize remaining findings by review thread state
        new_findings, skip, reopen = self._categorize_by_thread_state(
            pr_number, findings, threads=threads,
        )

        # Reply for resolved threads where the defect persists
        reopened_findings = [f for f, _ in reopen]
        for finding, thread in reopen:
            logger.info(
                "Replying to thread for '%s' in %s — defect persists",
                finding.title, finding.file,
            )
            reply_body = (
                "This finding was previously resolved, but the defect "
                "persists in the current PR:\n\n"
                + self._format_finding_body(finding)
            )
            self._host.reply_to_review_comment(
                pr_number, thread.comment_id, reply_body,
            )

        if not new_findings:
            logger.info("No new findings to post")
            return PostReviewResult(
                posted=[], reopened=reopened_findings,
                filtered=filtered, skipped=skip,
                disputed=disputed,
            )

        payload = self._build_review_payload(
            new_findings, valid_lines, summary, filtered=filtered,
        )

        try:
            self._host.post_pr_review(pr_number, payload)
        except GitHubError as exc:
            logger.warning("Failed to post review: %s", exc)
            raise

        return PostReviewResult(
            posted=new_findings, reopened=reopened_findings,
            filtered=filtered, skipped=skip,
            disputed=disputed,
        )

    # ------------------------------------------------------------------
    # Reply-coverage audit (read-only, makes no model call)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_unreplied_ai_thread(thread: ReviewThread) -> bool:
        """Return ``True`` if *thread* is an open AI finding awaiting a reply.

        An inline thread counts as unreplied iff its root matches
        :data:`_AI_FINDING_RE` (an icon + bold title the reviewer emits), it is
        neither resolved nor outdated (both exempt: a comment-remediation run
        does not act on them either), and it carries no reply
        (:attr:`ReviewThread.reply_bodies` is empty).
        """
        return (
            not thread.is_resolved
            and not thread.is_outdated
            and not thread.reply_bodies
            and _AI_FINDING_RE.match(thread.body or "") is not None
        )

    def _union_overflow_findings(self, pr_number: int) -> list[OverflowFinding]:
        """Return the union of overflow findings across *every* review body.

        The reviewer emits a non-cumulative, per-commit block, so a finding that
        appeared only in a superseded middle review is still open — reading only
        the newest block silently dropped it from selection.  Walks
        every body carrying a block oldest-first, the newest occurrence winning
        on title/description so the displayed text tracks the latest review.
        Returns ``[]`` when no review carries a block.

        Deduping is *cross-body only*: since the handle is now the coarse
        ``severity|file|line`` key, two genuinely-different findings the
        reviewer emitted at one out-of-diff site share a fingerprint, so
        collapsing by fingerprint alone would silently drop one of them.  Within
        a single block, repeated occurrences of a handle are therefore distinct
        findings and all are kept; across blocks the same handle is the same
        logical finding re-rendered.  The block carrying the most occurrences of
        a handle wins (the newest on a tie), which keeps every same-site finding
        while a title re-wording across reviews still collapses to one.

        The dedupe key is the *canonical* one from :func:`_overflow_merge_key`,
        not the raw embedded handle, so a block carrying a legacy title-derived
        handle — re-minted each review — collapses onto the current coarse
        handle rather than surviving once per legacy review.
        """
        merged: dict[str, list[OverflowFinding]] = {}
        for body in self._host.get_pr_review_bodies(pr_number):
            if _extract_overflow_block(body) is None:
                continue
            per_body: dict[str, list[OverflowFinding]] = {}
            for finding in _parse_overflow_findings(body):
                per_body.setdefault(
                    _overflow_merge_key(finding), [],
                ).append(finding)
            for key, occurrences in per_body.items():
                if len(occurrences) >= len(merged.get(key, ())):
                    merged[key] = occurrences
        # Sorted by site rather than left in oldest-block-first insertion order,
        # so the audit's output is stable across runs regardless of how many
        # reviews happened to carry the block.
        return sorted(
            (f for occurrences in merged.values() for f in occurrences),
            key=lambda f: (f.file, f.line, f.fingerprint, f.title),
        )

    def find_unreplied_comments(self, pr_number: int) -> UnrepliedComments:
        """Audit which of the reviewer's own findings still lack a reply.

        Read-only: inspects the PR's current comment state and reports every
        finding the reviewer is responsible for — inline thread or overflow —
        that is still open and unanswered.  Makes no model call and posts
        nothing.

        * **Inline** — AI review threads accepted by
          :meth:`_is_unreplied_ai_thread`.
        * **Overflow** — the union of findings across every review body carrying
          an overflow block (:meth:`_union_overflow_findings`) whose
          ``of-<hash>`` fingerprint — current or legacy, per
          :func:`_acking_handles` — is absent from the acknowledgement roll-up
          (:func:`_parse_overflow_ack` over the PR's comment bodies).

        The host's read methods are best-effort and map every fetch failure to
        an empty result, so an unreachable ``gh`` would otherwise look like "all
        comments replied".  :meth:`CodeHost.check_connectivity` is probed first
        and raises :class:`~maestro_review.github.GitHubError` on a genuine
        outage, letting the gate fail closed rather than pass falsely.
        """
        self._host.check_connectivity(pr_number)
        raw_threads = self._host.get_pr_review_threads(pr_number)
        threads = self._parse_review_threads(raw_threads) if raw_threads else []
        inline = [t for t in threads if self._is_unreplied_ai_thread(t)]

        acked = _parse_overflow_ack(self._host.get_pr_comment_bodies(pr_number))
        overflow = [
            f for f in self._union_overflow_findings(pr_number)
            if not _acking_handles(f) & acked
        ]

        return UnrepliedComments(inline=inline, overflow=overflow)
