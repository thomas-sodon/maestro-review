"""Shared marker phrases for the comment-remediation workflow.

These are the terminal reply phrases posted on AI review threads by the
two comment-handling surfaces — the human-paced comment-remediation
surface and Maestro's autonomous remediation path.  ``review.py``
imports from here so the reviewer's
detection cannot drift from the posted wording; the core loop may not
import ``maestro_review`` (strict package separation), so
``maestro/evolution.py::_BUG_COMMENT_TERMINAL_MARKERS`` duplicates the
phrases and tests pin the two sides together.  This module deliberately
imports nothing from the package so it cannot be drawn into an import
cycle.
"""

from __future__ import annotations

# Marker phrase posted when a comment-remediation run cannot prove a
# defect.  Imported by ``review.py`` so
# ``PRReviewer._thread_disputes_finding`` can detect disputed findings
# and escalate them to an issue.
_DISAGREEMENT_MARKER = "Could not prove this defect"

# Marker phrase the human-paced comment-remediation surface posts when
# it fixes a bug locally.  A terminal marker: the autonomous path
# recognises it via ``_BUG_COMMENT_TERMINAL_MARKERS`` so an already-fixed
# thread is never re-examined.  Kept here alongside ``_DISAGREEMENT_MARKER``
# so the posted wording and the reviewer's detection cannot drift.
_FIXED_LOCALLY_MARKER = "Fixed locally with implementation-plan context."

# Every terminal verdict phrase a remediation surface can post — bug verdicts
# plus the suggestion / nit / human verdicts.  A line carrying one closes the
# thread (or, on the overflow roll-up, the ``of-<hash>`` finding on that line).
# ``review.py::_parse_overflow_ack`` requires one of these on a handle's line so
# its "acknowledged" set matches the core consumer
# ``maestro/evolution.py::_acknowledged_overflow_fingerprints`` rather than
# counting any handle that merely co-occurs with the roll-up sentinel.  Mirrors
# ``maestro/evolution.py::_ALL_COMMENT_TERMINAL_MARKERS`` verbatim (strict
# package separation forbids importing it); the cross-package test pins them
# together.
_ALL_COMMENT_TERMINAL_MARKERS: tuple[str, ...] = (
    _DISAGREEMENT_MARKER,
    "Could not prove this bug",
    "Fixed automatically",
    _FIXED_LOCALLY_MARKER,
    "Tracked in #",
    "Declined with implementation-plan context.",
    "Answered with implementation-plan context.",
)
