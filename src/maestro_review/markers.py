"""Shared marker phrases for the comment-remediation workflow.

These are the terminal reply phrases a comment-remediation tool posts on
AI review threads to record its verdict on a finding.  ``review.py``
imports from here so the reviewer's detection cannot drift from the
posted wording.  The phrases are a wire contract: downstream consumers
(for example the Maestro agent, thomas-sodon/maestro-agent) match them
verbatim, so changing one is a breaking change.  This module
deliberately imports nothing from the package so it cannot be drawn
into an import cycle.
"""

from __future__ import annotations

# Marker phrase posted when a comment-remediation run cannot prove a
# defect.  Imported by ``review.py`` so
# ``PRReviewer._thread_disputes_finding`` can detect disputed findings
# and escalate them to an issue.
_DISAGREEMENT_MARKER = "Could not prove this defect"

# Marker phrase a comment-remediation run posts when it fixes a bug
# locally.  A terminal marker: consumers recognise it so an already-fixed
# thread is never re-examined.  Kept here alongside ``_DISAGREEMENT_MARKER``
# so the posted wording and the reviewer's detection cannot drift.
_FIXED_LOCALLY_MARKER = "Fixed locally with implementation-plan context."

# Every terminal verdict phrase a comment-remediation tool can post — bug verdicts
# plus the suggestion / nit / human verdicts.  A line carrying one closes the
# thread (or, on the overflow roll-up, the ``of-<hash>`` finding on that line).
# ``review.py::_parse_overflow_ack`` requires one of these on a handle's line
# rather than counting any handle that merely co-occurs with the roll-up
# sentinel.  Part of the wire contract described in the module docstring:
# downstream consumers match these phrases verbatim, so changing one is a
# breaking change.
_ALL_COMMENT_TERMINAL_MARKERS: tuple[str, ...] = (
    _DISAGREEMENT_MARKER,
    "Could not prove this bug",
    "Fixed automatically",
    _FIXED_LOCALLY_MARKER,
    "Tracked in #",
    "Declined with implementation-plan context.",
    "Answered with implementation-plan context.",
)
