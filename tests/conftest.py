"""Shared fixtures for the maestro-review test suite."""

from __future__ import annotations

import pytest

from maestro_review.cli import _CONFIG_KEYS, _ENV_PREFIX, CONFIG_ENV_VAR


@pytest.fixture(autouse=True)
def _isolate_review_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's MAESTRO_REVIEW_* environment out of every test."""
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    for key in _CONFIG_KEYS:
        monkeypatch.delenv(_ENV_PREFIX + key.upper(), raising=False)
