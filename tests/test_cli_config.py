"""Config-file and environment-variable defaults for the CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maestro_review.cli import _load_review_defaults


def _empty_pkg_parent(tmp_path: Path) -> str:
    return str(tmp_path / "nonexistent" / "maestro_review" / "cli.py")


class TestConfigFiles:
    def test_explicit_config_env_var_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        explicit = tmp_path / "custom.toml"
        explicit.write_text("[review]\nmax_files = 7\n")
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / "maestro-review.toml").write_text("[review]\nmax_files = 99\n")
        monkeypatch.setenv("MAESTRO_REVIEW_CONFIG", str(explicit))
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result["max_files"] == 7

    def test_maestro_review_toml_before_maestro_toml(self, tmp_path: Path) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / "maestro-review.toml").write_text("[review]\ngh_timeout = 11\n")
        (cwd / "maestro.toml").write_text("[review]\ngh_timeout = 22\n")
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result["gh_timeout"] == 11

    def test_missing_explicit_config_falls_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAESTRO_REVIEW_CONFIG", str(tmp_path / "absent.toml"))
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result == {}


class TestEnvOverrides:
    def test_env_overrides_file_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        (cwd / "maestro-review.toml").write_text("[review]\nmax_diff_chars = 1000\nmax_files = 5\n")
        monkeypatch.setenv("MAESTRO_REVIEW_MAX_DIFF_CHARS", " 2000 ")
        monkeypatch.setenv("MAESTRO_REVIEW_REQUIRE_COMMENT_REPLIES", "1")
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result["max_diff_chars"] == 2000
        assert result["max_files"] == 5
        assert result["require_comment_replies"] == 1

    def test_invalid_and_blank_env_values_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.setenv("MAESTRO_REVIEW_MAX_FILES", "lots")
        monkeypatch.setenv("MAESTRO_REVIEW_GH_TIMEOUT", "   ")
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result == {}

    def test_env_only_without_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.setenv("MAESTRO_REVIEW_GH_TIMEOUT", "45")
        with (
            patch("maestro_review.cli.Path.cwd", return_value=cwd),
            patch("maestro_review.cli.__file__", _empty_pkg_parent(tmp_path)),
        ):
            result = _load_review_defaults()
        assert result == {"gh_timeout": 45}
