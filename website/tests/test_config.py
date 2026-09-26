"""Config loading: precedence, env overrides, and the shape of the defaults.

These rules decide where the server reads its ports and model paths from, so a
mistake here shows up as a server that starts on the wrong port -- or silently
fails to find the weights.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from server.config import Config, _deep_merge


def write_yaml(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


class TestDefaults:
    def test_the_shipped_config_loads(self) -> None:
        cfg = Config.load()
        assert cfg.server.port == 8000
        assert cfg.ingest.port == 5000
        assert cfg.ingest.sample_rate == 16_000
        assert cfg.ingest.channels == 1
        assert cfg.ingest.bits == 16

    def test_the_audio_format_matches_the_ingest_contract(self, tmp_path: Path) -> None:
        """A mismatch here silently garbles every recording."""
        cfg = Config.load()
        assert cfg.ingest.sample_rate == 16_000
        assert cfg.ingest.bits == 16
        assert cfg.ingest.channels == 1

    def test_chat_format_can_call_tools(self) -> None:
        """Plain chatml makes the model ignore tools and hallucinate instead."""
        assert Config.load().llm.chat_format == "chatml-function-calling"

    def test_retention_is_bounded_by_default(self) -> None:
        retention = Config.load().retention
        assert 0 < retention.max_turns <= 1000
        assert retention.delete_audio is True

    def test_paths_resolve_against_the_project_root(self) -> None:
        cfg = Config.load()
        assert cfg.models_dir.is_absolute()
        assert cfg.data_dir.is_absolute()


class TestLocalOverride:
    def test_merges_a_single_section(self, tmp_path: Path, monkeypatch) -> None:
        base = write_yaml(tmp_path / "config.yaml", """
            ingest:
              port: 5000
              host: 0.0.0.0
            server:
              port: 8000
        """)
        local = write_yaml(tmp_path / "config.local.yaml", """
            ingest:
              port: 9000
        """)
        monkeypatch.setattr("server.config.PROJECT_ROOT", tmp_path)

        cfg = Config.load(base)
        assert cfg.ingest.port == 9000, "the local file must win"
        # The sibling key it did not mention must survive.
        assert cfg.ingest.host == "0.0.0.0"
        assert cfg.server.port == 8000
        assert local.exists()

    def test_replaces_a_whole_section_rather_than_merging_deeply(self, tmp_path: Path) -> None:
        """Deep-merging a list would inherit siblings the editor never saw."""
        base = write_yaml(tmp_path / "config.yaml", """
            stt:
              beam_size: 5
              hotwords: [Pune, Mumbai, Delhi]
        """)
        write_yaml(tmp_path / "config.local.yaml", """
            stt:
              hotwords: [Nashik]
        """)
        original = Path(__file__).resolve().parent.parent
        import server.config as config_module

        old_root = config_module.PROJECT_ROOT
        try:
            config_module.PROJECT_ROOT = tmp_path
            cfg = Config.load(base)
        finally:
            config_module.PROJECT_ROOT = old_root
        assert cfg.stt.hotwords == ["Nashik"]
        assert cfg.stt.beam_size == 5
        assert original.exists()

    def test_a_missing_local_file_is_fine(self, tmp_path: Path) -> None:
        base = write_yaml(tmp_path / "config.yaml", "server:\n  port: 8123\n")
        cfg = Config.load(base)
        assert cfg.server.port == 8123

    def test_invalid_yaml_in_the_local_file_is_ignored_not_fatal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo in a local override must not stop the server booting."""
        base = write_yaml(tmp_path / "config.yaml", "server:\n  port: 8123\n")
        (tmp_path / "config.local.yaml").write_text("ingest:\n  port: [unclosed\n")

        cfg = Config.load(base)
        assert cfg.server.port == 8123
        assert cfg.ingest.port == 5000  # the default, not a crash


class TestEnvOverrides:
    def test_section_and_key(self, monkeypatch) -> None:
        monkeypatch.setenv("FRIDAY_STT__DEVICE", "cpu")
        assert Config.load().stt.device == "cpu"

    def test_beats_the_local_file(self, tmp_path: Path, monkeypatch) -> None:
        base = write_yaml(tmp_path / "config.yaml", "ingest:\n  port: 5000\n")
        write_yaml(tmp_path / "config.local.yaml", "ingest:\n  port: 6000\n")
        monkeypatch.setenv("FRIDAY_INGEST__PORT", "7000")

        old_root = Path(__file__).resolve().parent.parent
        import server.config as config_module

        try:
            config_module.PROJECT_ROOT = tmp_path
            cfg = Config.load(base)
        finally:
            config_module.PROJECT_ROOT = old_root
        assert cfg.ingest.port == 7000

    def test_coerces_to_the_declared_type(self, monkeypatch) -> None:
        monkeypatch.setenv("FRIDAY_SERVER__PORT", "9001")
        monkeypatch.setenv("FRIDAY_SERVER__SERVE_WEB", "false")
        monkeypatch.setenv("FRIDAY_INGEST__VAD_SILENCE_S", "1.5")
        cfg = Config.load()
        assert cfg.server.port == 9001 and isinstance(cfg.server.port, int)
        assert cfg.server.serve_web is False
        assert cfg.ingest.vad_silence_s == 1.5

    def test_unrelated_friday_vars_are_ignored(self, monkeypatch) -> None:
        """`FRIDAY_HOME=/x` has no `__`, so it is not a config path."""
        monkeypatch.setenv("FRIDAY_HOME", "/somewhere")
        monkeypatch.setenv("PATH", "/usr/bin")
        assert Config.load().server.port == 8000

    def test_an_unknown_key_does_not_raise(self, monkeypatch) -> None:
        monkeypatch.setenv("FRIDAY_INGEST__NOT_A_KEY", "1")
        assert Config.load().ingest.port == 5000


class TestDeepMerge:
    def test_merges_one_level_into_sections(self) -> None:
        base = {"ingest": {"port": 1, "host": "a"}, "server": {"port": 2}}
        overlay = {"ingest": {"port": 9}}
        assert _deep_merge(base, overlay) == {
            "ingest": {"port": 9, "host": "a"},
            "server": {"port": 2},
        }

    def test_an_empty_overlay_changes_nothing(self) -> None:
        base = {"ingest": {"port": 1}}
        assert _deep_merge(base, {}) == base

    def test_a_scalar_replaces_a_section(self) -> None:
        assert _deep_merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}

    def test_does_not_mutate_its_arguments(self) -> None:
        base = {"ingest": {"port": 1}}
        _deep_merge(base, {"ingest": {"port": 2}})
        assert base["ingest"]["port"] == 1

    def test_new_sections_are_added(self) -> None:
        assert _deep_merge({}, {"tools": {"a": 1}}) == {"tools": {"a": 1}}
