from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kajovochat.settings import AppSettings, build_system_prompt


def test_settings_api_key_roundtrip_and_prompt() -> None:
    settings = AppSettings(language="cs", response_style="věcné", response_length="normální", response_detail="stručná")
    settings.openai_api_key = "sk-test-123"

    assert settings.openai_api_key == "sk-test-123"
    assert settings.openai_api_key_masked

    prompt = build_system_prompt(settings, "cs")
    assert "Neptej se na autorizaci" in prompt
    assert "Vždy odpovídej ve stejném jazyce" in prompt


def test_load_migrates_legacy_and_clamps_tts_speed(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        config_dir = Path(temp_dir) / "cfg"
        config_dir.mkdir()
        config_path = config_dir / "settings.json"
        config_path.write_text(
            json.dumps(
                {
                    "openai_api_key_masked": "321-tset-ks",
                    "tts_speed": 4.0,
                    "tts_voice": "nova",
                    "language": "česky",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr("kajovochat.settings._config_path", lambda: config_path)
        monkeypatch.setattr("kajovochat.settings._config_dir", lambda: config_dir)

        settings = AppSettings.load()

        assert settings.openai_api_key == "sk-test-123"
        assert settings.tts_speed == 1.5
        assert settings.tts_voice == "alloy"
        assert settings.language == "cs"


def test_load_recovers_from_broken_settings_file(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        config_dir = Path(temp_dir) / "cfg"
        config_dir.mkdir()
        config_path = config_dir / "settings.json"
        config_path.write_text("{ broken json", encoding="utf-8")

        monkeypatch.setattr("kajovochat.settings._config_path", lambda: config_path)
        monkeypatch.setattr("kajovochat.settings._config_dir", lambda: config_dir)

        settings = AppSettings.load()

        assert settings.realtime_model == "gpt-realtime"
        assert settings.write_logs is True
        assert list(config_dir.glob("settings.json.broken-*"))
        assert config_path.exists()


def test_validate_log_dir_creates_writable_probe() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        settings = AppSettings(log_dir=str(temp_dir))
        resolved = settings.validate_log_dir()
        assert resolved == Path(temp_dir).resolve()
