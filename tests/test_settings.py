from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kajovochat.settings import (
    AppSettings,
    DEFAULT_AUDIO_GUARD_PROFILE,
    RESPONSE_STYLE_CHOICES,
    build_system_prompt,
)


def test_settings_api_key_roundtrip_and_prompt_follow_input() -> None:
    settings = AppSettings(answer_language_mode="follow_input", fixed_answer_language="en", response_style="normální")
    settings.openai_api_key = "sk-test-123"

    assert settings.openai_api_key == "sk-test-123"
    assert settings.openai_api_key_masked

    prompt = build_system_prompt(settings, "de")
    assert "Odpovídej ve stejném jazyce" in prompt
    assert "Antworte auf Deutsch." in prompt
    assert "bez ohledu na jazyk vstupního dotazu" not in prompt


def test_prompt_fixed_output_language() -> None:
    settings = AppSettings(answer_language_mode="fixed", fixed_answer_language="fr", response_style="stručný")
    prompt = build_system_prompt(settings, "cs")

    assert "Réponds en français." in prompt
    assert "bez ohledu na jazyk vstupního dotazu" in prompt
    assert "Odpovídej ve stejném jazyce" not in prompt


def test_prompt_styles_exact_three_presets() -> None:
    assert [code for code, _ in RESPONSE_STYLE_CHOICES] == ["stručný", "vědecký_s_analýzou", "normální"]

    scientific_prompt = build_system_prompt(
        AppSettings(answer_language_mode="follow_input", fixed_answer_language="cs", response_style="vědecký_s_analýzou"),
        "cs",
    )
    assert "analyticky a strukturovaně" in scientific_prompt
    assert "hysterick" not in scientific_prompt.lower()


def test_load_migrates_legacy_and_reduces_schema(monkeypatch) -> None:
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
                    "response_style": "strohé",
                    "temperature": 0.9,
                    "max_output_tokens": 2048,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr("kajovochat.settings._config_path", lambda: config_path)
        monkeypatch.setattr("kajovochat.settings._config_dir", lambda: config_dir)

        settings = AppSettings.load()

        assert settings.openai_api_key == "sk-test-123"
        assert settings.answer_language_mode == "fixed"
        assert settings.fixed_answer_language == "cs"
        assert settings.response_style == "stručný"
        assert not hasattr(settings, "temperature")


def test_load_recovers_from_broken_settings_file(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        config_dir = Path(temp_dir) / "cfg"
        config_dir.mkdir()
        config_path = config_dir / "settings.json"
        config_path.write_text("{ broken json", encoding="utf-8")

        monkeypatch.setattr("kajovochat.settings._config_path", lambda: config_path)
        monkeypatch.setattr("kajovochat.settings._config_dir", lambda: config_dir)

        settings = AppSettings.load()

        assert settings.answer_language_mode == "follow_input"
        assert settings.response_style == "normální"
        assert list(config_dir.glob("settings.json.broken-*"))
        assert config_path.exists()


def test_validate_log_dir_creates_writable_probe() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        settings = AppSettings(log_dir=str(temp_dir))
        resolved = settings.validate_log_dir()
        assert resolved == Path(temp_dir).resolve()


def test_audio_guard_profile_is_normalized() -> None:
    settings = AppSettings(audio_guard_profile={"echo_similarity_soft": 0.9, "echo_similarity_drop": 0.85})
    profile = settings.normalized_audio_guard_profile()

    assert profile["echo_similarity_soft"] == 0.9
    assert profile["echo_similarity_drop"] >= profile["echo_similarity_soft"] + 0.04
    assert profile["server_vad_threshold"] == DEFAULT_AUDIO_GUARD_PROFILE["server_vad_threshold"]
