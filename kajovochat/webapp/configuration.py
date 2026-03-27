from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..settings import AppSettings, build_system_prompt, normalize_fixed_language, normalize_realtime_voice

DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_OUTPUT_SPEED = 1.0
DEFAULT_TURN_DETECTION = {
    "type": "semantic_vad",
    "eagerness": "low",
    "create_response": True,
    "interrupt_response": True,
}

_HEADSET_TOPOLOGIES = {
    "headset",
    "headphones",
    "wired_headset",
    "bluetooth_headset",
    "external_headphones",
}


def infer_noise_reduction(audio_topology: str | None) -> str:
    topology = (audio_topology or "").strip().lower()
    if topology in _HEADSET_TOPOLOGIES:
        return "near_field"
    return "far_field"


def _transcription_config(settings: AppSettings, browser_language: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"model": DEFAULT_TRANSCRIPTION_MODEL}
    if settings.answer_language_mode == "fixed":
        fixed = normalize_fixed_language(settings.fixed_answer_language)
        if fixed and fixed != "auto":
            config["language"] = fixed
            return config
    language = (browser_language or "").strip().lower().split("-", 1)[0]
    if language in {"cs", "en", "de", "sk", "fr"}:
        config["language"] = language
    return config


def build_realtime_session_config(settings: AppSettings, client_hints: dict[str, Any] | None = None) -> dict[str, Any]:
    hints = client_hints or {}
    fallback_language = normalize_fixed_language(settings.fixed_answer_language)
    instructions = build_system_prompt(settings, fallback_language)
    audio_topology = hints.get("audio_topology") or settings.audio_device_mode
    noise_reduction = infer_noise_reduction(str(audio_topology or ""))
    voice = normalize_realtime_voice(getattr(settings, "realtime_voice", "marin"))

    session = {
        "type": "realtime",
        "model": DEFAULT_REALTIME_MODEL,
        "instructions": instructions,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": _transcription_config(settings, hints.get("browser_language")),
                "noise_reduction": {"type": noise_reduction},
                "turn_detection": dict(DEFAULT_TURN_DETECTION),
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": voice,
                "speed": float(DEFAULT_OUTPUT_SPEED),
            },
        },
        "output_modalities": ["audio"],
    }
    return session


def public_settings_payload(settings: AppSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload.pop("openai_api_key_masked", None)
    payload["has_api_key"] = bool(settings.openai_api_key)
    payload["realtime_voice"] = normalize_realtime_voice(getattr(settings, "realtime_voice", "marin"))
    return payload
