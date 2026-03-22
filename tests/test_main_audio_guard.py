from __future__ import annotations

from kajovochat.main import (
    _REALTIME_MODEL,
    _SERVER_VAD_PREFIX_MS,
    _SERVER_VAD_SILENCE_MS,
    _SERVER_VAD_THRESHOLD,
    _TTS_SPEED,
    _TTS_VOICE,
    _should_drop_mic_chunk,
    ConversationWorker,
    run_audio_guard_selftest,
)
from kajovochat.settings import AppSettings
from kajovochat.services.audio_service import calibrate_audio_devices_advanced


class _FakeRealtimeService:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.is_connected = False
        self.on_status = None
        self.on_error = None
        self.on_user_transcript = None
        self.on_assistant_text_delta = None
        self.on_assistant_text_done = None
        self.on_assistant_audio_delta = None
        self.on_vad_speech_started = None
        self.on_vad_speech_stopped = None
        self.on_response_done = None

    def connect(self) -> None:
        self.is_connected = True

    def close(self) -> None:
        self.is_connected = False

    def update_session(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.cfg, key, value)


def test_realtime_config_is_hardcoded_and_not_user_tunable(monkeypatch) -> None:
    holder: dict[str, object] = {}

    def _factory(cfg):
        service = _FakeRealtimeService(cfg)
        holder["service"] = service
        return service

    monkeypatch.setattr("kajovochat.main.RealtimeService", _factory)

    settings = AppSettings(answer_language_mode="fixed", fixed_answer_language="fr", response_style="stručný")
    settings.openai_api_key = "sk-test-123"
    worker = ConversationWorker(settings)

    worker._ensure_realtime("server_vad")
    service = holder["service"]

    assert service.cfg.model == _REALTIME_MODEL
    assert service.cfg.voice == _TTS_VOICE
    assert service.cfg.output_speed == _TTS_SPEED
    assert service.cfg.server_vad_silence_ms == _SERVER_VAD_SILENCE_MS
    assert service.cfg.server_vad_prefix_ms == _SERVER_VAD_PREFIX_MS
    assert service.cfg.server_vad_threshold == _SERVER_VAD_THRESHOLD
    assert service.cfg.language_hint == "auto"


def test_echo_chunks_are_dropped_when_similarity_is_high() -> None:
    dropped, reason = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.93,
        input_level=0.05,
        output_level=0.08,
    )

    assert dropped is True
    assert reason == "echo_similarity"


def test_user_voice_passes_when_similarity_is_low() -> None:
    dropped, reason = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.18,
        input_level=0.18,
        output_level=0.06,
    )

    assert dropped is False
    assert reason == ""


def test_audio_guard_selftest_reports_all_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        "kajovochat.main.pick_audio_device",
        lambda kind, preferred: (3 if kind == "input" else 7, "selected:test"),
    )
    monkeypatch.setattr(
        "kajovochat.main.list_audio_devices",
        lambda: {"inputs": [{"index": 3, "name": "Mic"}], "outputs": [{"index": 7, "name": "Speaker"}]},
    )
    monkeypatch.setattr(
        "kajovochat.main.calibrate_audio_devices_advanced",
        lambda **kwargs: type(
            "Calibration",
            (),
            {
                "playback_rms": 0.08,
                "ambient_rms": 0.01,
                "bleed_ratio": 8.0,
                "similarity": 0.41,
                "notes": ["ambient_rms=0.0100", "playback_rms=0.0800", "bleed_ratio=8.00", "similarity=0.410"],
                "recommended_profile": {
                    "server_vad_threshold": 0.73,
                    "playback_activity_level": 0.04,
                    "echo_similarity_drop": 0.84,
                    "echo_similarity_soft": 0.66,
                    "barge_in_min_input_level": 0.07,
                    "barge_in_output_ratio": 1.42,
                },
            },
        )(),
    )

    result = run_audio_guard_selftest()

    assert result["ok"] is True
    assert [item["name"] for item in result["checks"]] == ["echo_drop", "voice_pass", "devices_present", "auto_calibration"]
    assert result["profile"]["echo_similarity_drop"] == 0.84


def test_audio_guard_selftest_tolerates_low_similarity_when_bleed_is_clear(monkeypatch) -> None:
    monkeypatch.setattr(
        "kajovochat.main.pick_audio_device",
        lambda kind, preferred: (2 if kind == "input" else 3, "selected:test"),
    )
    monkeypatch.setattr(
        "kajovochat.main.list_audio_devices",
        lambda: {"inputs": [{"index": 2, "name": "Mic"}], "outputs": [{"index": 3, "name": "Speaker"}]},
    )
    monkeypatch.setattr(
        "kajovochat.main.calibrate_audio_devices_advanced",
        lambda **kwargs: type(
            "Calibration",
            (),
            {
                "playback_rms": 0.0212,
                "ambient_rms": 0.0031,
                "bleed_ratio": 6.78,
                "similarity": 0.012,
                "notes": ["passes=3", "ambient_med=0.0031", "playback_med=0.0212", "bleed_peak=6.78", "similarity_peak=0.012"],
                "recommended_profile": {
                    "server_vad_threshold": 0.722,
                    "playback_activity_level": 0.028,
                    "echo_similarity_drop": 0.782,
                    "echo_similarity_soft": 0.611,
                    "barge_in_min_input_level": 0.05,
                    "barge_in_output_ratio": 1.61,
                },
            },
        )(),
    )

    result = run_audio_guard_selftest()

    assert result["ok"] is True
    auto = result["checks"][-1]
    assert auto["name"] == "auto_calibration"
    assert auto["ok"] is True


def test_advanced_calibration_tries_fallback_samplerates(monkeypatch) -> None:
    calls: list[int] = []

    def _fake_calibrate(**kwargs):
        calls.append(int(kwargs["samplerate"]))
        if int(kwargs["samplerate"]) == 24000:
            raise RuntimeError("unsupported")
        return type(
            "Calibration",
            (),
            {
                "input_device": 1,
                "output_device": 2,
                "ambient_rms": 0.002,
                "playback_rms": 0.02,
                "bleed_ratio": 5.0,
                "similarity": 0.2,
                "recommended_profile": {
                    "server_vad_threshold": 0.73,
                    "playback_activity_level": 0.03,
                    "echo_similarity_drop": 0.82,
                    "echo_similarity_soft": 0.62,
                    "barge_in_min_input_level": 0.05,
                    "barge_in_output_ratio": 1.4,
                },
                "notes": ["ok"],
            },
        )()

    monkeypatch.setattr("kajovochat.services.audio_service.calibrate_audio_devices", _fake_calibrate)

    result = calibrate_audio_devices_advanced(input_device=1, output_device=2, samplerate=24000)

    assert 24000 in calls
    assert 48000 in calls
    assert result.playback_rms > result.ambient_rms
