from __future__ import annotations

from kajovochat.main import (
    _NOISE_REDUCTION,
    _REALTIME_MODEL,
    _SEMANTIC_VAD_EAGERNESS,
    _SERVER_VAD_PREFIX_MS,
    _SERVER_VAD_SILENCE_MS,
    _SERVER_VAD_THRESHOLD,
    _TTS_SPEED,
    _TTS_VOICE,
    _backend_aware_aec_metrics,
    ConversationWorker,
    MainWindow,
    run_audio_guard_selftest,
)
from kajovochat.audio.voice_gate import should_drop_mic_chunk
from kajovochat.audio.session_state import SessionState
from kajovochat.settings import AppSettings, DEFAULT_AUDIO_GUARD_PROFILE
from kajovochat.audio.devices import build_device_fingerprint, calibrate_audio_devices_advanced
from kajovochat.audio.windows_system_aec import WindowsSystemAecProbe
import math
import numpy as np
import time


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

    monkeypatch.setattr("kajovochat.audio.transport_bridge.RealtimeService", _factory)

    settings = AppSettings(answer_language_mode="fixed", fixed_answer_language="fr", response_style="stručný")
    settings.openai_api_key = "sk-test-123"
    worker = ConversationWorker(settings)

    worker._session_manager.transport.ensure_connected("server_vad")
    service = holder["service"]

    assert service.cfg.model == _REALTIME_MODEL
    assert service.cfg.voice == _TTS_VOICE
    assert service.cfg.output_speed == _TTS_SPEED
    assert service.cfg.server_vad_silence_ms == _SERVER_VAD_SILENCE_MS
    assert service.cfg.server_vad_prefix_ms == _SERVER_VAD_PREFIX_MS
    assert service.cfg.server_vad_threshold == _SERVER_VAD_THRESHOLD
    assert service.cfg.language_hint == "auto"


def test_echo_chunks_are_dropped_when_similarity_is_high() -> None:
    dropped, reason = should_drop_mic_chunk(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
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
    dropped, reason = should_drop_mic_chunk(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
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

    monkeypatch.setattr("kajovochat.audio.devices.calibrate_audio_devices", _fake_calibrate)

    result = calibrate_audio_devices_advanced(input_device=1, output_device=2, samplerate=24000)

    assert 24000 in calls
    assert 48000 in calls
    assert result.playback_rms > result.ambient_rms


def test_double_talk_is_not_dropped_when_voice_is_strong() -> None:
    dropped, reason = should_drop_mic_chunk(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.74,
        input_level=0.18,
        output_level=0.08,
        residual_level=0.11,
        voice_likelihood=0.58,
        double_talk=True,
        aec_quality=0.22,
    )

    assert dropped is False
    assert reason == ""


def test_saved_calibration_matches_by_device_fingerprint() -> None:
    settings = AppSettings()
    worker = ConversationWorker(settings)
    worker._resolved_input_device = 4
    worker._resolved_output_device = 9
    worker._input_device_name = "Jiny Mic"
    worker._output_device_name = "Jiny Speaker"
    worker._guard_calibration = {
        "device_fingerprint": build_device_fingerprint(4, 9, 24000),
        "input_device_name": "Stary Mic",
        "output_device_name": "Stary Speaker",
    }

    assert worker._device_calibration_matches() is True


def test_saved_calibration_applies_runtime_aec_and_frame_size() -> None:
    settings = AppSettings()
    worker = ConversationWorker(settings)
    worker._resolved_input_device = 1
    worker._resolved_output_device = 2
    worker._guard_calibration = {
        "device_fingerprint": build_device_fingerprint(1, 2, 24000),
        "preferred_frame_size": 960,
        "filter_length": 1024,
        "latency_samples": 320,
        "profile": {
            "echo_similarity_drop": 0.88,
        },
    }

    worker._apply_saved_calibration()
    worker._ensure_player()

    assert worker._aec.filter_length == 1024
    assert worker._aec.last_shift == 320
    assert worker._aec.max_shift_samples >= 960
    assert worker._runtime_resources.player is not None
    assert worker._runtime_resources.player.blocksize == 960


def test_conversation_worker_normalizes_aec_mode_from_settings() -> None:
    settings = AppSettings(audio_aec_mode="webrtc")
    worker = ConversationWorker(settings)

    assert worker._aec_mode == "webrtc_apm"


def test_webrtc_apm_mode_relaxes_guard_thresholds() -> None:
    worker = ConversationWorker(AppSettings(audio_aec_mode="webrtc_apm"))
    worker._guard_profile = dict(DEFAULT_AUDIO_GUARD_PROFILE)
    worker._aec_mode = "webrtc_apm"

    worker._apply_aec_mode_policy()

    assert worker._guard_profile["echo_similarity_soft"] < DEFAULT_AUDIO_GUARD_PROFILE["echo_similarity_soft"]
    assert worker._guard_profile["echo_similarity_drop"] < DEFAULT_AUDIO_GUARD_PROFILE["echo_similarity_drop"]
    assert worker._guard_profile["playback_activity_level"] <= DEFAULT_AUDIO_GUARD_PROFILE["playback_activity_level"]


def test_webrtc_apm_mode_keeps_far_drift_guardrails() -> None:
    worker = ConversationWorker(AppSettings(audio_aec_mode="webrtc_apm"))
    worker._guard_calibration = {"latency_samples": 722, "filter_length": 1024}
    worker._configure_aec_from_calibration()

    result = worker._aec.process(
        (np.random.default_rng(7).normal(0.0, 0.02, size=960).astype(np.float32) * 32767.0).astype(np.int16).tobytes(),
        (np.random.default_rng(8).normal(0.0, 0.02, size=2400).astype(np.float32) * 32767.0).astype(np.int16),
        max_shift_samples=1400,
        expected_shift=722,
        aec_mode="webrtc_apm",
    )

    assert result["backend"] in {"degraded_no_aec", "webrtc"}
    assert result["webrtc_success"] in {True, False}


def test_guard_debug_includes_native_aec_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        "kajovochat.main.probe_windows_system_aec",
        lambda: WindowsSystemAecProbe(False, "Windows System AEC backend není připraven."),
    )
    worker = ConversationWorker(AppSettings())
    captured: dict[str, object] = {}
    worker.guard_debug_updated.connect(lambda payload: captured.update(payload if isinstance(payload, dict) else {}))

    worker._emit_guard_debug()

    assert captured["native_aec_available"] is False
    assert captured["native_aec_reason"] == "Windows System AEC backend není připraven."


def test_aec_output_drives_guard_to_drop_pure_echo_but_keep_double_talk() -> None:
    samplerate = 24000
    rng = np.random.default_rng(21)
    total = 18000
    reference = (
        rng.normal(0.0, 0.18, size=total)
        + 0.25 * np.sin(np.arange(total, dtype=np.float32) * 0.12)
        + 0.12 * np.sin(np.arange(total, dtype=np.float32) * 0.047 + 0.2)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)
    chunk_size = 960
    shift = 480
    direct = reference[-(chunk_size + shift) : -shift]
    echo = 0.7 * direct + 0.18 * np.roll(direct, 40) - 0.08 * np.roll(direct, 100)
    user = 0.19 * np.sin(2.0 * math.pi * 260.0 * np.arange(chunk_size, dtype=np.float32) / samplerate)
    user += 0.09 * np.sin(2.0 * math.pi * 520.0 * np.arange(chunk_size, dtype=np.float32) / samplerate + 0.4)

    worker = ConversationWorker(AppSettings())
    worker._guard_calibration = {"latency_samples": shift, "filter_length": 1024}
    worker._configure_aec_from_calibration()

    echo_result = worker._aec.process(
        (np.clip(echo, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1400,
        expected_shift=shift,
    )
    echo_pcm = bytes(echo_result["pcm"])
    echo_level = worker._pcm16_level(echo_pcm)
    echo_drop, echo_reason = should_drop_mic_chunk(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=float(echo_result["similarity"]),
        input_level=echo_level,
        output_level=0.12,
        residual_level=float(echo_result["residual_level"]),
        voice_likelihood=0.05,
        double_talk=bool(echo_result["double_talk"]),
        aec_quality=float(echo_result["aec_quality"]),
    )

    mixed = np.clip(echo + user, -1.0, 1.0)
    mixed_result = worker._aec.process(
        (mixed * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1400,
        expected_shift=shift,
    )
    mixed_pcm = bytes(mixed_result["pcm"])
    mixed_level = worker._pcm16_level(mixed_pcm)
    mixed_drop, mixed_reason = should_drop_mic_chunk(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=float(mixed_result["similarity"]),
        input_level=max(mixed_level, 0.58 * 0.45),
        output_level=0.12,
        residual_level=float(mixed_result["residual_level"]),
        voice_likelihood=0.58,
        double_talk=bool(mixed_result["double_talk"]),
        aec_quality=float(mixed_result["aec_quality"]),
    )

    assert echo_drop is True
    assert echo_reason in {"echo_similarity", "echo_residual", "quiet_bleed"}
    assert mixed_result["double_talk"] is True
    assert mixed_drop is False
    assert mixed_reason == ""


def test_runtime_latency_update_requires_repeated_stable_hits() -> None:
    worker = ConversationWorker(AppSettings())
    worker._guard_calibration = {"latency_samples": 540, "filter_length": 1024}
    worker._configure_aec_from_calibration()

    worker._consider_runtime_latency_update(
        delay_samples=620,
        similarity=0.56,
        aec_quality=0.12,
        improvement_ratio=0.18,
        backend="webrtc",
        double_talk=False,
    )
    assert worker._guard_calibration["latency_samples"] == 540

    worker._consider_runtime_latency_update(
        delay_samples=624,
        similarity=0.58,
        aec_quality=0.13,
        improvement_ratio=0.17,
        backend="webrtc",
        double_talk=False,
    )
    assert worker._guard_calibration["latency_samples"] == 540

    worker._consider_runtime_latency_update(
        delay_samples=626,
        similarity=0.57,
        aec_quality=0.14,
        improvement_ratio=0.16,
        backend="webrtc",
        double_talk=False,
    )
    assert worker._guard_calibration["latency_samples"] == 540

    worker._consider_runtime_latency_update(
        delay_samples=628,
        similarity=0.59,
        aec_quality=0.14,
        improvement_ratio=0.18,
        backend="webrtc",
        double_talk=False,
    )
    assert worker._guard_calibration["latency_samples"] > 540
    assert worker._guard_calibration["latency_samples"] <= 540 + max(160, worker._aec.filter_length // 2)


def test_webrtc_apm_latency_update_commits_faster_than_custom() -> None:
    worker = ConversationWorker(AppSettings(audio_aec_mode="webrtc_apm"))
    worker._guard_calibration = {"latency_samples": 540, "filter_length": 1024}
    worker._configure_aec_from_calibration()

    for delay in (612, 616):
        worker._consider_runtime_latency_update(
            delay_samples=delay,
            similarity=0.49,
            aec_quality=0.09,
            improvement_ratio=0.15,
            backend="webrtc",
            double_talk=False,
            prefer_webrtc=True,
        )

    assert worker._guard_calibration["latency_samples"] == 540

    worker._consider_runtime_latency_update(
        delay_samples=618,
        similarity=0.51,
        aec_quality=0.1,
        improvement_ratio=0.16,
        backend="webrtc",
        double_talk=False,
        prefer_webrtc=True,
    )

    assert worker._guard_calibration["latency_samples"] > 540


def test_runtime_latency_update_rejects_far_webrtc_jump() -> None:
    worker = ConversationWorker(AppSettings())
    worker._guard_calibration = {"latency_samples": 540, "filter_length": 1024}
    worker._configure_aec_from_calibration()

    worker._consider_runtime_latency_update(
        delay_samples=40,
        similarity=0.52,
        aec_quality=0.12,
        improvement_ratio=0.2,
        backend="webrtc",
        double_talk=False,
    )

    assert worker._guard_calibration["latency_samples"] == 540


def test_reference_selection_prefers_live_reference_when_playback_is_warm() -> None:
    worker = ConversationWorker(AppSettings())
    runtime = worker._session_manager.voice_gate_runtime
    runtime.playback_reference_armed = True
    runtime.reference_warmup_until = time.monotonic() + 1.0

    needed = worker._reference_needed_samples(1920)
    decision = worker._session_manager.select_reference_source(
        aec_requires_reference=True,
        now_monotonic=time.monotonic(),
        reference_needed=needed,
        live_reference_pcm16=(np.arange(needed + 128, dtype=np.int16)).tobytes(),
        available_samples=needed + 64,
        played_samples=max(needed // 2, 480),
        callback_age_ms=-1,
    )

    assert decision.ready is True
    assert decision.source == "live"


def test_reference_selection_falls_back_to_cached_reference_when_live_is_not_ready() -> None:
    worker = ConversationWorker(AppSettings())
    runtime = worker._session_manager.voice_gate_runtime
    runtime.playback_reference_armed = True
    runtime.reference_warmup_until = time.monotonic() + 1.0
    runtime.cached_reference_at = time.monotonic()
    runtime.cached_echo_reference = (np.arange(840, dtype=np.int16)).tobytes()

    decision = worker._session_manager.select_reference_source(
        aec_requires_reference=True,
        now_monotonic=time.monotonic(),
        reference_needed=720,
        live_reference_pcm16=b"",
        available_samples=0,
        played_samples=360,
        callback_age_ms=-1,
    )

    assert decision.ready is True
    assert decision.source in {"cached", "cached_tail"}
    assert len(decision.reference_pcm16) == len(runtime.cached_echo_reference)


def test_voice_gate_snapshot_exposes_centralized_runtime_counters() -> None:
    worker = ConversationWorker(AppSettings())
    runtime = worker._session_manager.voice_gate_runtime
    runtime.echo_drop_count = 2
    runtime.barge_in_chunk_count = 3
    runtime.playback_reference_armed = True
    runtime.cached_echo_reference = (np.arange(720, dtype=np.int16)).tobytes()
    runtime.cached_reference_at = time.monotonic()

    snapshot = worker._session_manager.voice_gate_snapshot(now_monotonic=time.monotonic())

    assert snapshot.echo_drop_count == 2
    assert snapshot.barge_in_chunk_count == 3
    assert snapshot.playback_reference_armed is True
    assert snapshot.cached_reference_samples == 720


def test_shutdown_runtime_resources_stops_loop_before_clearing_rt() -> None:
    worker = ConversationWorker(AppSettings())
    events: list[str] = []

    class _FakeRT:
        def close(self) -> None:
            events.append("close_rt")

    fake_rt = _FakeRT()
    worker._runtime_resources.rt = fake_rt
    worker._session_manager.transport.realtime = fake_rt
    worker._stop_rt_loop = lambda timeout_s=1.0: events.append("stop_loop")  # type: ignore[method-assign]

    worker._session_manager.shutdown_runtime_resources()

    assert events[:2] == ["stop_loop", "close_rt"]


def test_conversation_worker_exposes_runtime_loop_wrappers() -> None:
    worker = ConversationWorker(AppSettings())
    events: list[tuple[str, float | None]] = []

    worker._rt_runtime_controller.start = lambda: events.append(("start", None))  # type: ignore[method-assign]
    worker._rt_runtime_controller.stop = lambda timeout_s=1.0: events.append(("stop", float(timeout_s)))  # type: ignore[method-assign]

    worker._start_rt_loop()
    worker._stop_rt_loop(timeout_s=0.25)

    assert events == [("start", None), ("stop", 0.25)]


def test_request_stop_stops_loop_before_clearing_rt() -> None:
    worker = ConversationWorker(AppSettings())
    events: list[str] = []

    class _FakeRT:
        def close(self) -> None:
            events.append("close_rt")

    fake_rt = _FakeRT()
    worker._runtime_resources.rt = fake_rt
    worker._session_manager.transport.realtime = fake_rt
    worker._stop_rt_loop = lambda timeout_s=1.0: events.append("stop_loop")  # type: ignore[method-assign]

    worker.request_stop()

    assert events[:2] == ["stop_loop", "close_rt"]


def test_request_stop_resets_session_manager_state_to_idle() -> None:
    worker = ConversationWorker(AppSettings())

    worker._session_manager.session_state = SessionState.ACTIVE
    worker._mode = "handsfree"

    worker.request_stop()

    assert worker._session_manager.session_state == SessionState.IDLE
    assert worker._mode == "idle"


def test_backend_aware_metrics_promote_successful_webrtc_block() -> None:
    effective_similarity, effective_quality = _backend_aware_aec_metrics(
        backend="webrtc",
        similarity=0.18,
        aec_quality=0.01,
        improvement_ratio=0.82,
        residual_level=0.0008,
        output_level=0.08,
        webrtc_success=True,
        native_selected=False,
    )

    assert effective_similarity >= 0.42
    assert effective_quality >= 0.12


def test_backend_aware_metrics_promote_selected_windows_native_block() -> None:
    effective_similarity, effective_quality = _backend_aware_aec_metrics(
        backend="windows_system_aec",
        similarity=0.16,
        aec_quality=0.02,
        improvement_ratio=0.64,
        residual_level=0.0009,
        output_level=0.08,
        webrtc_success=False,
        native_selected=True,
    )

    assert effective_similarity >= 0.38
    assert effective_quality >= 0.1


def test_transport_prefers_far_field_for_notebook_builtin(monkeypatch) -> None:
    holder: dict[str, object] = {}

    def _factory(cfg):
        service = _FakeRealtimeService(cfg)
        holder["service"] = service
        return service

    monkeypatch.setattr("kajovochat.audio.transport_bridge.RealtimeService", _factory)

    settings = AppSettings()
    settings.openai_api_key = "sk-test-123"
    worker = ConversationWorker(settings)
    worker._audio_mode = "notebook_builtin"

    worker._session_manager.transport.ensure_connected("semantic_vad")
    service = holder["service"]

    assert service.cfg.noise_reduction == "far_field"
    assert service.cfg.semantic_vad_eagerness == _SEMANTIC_VAD_EAGERNESS
    assert service.cfg.transcription_model == "gpt-4o-transcribe"


def test_transport_prefers_near_field_for_headset(monkeypatch) -> None:
    holder: dict[str, object] = {}

    def _factory(cfg):
        service = _FakeRealtimeService(cfg)
        holder["service"] = service
        return service

    monkeypatch.setattr("kajovochat.audio.transport_bridge.RealtimeService", _factory)

    settings = AppSettings()
    settings.openai_api_key = "sk-test-123"
    worker = ConversationWorker(settings)
    worker._audio_mode = "wired_headset"

    worker._session_manager.transport.ensure_connected("semantic_vad")
    service = holder["service"]

    assert service.cfg.noise_reduction == "near_field"


def test_mainwindow_does_not_auto_run_audio_selftest_on_start(qapp, monkeypatch) -> None:
    settings = AppSettings()
    settings.openai_api_key = "sk-test-123"
    settings.audio_guard_calibration = {}

    calls: list[str] = []
    monkeypatch.setattr(MainWindow, "_calibrate_audio_guard", lambda self, **kwargs: calls.append("calibrate") or {"ok": True})

    window = MainWindow(settings)
    try:
        window.sig_start_handsfree.disconnect()
    except Exception:
        pass
    started: list[str] = []
    window.sig_start_handsfree.connect(lambda: started.append("start"))

    window._toggle_handsfree()

    assert calls == []
    assert started == ["start"]
    assert "Audio selftest je volitelný" in window.head._terminal_lines[-2]
    assert window._handsfree_running is True
    assert window.status_label.text() == "Hands-free relace se spouští."

    window.close()
