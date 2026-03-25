from __future__ import annotations

from kajovochat.audio.contracts import BackendHealthSnapshot, CaptureFrame, RenderFrame, SessionHealth
from kajovochat.audio.io import CapturedAudioChunk, DuplexAudioSession


def test_captured_audio_chunk_converts_to_capture_frame() -> None:
    chunk = CapturedAudioChunk(pcm_bytes=b"\x01\x00\x02\x00", captured_at_mono_ns=123456789)

    frame = chunk.to_capture_frame(
        frame_index=7,
        sample_rate=24000,
        aec_backend="webrtc_apm",
        aec_quality=0.42,
        residual_level=0.01,
        vad_probability=0.9,
        double_talk=False,
        stream_delay_ms=18,
    )

    assert isinstance(frame, CaptureFrame)
    assert frame.frame_index == 7
    assert frame.mono_ns == 123456789
    assert frame.raw_mic_pcm16 == b"\x01\x00\x02\x00"
    assert frame.processed_mic_pcm16 == b"\x01\x00\x02\x00"
    assert frame.aec_backend == "webrtc_apm"
    assert frame.to_log_payload()["raw_mic_bytes"] == 4


def test_duplex_audio_session_builds_render_frame_contract() -> None:
    duplex = DuplexAudioSession(samplerate=24000, input_device=None, output_device=None, blocksize=480)

    frame = duplex.build_render_frame(
        frame_index=3,
        mono_ns=456789123,
        pcm16=b"\x05\x00\x06\x00",
        tts_active=True,
        prompted_by_assistant_turn="turn-1",
    )

    assert isinstance(frame, RenderFrame)
    assert frame.frame_index == 3
    assert frame.mono_ns == 456789123
    assert frame.tts_active is True
    assert frame.prompted_by_assistant_turn == "turn-1"
    assert frame.to_log_payload()["pcm_bytes"] == 4


def test_session_health_contract_exposes_log_payload() -> None:
    snapshot = SessionHealth(
        requested_backend="windows_system_aec",
        selected_backend="webrtc_apm",
        fallback_reason="windows_system_aec_unavailable",
        degradation_cause="",
        device_fingerprint="fp-1",
        audio_mode="notebook_builtin",
        session_state="active",
        session_started_at_mono=1.0,
        session_activated_at_mono=2.0,
        uptime_s=3.5,
        active_for_s=1.5,
        last_server_activity_age_s=0.25,
        reference_ready=True,
        reference_health="ready",
        reference_available_samples=480,
        reference_callback_age_ms=12,
        reference_ready_events=3,
        reference_miss_events=1,
        reference_consecutive_misses=0,
        poor_aec_events=0,
        poor_aec_consecutive=0,
        recovery_attempts_scheduled=0,
        recovery_attempts_total=1,
        next_reconnect_at_mono=0.0,
        last_failure_reason="",
        backend_health=BackendHealthSnapshot(backend="webrtc_apm", health_score=0.9),
    )

    payload = snapshot.to_log_dict()
    assert payload["selected_backend"] == "webrtc_apm"
    assert payload["backend_health"]["backend"] == "webrtc_apm"
    assert payload["recovery_attempts"]["total"] == 1
