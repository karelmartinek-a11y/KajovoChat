from __future__ import annotations

import time

from kajovochat.services.audio_service import AudioPlayer


def test_audio_player_echo_reference_tracks_played_target_samples() -> None:
    player = AudioPlayer(samplerate=24000)
    first = (b"\x01\x00" * 100)
    second = (b"\x02\x00" * 120)

    player._echo_reference_enqueued_samples = 220
    player._echo_reference_played_samples = 180
    player._echo_reference_chunks.append((100, first))
    player._echo_reference_chunks.append((220, second))

    tail = player.get_echo_reference(max_samples=80)
    stats = player.get_echo_reference_stats()

    assert tail.size == 80
    assert int(tail[0]) == 2
    assert int(tail[-1]) == 2
    assert stats["played_samples"] == 180
    assert stats["total_samples"] == 220


def test_audio_player_echo_reference_respects_capture_time() -> None:
    player = AudioPlayer(samplerate=24000)
    first = (b"\x01\x00" * 120)
    second = (b"\x02\x00" * 120)

    player._echo_reference_enqueued_samples = 240
    player._echo_reference_played_samples = 240
    player._echo_reference_chunks.append((120, first))
    player._echo_reference_chunks.append((240, second))
    now_ns = time.monotonic_ns()
    player._last_callback_mono_ns = now_ns
    player._echo_reference_played_end_mono_ns = now_ns + int(round(120 * (1_000_000_000.0 / 24000.0)))

    immediate = player.get_echo_reference_for_capture(max_samples=120, captured_at_mono_ns=now_ns)
    delayed = player.get_echo_reference_for_capture(
        max_samples=120,
        captured_at_mono_ns=player._echo_reference_played_end_mono_ns + 1_000_000,
    )

    assert immediate.size == 120
    assert int(immediate[0]) == 1
    assert int(immediate[-1]) == 1
    assert delayed.size == 120
    assert int(delayed[0]) == 2
    assert int(delayed[-1]) == 2
