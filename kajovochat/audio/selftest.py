from __future__ import annotations

from .voice_gate import should_drop_mic_chunk


def run_audio_guard_selftest(
    *,
    default_profile: dict[str, float],
    sanitize_text_fn,
    pick_audio_device_fn,
    list_audio_devices_fn,
    calibrate_audio_devices_advanced_fn,
) -> dict[str, object]:
    """Lehký lokální selftest audio guardu a dostupnosti zařízení."""
    checks: list[dict[str, object]] = []
    profile = dict(default_profile)

    drop_echo, reason_echo = should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.94,
        input_level=0.03,
        output_level=0.09,
        default_profile=default_profile,
        profile=profile,
    )
    checks.append(
        {
            "name": "echo_drop",
            "ok": drop_echo and reason_echo == "echo_similarity",
            "detail": f"dropped={drop_echo}, reason={reason_echo or '-'}",
        }
    )

    keep_voice, reason_voice = should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.21,
        input_level=0.19,
        output_level=0.05,
        default_profile=default_profile,
        profile=profile,
    )
    checks.append(
        {
            "name": "voice_pass",
            "ok": (not keep_voice) and reason_voice == "",
            "detail": f"dropped={keep_voice}, reason={reason_voice or '-'}",
        }
    )

    input_device, input_note = pick_audio_device_fn("input", None)
    output_device, output_note = pick_audio_device_fn("output", None)
    devices = list_audio_devices_fn()
    checks.append(
        {
            "name": "devices_present",
            "ok": input_device is not None and output_device is not None,
            "detail": (
                f"in={input_device if input_device is not None else 'none'} ({input_note}), "
                f"out={output_device if output_device is not None else 'none'} ({output_note}), "
                f"inputs={len(devices.get('inputs', []))}, outputs={len(devices.get('outputs', []))}"
            ),
        }
    )

    if input_device is not None and output_device is not None:
        try:
            calibration = calibrate_audio_devices_advanced_fn(input_device=input_device, output_device=output_device)
            strong_playback_capture = calibration.playback_rms >= max(
                calibration.ambient_rms * 2.4,
                calibration.ambient_rms + 0.006,
            )
            strong_bleed_evidence = calibration.bleed_ratio >= 2.8
            correlation_detected = calibration.similarity >= 0.03
            auto_ok = strong_playback_capture and (strong_bleed_evidence or correlation_detected)
            checks.append(
                {
                    "name": "auto_calibration",
                    "ok": auto_ok,
                    "detail": "; ".join(calibration.notes),
                    "profile": calibration.recommended_profile,
                    "calibration": {
                        "latency_samples": getattr(calibration, "latency_samples", 0),
                        "preferred_frame_size": getattr(calibration, "preferred_frame_size", 480),
                        "filter_length": getattr(calibration, "filter_length", 256),
                        "audio_mode": getattr(calibration, "audio_mode", "notebook_builtin"),
                        "device_fingerprint": getattr(calibration, "device_fingerprint", "unknown"),
                    },
                    "non_blocking": strong_playback_capture,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "auto_calibration",
                    "ok": False,
                    "detail": sanitize_text_fn(str(exc)),
                    "profile": dict(profile),
                    "non_blocking": False,
                }
            )

    overall_ok = True
    for item in checks:
        if item["ok"]:
            continue
        if item.get("non_blocking"):
            continue
        overall_ok = False
        break

    return {
        "ok": overall_ok,
        "checks": checks,
        "profile": next((dict(item.get("profile", {})) for item in reversed(checks) if item.get("profile")), dict(profile)),
        "calibration": next((dict(item.get("calibration", {})) for item in reversed(checks) if item.get("calibration")), {}),
    }
