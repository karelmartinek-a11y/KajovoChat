from __future__ import annotations

from .voice_gate import should_drop_mic_chunk


def _is_echo_guard_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    return normalized in {"playback_voice_echo", "playback_voice_lock"}


def evaluate_audio_preflight(
    *,
    calibration: dict[str, object] | None,
    current_device_fingerprint: str,
) -> tuple[bool, str]:
    """Vrátí, zda má před startem běžet aktivní preflight selftest."""
    saved = calibration if isinstance(calibration, dict) else {}
    if not saved:
        return True, "missing_calibration"

    saved_fingerprint = str(saved.get("device_fingerprint", "") or "").strip().lower()
    current_fingerprint = str(current_device_fingerprint or "").strip().lower()
    if not saved_fingerprint or saved_fingerprint == "unknown":
        return True, "missing_device_fingerprint"
    if current_fingerprint and saved_fingerprint != current_fingerprint:
        return True, "device_changed"

    saved_profile = saved.get("profile")
    if not isinstance(saved_profile, dict) or not saved_profile:
        return True, "missing_profile"
    if int(saved.get("latency_samples", 0) or 0) <= 0:
        return True, "missing_latency"

    last_recommendation = str(saved.get("last_monitor_recommendation", "") or "").strip().lower()
    if last_recommendation == "needs_preflight":
        return True, "monitor_requested_preflight"

    last_guard_state = str(saved.get("last_guard_state", "") or "").strip().lower()
    last_guard_top_reason = str(saved.get("last_guard_top_reason", "") or "").strip().lower()
    last_drop_rate = float(saved.get("last_drop_rate", 0.0) or 0.0)
    if last_guard_state == "echo_heavy" and _is_echo_guard_reason(last_guard_top_reason):
        return True, "previous_session_echo_heavy"
    if _is_echo_guard_reason(last_guard_top_reason) and last_drop_rate >= 0.18:
        return True, "previous_session_high_echo_drop"

    return False, ""


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

    def _calibration_mode_requires_latency(audio_mode: str) -> bool:
        normalized = str(audio_mode or "").strip().lower()
        return normalized in {"notebook_builtin", "external_speakers", "monitor_speakers"}

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
            calibration_latency = int(getattr(calibration, "latency_samples", 0) or 0)
            calibration_audio_mode = str(getattr(calibration, "audio_mode", "notebook_builtin") or "notebook_builtin")
            strong_playback_capture = calibration.playback_rms >= max(
                calibration.ambient_rms * 2.4,
                calibration.ambient_rms + 0.006,
            )
            strong_bleed_evidence = calibration.bleed_ratio >= 2.8
            correlation_detected = calibration.similarity >= 0.03
            latency_required = _calibration_mode_requires_latency(calibration_audio_mode)
            latency_detected = calibration_latency > 0
            auto_ok = (
                strong_playback_capture
                and (strong_bleed_evidence or correlation_detected)
                and (latency_detected or not latency_required)
            )
            checks.append(
                {
                    "name": "auto_calibration",
                    "ok": auto_ok,
                    "detail": "; ".join(calibration.notes),
                    "profile": calibration.recommended_profile,
                    "calibration": {
                        "latency_samples": calibration_latency,
                        "preferred_frame_size": getattr(calibration, "preferred_frame_size", 480),
                        "filter_length": getattr(calibration, "filter_length", 256),
                        "audio_mode": calibration_audio_mode,
                        "device_fingerprint": getattr(calibration, "device_fingerprint", "unknown"),
                    },
                    "non_blocking": strong_playback_capture and not latency_required,
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
        "startup_ready": bool(overall_ok),
    }
