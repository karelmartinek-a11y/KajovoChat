from __future__ import annotations


def format_audio_log_payload(record_type: str, payload: dict[str, object]) -> dict[str, object]:
    """Doplní audio-specifické formátování log payloadu."""

    if record_type == "aec_diag" and "text_line" not in payload:
        payload["text_line"] = (
            "AEC"
            f" sim={float(payload.get('similarity', 0.0) or 0.0):.3f}"
            f" residual={float(payload.get('residual_level', 0.0) or 0.0):.4f}"
            f" q={float(payload.get('aec_quality', 0.0) or 0.0):.3f}"
            f" pred={float(payload.get('predicted_level', 0.0) or 0.0):.4f}"
            f" improve={float(payload.get('improvement_ratio', 0.0) or 0.0):.3f}"
            f" backend={str(payload.get('backend', 'unknown') or 'unknown')}"
            f" sel={str(payload.get('selection_reason', 'n/a') or 'n/a')}"
            f" ws={'on' if bool(payload.get('webrtc_success')) else 'off'}"
            f" native={'on' if bool(payload.get('native_selected')) else 'off'}"
            f" natry={'on' if bool(payload.get('native_attempted')) else 'off'}"
            f" dt={'on' if bool(payload.get('double_talk')) else 'off'}"
            f" voice={float(payload.get('voice_likelihood', 0.0) or 0.0):.3f}"
            f" delay={int(payload.get('delay_samples', 0) or 0)}"
            f" calib={int(payload.get('calibration_latency', 0) or 0)}"
            f" ref={int(payload.get('reference_available', 0) or 0)}"
            f" age_ms={int(payload.get('reference_callback_age_ms', -1) or -1)}"
            f" miss={'on' if bool(payload.get('reference_miss')) else 'off'}"
        )
    return payload
