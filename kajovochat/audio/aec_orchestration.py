from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..settings import normalize_audio_aec_mode


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def _normalized_similarity(reference: np.ndarray, candidate: np.ndarray, *, max_shift_samples: int) -> float:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if ref.size == 0 or cand.size == 0:
        return 0.0
    if ref.size != cand.size:
        size = min(ref.size, cand.size)
        ref = ref[:size]
        cand = cand[:size]
    ref -= float(np.mean(ref))
    cand -= float(np.mean(cand))
    ref_norm = float(np.linalg.norm(ref) + 1e-6)
    cand_norm = float(np.linalg.norm(cand) + 1e-6)
    if ref_norm <= 1e-6 or cand_norm <= 1e-6:
        return 0.0
    if max_shift_samples <= 0:
        return float(abs(float(np.dot(ref, cand)) / (ref_norm * cand_norm)))
    best = 0.0
    max_shift = max(0, int(max_shift_samples))
    for shift in range(-max_shift, max_shift + 1, 32):
        if shift >= 0:
            lhs = ref[shift:]
            rhs = cand[: lhs.size]
        else:
            rhs = cand[-shift:]
            lhs = ref[: rhs.size]
        if lhs.size < 64 or rhs.size != lhs.size:
            continue
        lhs_norm = float(np.linalg.norm(lhs) + 1e-6)
        rhs_norm = float(np.linalg.norm(rhs) + 1e-6)
        if lhs_norm <= 1e-6 or rhs_norm <= 1e-6:
            continue
        corr = abs(float(np.dot(lhs, rhs)) / (lhs_norm * rhs_norm))
        if corr > best:
            best = corr
    return float(best)


def process_adaptive_echo(
    owner: Any,
    mic_pcm: bytes,
    reference: np.ndarray,
    *,
    max_shift_samples: Optional[int] = None,
    expected_shift: Optional[int] = None,
    aec_mode: str = "custom_lab",
) -> dict[str, object]:
    aec_mode = normalize_audio_aec_mode(aec_mode)
    prefer_native_mode = bool(aec_mode == "windows_system_aec")
    prefer_webrtc_mode = bool(aec_mode == "webrtc_apm")
    production_external_mode = bool(prefer_native_mode or prefer_webrtc_mode)
    headset_clean_mode = bool(aec_mode == "headset_clean")
    degraded_mode = bool(aec_mode == "degraded_no_aec")
    custom_lab_mode = bool(aec_mode == "custom_lab")
    if not mic_pcm:
        return {
            "pcm": b"",
            "similarity": 0.0,
            "delay_samples": int(owner._last_shift),
            "double_talk": False,
            "residual_level": 0.0,
            "mic_level": 0.0,
            "aec_quality": 0.0,
            "webrtc_success": False,
            "voice_likelihood": 0.0,
        }

    if prefer_native_mode and getattr(owner._windows_native_probe, "installed_driver", False):
        return owner._process_windows_system_capture(mic_pcm)

    if prefer_webrtc_mode:
        return owner._process_webrtc_only(
            mic_pcm=mic_pcm,
            reference=reference,
            expected_shift=expected_shift,
            max_shift_samples=max_shift_samples,
        )

    prep, early_result = owner._prepare_reference_window(
        mic_pcm=mic_pcm,
        reference=reference,
        expected_shift=expected_shift,
        max_shift_samples=max_shift_samples,
    )
    if early_result is not None:
        if degraded_mode or headset_clean_mode:
            early_result = dict(early_result)
            early_result.update(
                {
                    "predicted_level": 0.0,
                    "improvement_ratio": 0.0,
                    "backend": "headset_clean" if headset_clean_mode else "degraded_no_aec",
                    "native_attempted": False,
                    "native_selected": False,
                    "selection_reason": "headset_clean" if headset_clean_mode else "degraded_passthrough",
                }
            )
        return early_result
    assert prep is not None
    voice_likelihood = prep.voice_likelihood
    if degraded_mode or headset_clean_mode:
        level = _rms(prep.mic / 32768.0)
        return {
            "pcm": mic_pcm,
            "similarity": 0.0,
            "delay_samples": int(owner._last_shift),
            "double_talk": False,
            "residual_level": float(level),
            "mic_level": float(level),
            "aec_quality": 0.0,
            "predicted_level": 0.0,
            "improvement_ratio": 0.0,
            "backend": "headset_clean" if headset_clean_mode else "degraded_no_aec",
            "webrtc_success": False,
            "native_attempted": False,
            "native_selected": False,
            "selection_reason": "headset_clean" if headset_clean_mode else "degraded_passthrough",
            "voice_likelihood": voice_likelihood,
        }
    mic = prep.mic
    mic_centered = prep.mic_centered
    context_norm = prep.context_norm
    design = prep.design
    segment = prep.segment
    similarity = prep.similarity
    best_shift = prep.best_shift
    mic_level = prep.mic_level
    ref_level = prep.ref_level
    anchor_shift = prep.anchor_shift
    shift_error = prep.shift_error
    stable_delay_lock = prep.stable_delay_lock
    taps = int(owner._filter_length)
    active_weights = owner._weights.astype(np.float32, copy=True)
    predicted_before = design @ active_weights

    residual_before = mic_centered - predicted_before

    probe_weights, predicted_probe = owner._nlms_candidate(design, mic_centered, iterations=1)
    residual_probe = mic_centered - predicted_probe
    residual_level_probe = _rms(residual_probe)
    predicted_level_probe = _rms(predicted_probe)
    improvement_ratio = 1.0 - min(1.0, residual_level_probe / max(mic_level, 1e-4))
    custom_anchor_guard = bool(anchor_shift > 0 and shift_error > max(176, taps // 3) and similarity < 0.78)
    double_talk = owner._detect_double_talk(
        similarity=similarity,
        mic_level=mic_level,
        ref_level=ref_level,
        predicted_level=predicted_level_probe,
        residual_level=residual_level_probe,
        improvement_ratio=improvement_ratio,
        voice_likelihood=voice_likelihood,
        previous_double_talk=owner._last_double_talk,
    )
    adapt_allowed = bool(
        not production_external_mode
        and similarity >= 0.45
        and ref_level >= 0.012
        and predicted_level_probe >= 0.015
        and not double_talk
        and not custom_anchor_guard
        and (anchor_shift <= 0 or shift_error <= max(176, taps // 3) or stable_delay_lock or similarity >= 0.88)
    )

    if adapt_allowed:
        iterations = 2 if similarity >= 0.72 else 1
        nlms_weights, _predicted_after = owner._nlms_candidate(
            design,
            mic_centered,
            iterations=iterations,
            initial_weights=probe_weights,
        )
        candidate_weights = nlms_weights
        if similarity >= 0.45:
            ridge_weights, ridge_predicted = owner._ridge_candidate(
                design,
                mic_centered,
                ridge=max(owner.ridge, 2.5e-3),
            )
            ridge_residual = mic_centered - ridge_predicted
            ridge_residual_level = _rms(ridge_residual)
            ridge_predicted_level = _rms(ridge_predicted)
            ridge_improvement = 1.0 - min(1.0, ridge_residual_level / max(mic_level, 1e-4))
            if ridge_predicted_level >= predicted_level_probe * 0.8 and ridge_improvement >= improvement_ratio - 0.04:
                candidate_weights = (nlms_weights * 0.4) + (ridge_weights * 0.6)
        if np.any(owner._weights):
            blend = 0.18
            if similarity >= 0.72 and improvement_ratio >= 0.12:
                blend = 0.42
            elif similarity >= 0.55 and improvement_ratio >= 0.06:
                blend = 0.28
            owner._weights = (owner._weights * (1.0 - blend)) + (candidate_weights * blend)
        else:
            owner._weights = candidate_weights
        active_weights = owner._weights
        predicted = design @ active_weights
        residual = mic_centered - predicted
    else:
        if production_external_mode:
            predicted = np.zeros_like(mic_centered)
            residual = mic_centered
        else:
            predicted = predicted_before
            residual = residual_before

    post_similarity = _normalized_similarity(segment.astype(np.int16), (residual * 32767.0).astype(np.int16), max_shift_samples=0)
    if not production_external_mode and similarity >= 0.24 and post_similarity >= 0.1 and not double_talk:
        suppress = min(0.2, max(0.0, similarity - post_similarity) * 0.3 + 0.06)
        residual -= context_norm[taps - 1 :] * suppress
    cleaned = np.clip((residual + float(np.mean(mic / 32768.0))) * 32767.0, -32768.0, 32767.0).astype(np.int16)
    residual_level = _rms(cleaned.astype(np.float32) / 32768.0)
    predicted_level = _rms(predicted)
    improvement_ratio = max(0.0, 1.0 - min(1.0, residual_level / max(mic_level, 1e-4)))
    backend_used = "custom" if custom_lab_mode else "degraded_no_aec"
    webrtc_success = False
    native_attempted = False
    native_selected = False
    selection_reason = "custom_fallback" if custom_lab_mode else "degraded_fallback"
    probe_similarity = float(similarity)
    probe_shift_error = int(shift_error)
    if (
        not production_external_mode
        and similarity >= 0.2
        and backend_used == "custom"
        and improvement_ratio < 0.03
        and not double_talk
    ):
        similarity = min(float(similarity), 0.08)
        predicted_level = 0.0
        residual_level = float(mic_level)
        best_shift = 0
        cleaned = mic.astype(np.int16, copy=False)
        selection_reason = "custom_low_gain"
    if (prefer_native_mode or custom_lab_mode) and segment.size == mic.size and not double_talk:
        native_result = owner._run_windows_system_aec(
            prep=prep,
            mic_pcm=mic_pcm,
            residual_level=residual_level,
            improvement_ratio=improvement_ratio,
            predicted_level=predicted_level,
            backend_used=backend_used,
        )
        if native_result is not None:
            native_attempted = bool(native_result.get("native_attempted", False))
            if native_result.get("native_selected"):
                cleaned = native_result["pcm"]
                similarity = float(native_result["similarity"])
                residual_level = float(native_result["residual_level"])
                improvement_ratio = float(native_result["improvement_ratio"])
                predicted_level = float(native_result["predicted_level"])
                best_shift = int(native_result["delay_samples"])
                backend_used = str(native_result["backend"])
                webrtc_success = bool(native_result["webrtc_success"])
                native_selected = True
                selection_reason = str(native_result["selection_reason"])
    allow_webrtc_probe = bool(
        (
            probe_similarity >= 0.18
            and (anchor_shift <= 0 or probe_shift_error <= max(288, taps // 2) or probe_similarity >= 0.62)
        )
        or (
            stable_delay_lock
            and ref_level >= 0.012
            and mic_level >= 0.01
            and predicted_level_probe >= 0.003
        )
    )
    if degraded_mode or headset_clean_mode:
        allow_webrtc_probe = False
    elif custom_lab_mode:
        allow_webrtc_probe = True
    elif prefer_native_mode:
        allow_webrtc_probe = True
    elif prefer_webrtc_mode:
        allow_webrtc_probe = bool(
            allow_webrtc_probe
            or (
                stable_delay_lock
                and ref_level >= 0.008
                and mic_level >= 0.008
                and probe_similarity >= 0.14
            )
        )
    if allow_webrtc_probe and segment.size == mic.size and not double_talk:
        webrtc_result = owner._run_webrtc_apm(
            prep=prep,
            mic_pcm=mic_pcm,
            residual_level=residual_level,
            improvement_ratio=improvement_ratio,
            predicted_level=predicted_level,
            prefer_native_mode=prefer_native_mode,
            backend_used=backend_used,
        )
        if webrtc_result is not None:
            cleaned = webrtc_result["pcm"]
            similarity = float(webrtc_result["similarity"])
            residual_level = float(webrtc_result["residual_level"])
            improvement_ratio = float(webrtc_result["improvement_ratio"])
            predicted_level = float(webrtc_result["predicted_level"])
            best_shift = int(webrtc_result["delay_samples"])
            backend_used = str(webrtc_result["backend"])
            webrtc_success = bool(webrtc_result["webrtc_success"])
            native_selected = False
            selection_reason = str(webrtc_result["selection_reason"])
    if not custom_lab_mode and backend_used == "degraded_no_aec" and not degraded_mode and not headset_clean_mode:
        cleaned = mic.astype(np.int16, copy=False)
        residual_level = float(mic_level)
        predicted_level = 0.0
        improvement_ratio = 0.0
        similarity = min(float(similarity), 0.08 if probe_similarity >= 0.2 else float(similarity))
        best_shift = 0
        selection_reason = "degraded_fallback"
    quality = float(
        max(
            0.0,
            min(
                1.0,
                similarity
                * improvement_ratio
                * min(1.0, predicted_level / max(ref_level, 1e-4) * 1.2),
            ),
        )
    )
    owner._update_delay_tracker(best_shift, similarity)
    owner._last_double_talk = bool(double_talk)
    return {
        "pcm": cleaned.tobytes(),
        "similarity": float(similarity),
        "delay_samples": int(best_shift),
        "double_talk": bool(double_talk),
        "residual_level": float(residual_level),
        "mic_level": float(mic_level),
        "aec_quality": quality,
        "predicted_level": float(predicted_level),
        "improvement_ratio": float(improvement_ratio),
        "backend": backend_used,
        "backend_policy": aec_mode,
        "webrtc_success": bool(webrtc_success),
        "native_attempted": bool(native_attempted),
        "native_selected": bool(native_selected),
        "selection_reason": selection_reason,
        "voice_likelihood": voice_likelihood,
    }
