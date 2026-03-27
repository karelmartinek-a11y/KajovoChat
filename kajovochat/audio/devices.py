from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy import signal

from .dsp_helpers import _find_best_alignment, _rms

@dataclass
class AudioCalibrationResult:
    input_device: Optional[int]
    output_device: Optional[int]
    ambient_rms: float
    playback_rms: float
    bleed_ratio: float
    similarity: float
    recommended_profile: dict[str, float]
    notes: list[str]
    latency_samples: int = 0
    preferred_frame_size: int = 480
    filter_length: int = 256
    device_fingerprint: str = "unknown"
    audio_mode: str = "notebook_builtin"

def list_audio_devices() -> dict:
    """List audio devices for UI selection.

    This is best-effort: if enumeration fails, returns empty lists.

    Returns:
        {"inputs": [{"index": int, "name": str, "max_channels": int}],
         "outputs": [{"index": int, "name": str, "max_channels": int}]}
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return {"inputs": [], "outputs": []}

    inputs = []
    outputs = []
    for idx, d in enumerate(devices or []):
        name = str(d.get("name", f"Device {idx}"))
        mi = int(d.get("max_input_channels", 0) or 0)
        mo = int(d.get("max_output_channels", 0) or 0)
        if mi > 0:
            inputs.append({"index": idx, "name": name, "max_channels": mi})
        if mo > 0:
            outputs.append({"index": idx, "name": name, "max_channels": mo})

    return {"inputs": inputs, "outputs": outputs}


def _device_valid(index: Optional[int], kind: str) -> bool:
    if index is None:
        return True
    try:
        info = sd.query_devices(index, kind)
    except Exception:
        return False
    if kind == "input":
        return int(info.get("max_input_channels", 0) or 0) > 0
    if kind == "output":
        return int(info.get("max_output_channels", 0) or 0) > 0
    return False


def _score_name(name: str, kind: str) -> int:
    """Heuristic scoring to prefer built-in laptop mic/speakers.

    This is intentionally conservative and cross-platform-ish.
    """
    n = (name or "").lower()
    score = 0
    # Built-in / internal tends to be what users want for "NB mic/speakers".
    if any(k in n for k in ["built-in", "builtin", "internal", "integro", "notebook", "laptop"]):
        score += 40
    if kind == "input":
        if any(k in n for k in ["microphone", "mic", "array", "input"]):
            score += 25
        if any(k in n for k in ["usb", "webcam", "camera"]):
            score -= 10  # many users don't want these by default
    else:
        if any(k in n for k in ["speaker", "speakers", "output", "headphone", "headphones"]):
            score += 25
        if any(k in n for k in ["bluetooth", "bt"]):
            score -= 5

    # Common Windows drivers for internal audio
    if any(k in n for k in ["realtek", "conexant", "intel"]):
        score += 8
    # Avoid obvious "monitor"/"virtual"/"loopback" devices.
    if any(k in n for k in ["loopback", "virtual", "monitor", "cable", "vb-audio", "blackhole"]):
        score -= 30
    return score


def pick_audio_device(kind: str, preferred: Optional[int]) -> tuple[Optional[int], str]:
    """Pick a usable device index.

    Order:
      1) preferred (if valid)
      2) system default (if valid)
      3) best-effort heuristic match (built-in mic/speakers)

    Returns: (device_index_or_None, note)
    """
    kind = "input" if kind == "input" else "output"

    if preferred is not None and _device_valid(preferred, kind):
        return int(preferred), "selected:settings"

    # sounddevice default is either a scalar or a (in,out) pair.
    try:
        default = sd.default.device
        if isinstance(default, (list, tuple)) and len(default) >= 2:
            default_idx = default[0] if kind == "input" else default[1]
        else:
            default_idx = default
        if default_idx is not None and int(default_idx) >= 0 and _device_valid(int(default_idx), kind):
            return int(default_idx), "selected:system_default"
    except Exception:
        pass

    try:
        devices = sd.query_devices() or []
    except Exception:
        return None, "selected:none"

    best_idx: Optional[int] = None
    best_score = -10**9
    for idx, d in enumerate(devices):
        name = str(d.get("name", ""))
        mi = int(d.get("max_input_channels", 0) or 0)
        mo = int(d.get("max_output_channels", 0) or 0)
        if kind == "input" and mi <= 0:
            continue
        if kind == "output" and mo <= 0:
            continue
        s = _score_name(name, kind)
        # Slightly prefer devices that look like "default" in name.
        if "default" in name.lower():
            s += 5
        if s > best_score:
            best_score = s
            best_idx = idx

    if best_idx is not None and _device_valid(best_idx, kind):
        return int(best_idx), "selected:heuristic"
    return None, "selected:none"


def format_device_help() -> str:
    """User-facing device dump for error messages."""
    devs = list_audio_devices()
    lines = ["Dostupná audio zařízení (index: název):"]
    ins = devs.get("inputs", [])
    outs = devs.get("outputs", [])
    if ins:
        lines.append("Vstupy:")
        for d in ins[:30]:
            lines.append(f"  {d['index']}: {d['name']}")
    else:
        lines.append("Vstupy: (nenalezeno)")
    if outs:
        lines.append("Výstupy:")
        for d in outs[:30]:
            lines.append(f"  {d['index']}: {d['name']}")
    else:
        lines.append("Výstupy: (nenalezeno)")
    lines.append("Tip: Aplikace používá systémová výchozí zařízení, případně interní heuristiku pro vestavěný mikrofon a reproduktory.")
    return "\n".join(lines)

def _device_name(index: Optional[int], kind: str) -> str:
    try:
        if index is None:
            return "default"
        info = sd.query_devices(index, kind)
        return str(info.get("name", f"{kind}:{index}"))
    except Exception:
        return f"{kind}:{index if index is not None else 'default'}"


def _infer_audio_mode(input_name: str, output_name: str) -> str:
    combined = f"{input_name} {output_name}".lower()
    if any(token in combined for token in ("bluetooth", "airpods", "buds", "hands-free")):
        return "bluetooth_headset"
    if any(token in combined for token in ("headphone", "headphones", "headset", "earbuds", "usb audio")):
        return "wired_headset"
    if any(token in combined for token in ("speaker", "speakers", "monitor", "hdmi", "display audio", "dock")) and not any(
        token in combined for token in ("built-in", "builtin", "internal", "laptop", "notebook")
    ):
        return "external_speakers"
    return "notebook_builtin"


def _build_device_fingerprint(input_device: Optional[int], output_device: Optional[int], samplerate: int) -> str:
    payload = "|".join([
        _device_name(input_device, "input"),
        _device_name(output_device, "output"),
        str(int(samplerate)),
    ])
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_device_fingerprint(input_device: Optional[int], output_device: Optional[int], samplerate: int = 24000) -> str:
    """Vrati stabilni fingerprint aktualni dvojice vstup/vystup."""
    return _build_device_fingerprint(input_device, output_device, samplerate)

def _playrec_with_fallbacks(
    playback_buffer: np.ndarray,
    *,
    samplerate: int,
    input_device: Optional[int],
    output_device: Optional[int],
) -> np.ndarray:
    attempts = [
        {"device": (input_device, output_device)},
        {"device": (None, output_device)},
        {"device": None},
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return np.asarray(
                sd.playrec(
                    playback_buffer.reshape(-1, 1),
                    samplerate=samplerate,
                    channels=1,
                    dtype="float32",
                    device=attempt["device"],
                    blocking=True,
                ),
                dtype=np.float32,
            ).reshape(-1)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Audio kalibraci se nepodařilo spustit.")


def calibrate_audio_devices(
    *,
    input_device: Optional[int],
    output_device: Optional[int],
    samplerate: int = 24000,
    playback_seconds: float = 1.8,
    playback_gain: float = 0.28,
) -> AudioCalibrationResult:
    """Automaticky změří bleed mezi reproduktorem a mikrofonem a navrhne guard profil."""
    samplerate = int(samplerate)
    ambient_frames = max(1, int(samplerate * 0.45))
    playback_frames = max(1, int(samplerate * max(1.2, playback_seconds)))
    total_frames = ambient_frames + playback_frames

    probe_time = np.linspace(0.0, playback_frames / samplerate, playback_frames, endpoint=False, dtype=np.float32)
    chirp = signal.chirp(probe_time, f0=180.0, f1=4200.0, t1=max(0.1, playback_seconds), method="logarithmic")
    envelope = np.hanning(playback_frames).astype(np.float32)
    pulse = np.sin(2.0 * np.pi * 42.0 * probe_time).astype(np.float32)
    playback_signal = ((chirp * 0.72) + (pulse * 0.28)) * envelope * float(playback_gain)
    playback_buffer = np.concatenate(
        [
            np.zeros((ambient_frames,), dtype=np.float32),
            playback_signal.astype(np.float32),
        ]
    )

    recorded = _playrec_with_fallbacks(
        playback_buffer,
        samplerate=samplerate,
        input_device=input_device,
        output_device=output_device,
    )
    if recorded.size < total_frames:
        padded = np.zeros((total_frames,), dtype=np.float32)
        padded[: recorded.size] = recorded
        recorded = padded

    ambient = recorded[:ambient_frames]
    captured = recorded[ambient_frames : ambient_frames + playback_frames]

    ambient_rms = _rms(ambient)
    playback_rms = _rms(captured)
    bleed_ratio = float(playback_rms / max(ambient_rms, 1e-4))
    _aligned_segment, similarity, latency_samples = _find_best_alignment(
        captured.astype(np.float32),
        playback_signal.astype(np.float32),
        max_shift_samples=int(samplerate * 0.12),
    )
    input_name = _device_name(input_device, "input")
    output_name = _device_name(output_device, "output")
    audio_mode = _infer_audio_mode(input_name, output_name)
    filter_length = int(np.clip(256 + max(0, latency_samples) * 2, 256, 2048))
    preferred_frame_size = 960 if audio_mode in {"notebook_builtin", "external_speakers"} else 480

    recommended_profile = {
        "server_vad_threshold": float(np.clip(0.72 + max(0.0, similarity - 0.45) * 0.18 + max(0.0, min(0.18, ambient_rms)) * 0.55, 0.68, 0.9)),
        "playback_activity_level": float(np.clip(max(0.028, playback_rms * 0.45), 0.028, 0.16)),
        "echo_similarity_drop": float(np.clip(0.78 + similarity * 0.14, 0.78, 0.96)),
        "echo_similarity_soft": float(np.clip(0.6 + similarity * 0.12, 0.58, 0.86)),
        "barge_in_min_input_level": float(np.clip(max(0.05, ambient_rms * 4.2, playback_rms * 0.3), 0.05, 0.22)),
        "barge_in_output_ratio": float(np.clip(1.2 + min(bleed_ratio, 8.0) * 0.06 + (0.08 if audio_mode == "notebook_builtin" else 0.0), 1.18, 1.9)),
    }
    if audio_mode in {"wired_headset", "bluetooth_headset"}:
        recommended_profile["echo_similarity_drop"] = float(np.clip(recommended_profile["echo_similarity_drop"] - 0.08, 0.68, 0.96))
        recommended_profile["echo_similarity_soft"] = float(np.clip(recommended_profile["echo_similarity_soft"] - 0.06, 0.54, 0.86))
        recommended_profile["playback_activity_level"] = float(np.clip(recommended_profile["playback_activity_level"] * 0.82, 0.02, 0.12))
    elif audio_mode in {"external_speakers", "notebook_builtin"}:
        recommended_profile["server_vad_threshold"] = float(np.clip(recommended_profile["server_vad_threshold"] + 0.01, 0.68, 0.9))
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"ambient_rms={ambient_rms:.4f}",
        f"playback_rms={playback_rms:.4f}",
        f"bleed_ratio={bleed_ratio:.2f}",
        f"similarity={similarity:.3f}",
        f"latency_samples={latency_samples}",
        f"audio_mode={audio_mode}",
        f"frame_size={preferred_frame_size}",
    ]
    return AudioCalibrationResult(
        input_device=input_device,
        output_device=output_device,
        ambient_rms=ambient_rms,
        playback_rms=playback_rms,
        bleed_ratio=bleed_ratio,
        similarity=similarity,
        recommended_profile=recommended_profile,
        notes=notes,
        latency_samples=int(latency_samples),
        preferred_frame_size=int(preferred_frame_size),
        filter_length=int(filter_length),
        device_fingerprint=_build_device_fingerprint(input_device, output_device, samplerate),
        audio_mode=audio_mode,
    )


def calibrate_audio_devices_advanced(
    *,
    input_device: Optional[int],
    output_device: Optional[int],
    samplerate: int = 24000,
) -> AudioCalibrationResult:
    """Víceprůchodová kalibrace s různou délkou a hlasitostí testovacího signálu."""
    samplerates = [int(samplerate), 48000, 44100]
    passes = [
        {"playback_seconds": 1.4, "playback_gain": 0.18},
        {"playback_seconds": 1.8, "playback_gain": 0.24},
        {"playback_seconds": 2.2, "playback_gain": 0.30},
    ]
    results: list[AudioCalibrationResult] = []
    errors: list[str] = []
    for current_samplerate in samplerates:
        for current in passes:
            try:
                results.append(
                    calibrate_audio_devices(
                        input_device=input_device,
                        output_device=output_device,
                        samplerate=int(current_samplerate),
                        playback_seconds=float(current["playback_seconds"]),
                        playback_gain=float(current["playback_gain"]),
                    )
                )
            except Exception as exc:
                errors.append(f"{current_samplerate}Hz/{current['playback_gain']:.2f}: {exc}")

    if not results:
        raise RuntimeError("Pokročilá audio kalibrace selhala: " + " | ".join(errors[:4]))

    if len(results) > 4:
        # Drž jen reprezentativní podmnožinu, aby nebyl profil přehnaně rozkolísaný.
        results = results[:4]

    ambient_values = np.asarray([item.ambient_rms for item in results], dtype=np.float32)
    playback_values = np.asarray([item.playback_rms for item in results], dtype=np.float32)
    bleed_values = np.asarray([item.bleed_ratio for item in results], dtype=np.float32)
    similarity_values = np.asarray([item.similarity for item in results], dtype=np.float32)

    recommended_profile = {
        "server_vad_threshold": float(max(item.recommended_profile["server_vad_threshold"] for item in results)),
        "playback_activity_level": float(np.median([item.recommended_profile["playback_activity_level"] for item in results])),
        "echo_similarity_drop": float(max(item.recommended_profile["echo_similarity_drop"] for item in results)),
        "echo_similarity_soft": float(np.median([item.recommended_profile["echo_similarity_soft"] for item in results])),
        "barge_in_min_input_level": float(max(item.recommended_profile["barge_in_min_input_level"] for item in results)),
        "barge_in_output_ratio": float(max(item.recommended_profile["barge_in_output_ratio"] for item in results)),
    }
    positive_latencies = [int(getattr(item, "latency_samples", 0) or 0) for item in results if int(getattr(item, "latency_samples", 0) or 0) > 0]
    if positive_latencies:
        latency_samples = int(round(float(np.median(positive_latencies))))
    else:
        best_alignment = max(
            results,
            key=lambda item: (
                float(getattr(item, "similarity", 0.0) or 0.0),
                float(getattr(item, "bleed_ratio", 0.0) or 0.0),
                float(getattr(item, "playback_rms", 0.0) or 0.0),
            ),
        )
        candidate_latency = int(getattr(best_alignment, "latency_samples", 0) or 0)
        latency_samples = candidate_latency if candidate_latency > 0 and float(getattr(best_alignment, "similarity", 0.0) or 0.0) >= 0.18 else 0
    preferred_frame_size = int(round(float(np.median([getattr(item, "preferred_frame_size", 480) for item in results]))))
    if latency_samples > 0:
        filter_length = int(np.clip(256 + max(0, latency_samples) * 2, 256, 2048))
    else:
        filter_length = int(round(float(np.median([getattr(item, "filter_length", 256) for item in results]))))
    mode_counts: dict[str, int] = {}
    for item in results:
        mode = getattr(item, "audio_mode", "notebook_builtin")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    audio_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "notebook_builtin"
    device_fingerprint = getattr(results[0], "device_fingerprint", "unknown")
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"passes={len(results)}",
        f"ambient_med={float(np.median(ambient_values)):.4f}",
        f"playback_med={float(np.median(playback_values)):.4f}",
        f"bleed_peak={float(np.max(bleed_values)):.2f}",
        f"similarity_peak={float(np.max(similarity_values)):.3f}",
        f"latency_med={latency_samples}",
        f"audio_mode={audio_mode}",
        f"frame_size={preferred_frame_size}",
    ]
    return AudioCalibrationResult(
        input_device=input_device,
        output_device=output_device,
        ambient_rms=float(np.median(ambient_values)),
        playback_rms=float(np.median(playback_values)),
        bleed_ratio=float(np.max(bleed_values)),
        similarity=float(np.max(similarity_values)),
        recommended_profile=recommended_profile,
        notes=notes,
        latency_samples=int(latency_samples),
        preferred_frame_size=int(preferred_frame_size),
        filter_length=int(filter_length),
        device_fingerprint=device_fingerprint,
        audio_mode=audio_mode,
    )
