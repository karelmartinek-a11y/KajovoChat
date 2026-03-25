from __future__ import annotations

import sys
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import sounddevice as sd

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QLineEdit,
    QStyle,
)

from .settings import (
    AppSettings,
    DEFAULT_AUDIO_GUARD_PROFILE,
    normalize_audio_aec_mode,
    normalize_audio_device_mode,
)
from .audio.voice_gate import backend_aware_aec_metrics as _backend_aware_aec_metrics
from .audio.bootstrap import build_conversation_audio_stack
from .audio.runtime_resources import AudioRuntimeResources
from .audio.selftest import run_audio_guard_selftest as _run_audio_guard_selftest_impl
from .dialogs.settings_dialog import SettingsDialog
from .audio.devices import (
    build_device_fingerprint,
    calibrate_audio_devices_advanced,
    format_device_help,
    list_audio_devices,
    pick_audio_device,
)
from .audio.dsp_helpers import AdaptiveEchoCanceller, suppress_echo_from_pcm16
from .audio.io import DuplexAudioSession
from .audio.windows_system_aec import probe_windows_system_aec
from .services.realtime_service import RealtimeService
from .services.log_service import RealtimeLogWriter
from .services.app_logging import install_app_logging
from .services.guard_adaptation import GuardAdaptor
from .services.guard_replay import append_guard_replay_metrics
from .services.guard_telemetry import GuardTelemetry
from .services.voice_features import estimate_voice_likelihood_from_pcm16
from .resources.assets import verify_asset_manifest
from .widgets.head_widget import HeadWidget
from .theme import Theme, app_stylesheet


_ALLOWED_LANGS = {"cs", "en", "de", "sk", "fr"}
_STATE_IDLE = "idle"
_STATE_CONNECTING = "connecting"
_STATE_LISTENING = "listening"
_STATE_TRANSCRIBING = "transcribing"
_STATE_THINKING = "thinking"
_STATE_SPEAKING = "speaking"
_STATE_RECONNECTING = "reconnecting"
_STATE_ERROR = "error"
_VALID_STATES = {
    _STATE_IDLE,
    _STATE_CONNECTING,
    _STATE_LISTENING,
    _STATE_TRANSCRIBING,
    _STATE_THINKING,
    _STATE_SPEAKING,
    _STATE_RECONNECTING,
    _STATE_ERROR,
}

_REALTIME_MODEL = "gpt-realtime"
_TTS_VOICE = "alloy"
_TTS_SPEED = 1.0
_NOISE_REDUCTION = "far_field"
_SERVER_VAD_SILENCE_MS = 900
_SERVER_VAD_PREFIX_MS = 300
_SERVER_VAD_THRESHOLD = 0.72
_PLAYBACK_ACTIVITY_LEVEL = 0.035
_ECHO_TRAILING_HOLD_S = 0.28
_ECHO_SIMILARITY_DROP = 0.82
_ECHO_SIMILARITY_SOFT = 0.68
_BARGE_IN_MIN_INPUT_LEVEL = 0.06
_BARGE_IN_OUTPUT_RATIO = 1.35


def _audio_guard_profile(settings: AppSettings) -> dict[str, float]:
    try:
        return settings.normalized_audio_guard_profile()
    except Exception:
        return dict(DEFAULT_AUDIO_GUARD_PROFILE)


def _closed_pose_snapshot() -> dict[str, object]:
    return {
        "pose": "closed",
        "openness": 0.0,
        "energy": 0.0,
        "weights": {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0},
    }


def _sanitize_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_OPENAI_KEY]", text)
    return text


def _pcm16_echo_similarity(mic_pcm: bytes, reference: object) -> float:
    try:
        import numpy as _np

        mic = _np.frombuffer(mic_pcm, dtype=_np.int16).astype(_np.float32)
        ref = _np.asarray(reference, dtype=_np.int16).astype(_np.float32).reshape(-1)
        if mic.size < 120 or ref.size < mic.size:
            return 0.0

        mic = mic - float(_np.mean(mic))
        mic_norm = float(_np.linalg.norm(mic) + 1e-6)
        if mic_norm <= 1e-6:
            return 0.0

        best = 0.0
        max_shift = min(max(0, ref.size - mic.size), 960)
        for shift in range(0, max_shift + 1, 120):
            segment = ref[ref.size - mic.size - shift : ref.size - shift if shift > 0 else ref.size]
            if segment.size != mic.size:
                continue
            segment = segment - float(_np.mean(segment))
            seg_norm = float(_np.linalg.norm(segment) + 1e-6)
            if seg_norm <= 1e-6:
                continue
            corr = abs(float(_np.dot(mic, segment)) / (mic_norm * seg_norm))
            if corr > best:
                best = corr
        return float(max(0.0, min(1.0, best)))
    except Exception:
        return 0.0



def run_audio_guard_selftest() -> dict[str, object]:
    return _run_audio_guard_selftest_impl(
        default_profile=DEFAULT_AUDIO_GUARD_PROFILE,
        sanitize_text_fn=_sanitize_text,
        pick_audio_device_fn=pick_audio_device,
        list_audio_devices_fn=list_audio_devices,
        calibrate_audio_devices_advanced_fn=calibrate_audio_devices_advanced,
    )


class ConversationWorker(QObject):
    """Realtime speech-to-speech conversation (WebSocket).

    The UI has two modes:
    - Hands-free: continuous mic streaming, server-side VAD triggers responses.
    - Push-to-talk: mic streams only while button is pressed; on release we commit+response.
    """

    state_changed = Signal(str)        # idle/listening/transcribing/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message
    guard_debug_updated = Signal(object)

    # Realtime levely pro animaci hlavy (0..1).
    input_level = Signal(float)
    output_level = Signal(float)
    output_pose = Signal(object)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._guard_profile = _audio_guard_profile(settings)
        self._guard_calibration = dict(settings.audio_guard_calibration or {})
        self._guard_telemetry = GuardTelemetry()
        self._guard_adaptor = GuardAdaptor()
        self._guard_last_adapt_at = 0.0
        self._guard_learning_until = 0.0
        self._guard_aec_aware = False
        self._aec_mode = normalize_audio_aec_mode(self.settings.audio_aec_mode)
        self._native_aec_probe = probe_windows_system_aec()
        self._session_log_dir: Optional[Path] = None
        self._session_name: str = ""
        self._input_device_name = "default"
        self._output_device_name = "default"
        self._audio_mode = "notebook_builtin"

        self._stop_all = threading.Event()

        self._captions = ""
        self._logger: Optional[RealtimeLogWriter] = None
        self._runtime_resources = AudioRuntimeResources()
        self._aec = AdaptiveEchoCanceller(samplerate=24000)
        self._resolved_input_device: Optional[int] = None
        self._resolved_output_device: Optional[int] = None

        self._mode: str = "idle"  # "handsfree" | "ptt" | "idle"
        # Level signals are throttled to avoid saturating the Qt event loop.
        self._last_in_level: float = 0.0
        self._last_out_level: float = 0.0
        self._last_level_emit_t: float = 0.0

        # True while waiting for server transcription completion.

        # Best-effort current UI state.
        self._ui_state = _STATE_IDLE

        audio_stack = build_conversation_audio_stack(
            self,
            estimate_voice_likelihood_from_pcm16=estimate_voice_likelihood_from_pcm16,
            backend_aware_aec_metrics=_backend_aware_aec_metrics,
            closed_pose_factory=_closed_pose_snapshot,
            sanitize_text_fn=_sanitize_text,
            format_device_help_fn=format_device_help,
            allowed_langs=_ALLOWED_LANGS,
            realtime_model=_REALTIME_MODEL,
            tts_voice=_TTS_VOICE,
            noise_reduction=_NOISE_REDUCTION,
            tts_speed=_TTS_SPEED,
            server_vad_silence_ms=_SERVER_VAD_SILENCE_MS,
            server_vad_prefix_ms=_SERVER_VAD_PREFIX_MS,
            server_vad_threshold=_SERVER_VAD_THRESHOLD,
            state_speaking=_STATE_SPEAKING,
            state_error=_STATE_ERROR,
            state_idle=_STATE_IDLE,
            echo_trailing_hold_s=_ECHO_TRAILING_HOLD_S,
            normalize_aec_mode=normalize_audio_aec_mode,
        )
        self._rt_runtime_controller = audio_stack.runtime_controller
        self._audio_lifecycle = audio_stack.lifecycle
        self._audio_observer = audio_stack.observer
        self._audio_policy = audio_stack.policy
        self._audio_controls = audio_stack.controls
        self._session_manager = audio_stack.session_manager


    @Slot()
    def reload_guard_profile(self) -> None:
        self._guard_profile = _audio_guard_profile(self.settings)
        self._emit_guard_debug()

    def _emit_guard_debug(self) -> None:
        self._audio_observer.emit_guard_debug()

    def _record_aec_diag_sample(
        self,
        *,
        residual_level: float,
        aec_quality: float,
        double_talk: bool,
        delay_samples: int,
        similarity: float,
        reference_miss: bool,
    ) -> None:
        self._audio_observer.record_aec_diag_sample(
            residual_level=residual_level,
            aec_quality=aec_quality,
            double_talk=double_talk,
            delay_samples=delay_samples,
            similarity=similarity,
            reference_miss=reference_miss,
        )

    def _build_aec_summary(self) -> dict[str, float]:
        return self._audio_observer.build_aec_summary()

    def _set_state(self, s: str) -> None:
        self._audio_observer.set_ui_state(s, valid_states=_VALID_STATES, idle_state=_STATE_IDLE)

    @property
    def ui_state(self) -> str:
        return self._ui_state

    @staticmethod
    def _pcm16_level(pcm: bytes) -> float:
        """Quick 0..1 loudness estimate from PCM16 mono bytes."""
        if not pcm:
            return 0.0
        try:
            import numpy as _np

            x = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32)
            if x.size == 0:
                return 0.0
            x = x / 32768.0
            rms = float(_np.sqrt(_np.mean(x * x) + 1e-12))
            peak = float(_np.max(_np.abs(x)))
            lvl = max(rms * 2.2, peak * 1.1)
            return float(max(0.0, min(1.0, lvl)))
        except Exception:
            return 0.0

    def _preferred_frame_size(self) -> int:
        value = int(self._guard_calibration.get("preferred_frame_size", 480) or 480)
        return max(240, min(1920, value))

    def _current_device_fingerprint(self) -> str:
        return self._audio_policy.current_device_fingerprint()

    def _configure_aec_from_calibration(self) -> None:
        self._audio_policy.configure_aec_from_calibration()

    def _consider_runtime_latency_update(
        self,
        *,
        delay_samples: int,
        similarity: float,
        aec_quality: float,
        improvement_ratio: float,
        backend: str,
        double_talk: bool,
        prefer_webrtc: bool = False,
    ) -> None:
        self._audio_policy.consider_runtime_latency_update(
            delay_samples=delay_samples,
            similarity=similarity,
            aec_quality=aec_quality,
            improvement_ratio=improvement_ratio,
            backend=backend,
            double_talk=double_talk,
            prefer_webrtc=prefer_webrtc,
        )

    def _reference_needed_samples(self, chunk_bytes: int) -> int:
        return self._audio_policy.reference_needed_samples(chunk_bytes)

    def _ensure_player(self) -> None:
        self._audio_policy.ensure_player()

    def _active_duplex(self) -> Optional[DuplexAudioSession]:
        return self._audio_policy.active_duplex()

    def _device_calibration_matches(self) -> bool:
        return self._audio_policy.device_calibration_matches()

    def _apply_device_class_policy(self) -> None:
        self._audio_policy.apply_device_class_policy()

    def _apply_aec_mode_policy(self) -> None:
        self._audio_policy.apply_aec_mode_policy()

    def _apply_saved_calibration(self) -> None:
        self._audio_policy.apply_saved_calibration()

    def _resolve_audio_devices(self) -> None:
        self._audio_policy.resolve_audio_devices()

    def _append_caption(self, line: str) -> None:
        self._audio_observer.append_caption(line)

    def _log_event(self, record_type: str, **extra) -> None:
        self._audio_observer.log_event(record_type, **extra)

    def _log_conversation_text(self, record_type: str, text: str) -> None:
        self._audio_observer.log_conversation_text(record_type, text)

    def _set_caption_preview(self, prefix: str, text: str) -> None:
        self._audio_observer.set_caption_preview(prefix, text)

    def _start_session_if_needed(self) -> None:
        self._audio_lifecycle.start_session_if_needed()

    def _start_rt_loop(self) -> None:
        self._rt_runtime_controller.start()

    def _stop_rt_loop(self, timeout_s: float = 1.0) -> None:
        self._rt_runtime_controller.stop(timeout_s=timeout_s)

    def _end_session(self) -> None:
        self._audio_lifecycle.end_session()

    def _handle_user_transcript(self, text: str) -> None:
        self._session_manager.handle_user_transcript(text)

    def _handle_assistant_done(self, text: str) -> None:
        self._session_manager.handle_assistant_done(text)

    def _handle_assistant_audio(self, pcm: bytes) -> None:
        self._session_manager.handle_assistant_audio(pcm)

    def _handle_speech_started(self) -> None:
        self._session_manager.handle_speech_started()

    def _handle_speech_stopped(self) -> None:
        self._session_manager.handle_speech_stopped()

    def _handle_response_done(self) -> None:
        self._session_manager.handle_response_done()

    @Slot()
    def request_stop(self) -> None:
        self._audio_controls.request_stop()

    # -------- Hands-free mode --------

    @Slot()
    def start_handsfree(self) -> None:
        self._audio_controls.start_handsfree()


    @Slot()
    def ptt_pressed(self) -> None:
        self._audio_controls.ptt_pressed()


    @Slot()
    def ptt_released(self) -> None:
        self._audio_controls.ptt_released()


class MainWindow(QMainWindow):
    sig_start_handsfree = Signal()
    sig_request_stop = Signal()
    sig_reload_guard_profile = Signal()

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._handsfree_running = False

        self._thread = QThread(self)
        self.worker = ConversationWorker(self.settings)
        self.worker.moveToThread(self._thread)
        self._thread.start()

        self.sig_start_handsfree.connect(self.worker.start_handsfree)
        self.sig_request_stop.connect(self.worker.request_stop)
        self.sig_reload_guard_profile.connect(self.worker.reload_guard_profile)

        self._theme = Theme()

        self.setWindowTitle("Chatbot Kája")
        try:
            assets_dir = Path(__file__).resolve().parent / "resources" / "assets"
            icon_path = assets_dir / "logo_chatbot_kaja.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass

        self._build_ui()
        self._wire()
        self._sync_start_stop_button()
        QTimer.singleShot(0, self._report_render_backend)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout()
        outer.setContentsMargins(18, 14, 18, 16)
        outer.setSpacing(14)
        root.setLayout(outer)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        logo = QLabel()
        logo.setFixedSize(56, 56)
        try:
            assets_dir = Path(__file__).resolve().parent / "resources" / "assets"
            logo_path = assets_dir / "logo_chatbot_kaja.png"
            if logo_path.exists():
                pm = QPixmap(str(logo_path))
                logo.setPixmap(pm.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(0)
        title = QLabel("Chatbot Kája")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"QLabel {{ color: {self._theme.text}; }}")
        subtitle = QLabel("Hlasový asistent s hands-free EKG vizualizací")
        subtitle.setStyleSheet(f"QLabel {{ color: {self._theme.text_muted}; font-size: 12px; }}")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)

        header.addWidget(logo)
        header.addSpacing(12)
        header.addLayout(title_wrap)
        header.addStretch(1)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("OpenAI API klíč")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setFixedWidth(250)
        if self.settings.openai_api_key:
            self.api_key_input.setText(self.settings.openai_api_key)

        self.btn_save_key = QPushButton("Uložit klíč")
        self.btn_delete_key = QPushButton("Smazat klíč")
        self.btn_audio_test = QPushButton("Audio test")
        self.btn_audio_test.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.btn_start_stop = QPushButton("Start")
        self.btn_settings = QPushButton("Nastavení")
        self.btn_save = QPushButton("Uložit")
        self.btn_clear = QPushButton("Vyčistit relaci")
        self.btn_exit = QPushButton("Konec")

        self.btn_start_stop.setProperty("variant", "primary")
        self.btn_settings.setProperty("variant", "primary")
        self.btn_exit.setProperty("variant", "danger")

        header.addWidget(self.api_key_input)
        header.addWidget(self.btn_save_key)
        header.addWidget(self.btn_delete_key)
        header.addWidget(self.btn_audio_test)
        header.addWidget(self.btn_start_stop)
        header.addWidget(self.btn_settings)
        header.addWidget(self.btn_save)
        header.addWidget(self.btn_clear)
        header.addSpacing(8)
        header.addWidget(self.btn_exit)

        outer.addLayout(header)

        self.status_label = QLabel("Připraveno")
        self.status_label.setStyleSheet(
            "QLabel {"
            "  padding: 10px 14px;"
            "  border-radius: 12px;"
            "  background-color: rgba(255,255,255,8);"
            "  border: 1px solid rgba(255,255,255,16);"
            "  font-size: 12px;"
            "}"
        )
        outer.addWidget(self.status_label)

        self.guard_debug_label = QLabel("")
        self.guard_debug_label.setStyleSheet(
            "QLabel {"
            "  padding: 10px 12px;"
            "  border-radius: 12px;"
            "  background-color: rgba(7,22,18,180);"
            "  border: 1px solid rgba(124,255,141,45);"
            "  color: #9BFFC3;"
            "  font-family: Consolas;"
            "  font-size: 11px;"
            "}"
        )
        self.guard_debug_label.setText(
            "GUARD: čekám na telemetrii\n"
            "stav=-  drop_rate=0.000  similarity=0.000  voice=0.000\n"
            "echo_soft=-  echo_drop=-  barge_in=-  playback=-\n"
            "mode=notebook_builtin  aec=off  learning=off"
        )
        outer.addWidget(self.guard_debug_label)

        head_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "head_photo.png")
        self.head = HeadWidget(head_path)
        self.head.setMinimumSize(520, 520)
        outer.addWidget(self.head, 1)

        if not self.settings.openai_api_key:
            self.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte.")

    def _wire(self) -> None:
        self.btn_exit.clicked.connect(lambda _=False: self.close())
        self.btn_save_key.clicked.connect(lambda _=False: self._save_api_key())
        self.btn_delete_key.clicked.connect(lambda _=False: self._delete_api_key())
        self.btn_audio_test.clicked.connect(lambda _=False: self._run_audio_test())
        self.btn_start_stop.clicked.connect(lambda _=False: self._toggle_handsfree())
        self.btn_settings.clicked.connect(lambda _=False: self._open_settings_dialog())
        self.btn_save.clicked.connect(lambda _=False: self._save_defaults())
        self.btn_clear.clicked.connect(lambda _=False: self._clear_session())

        self.worker.state_changed.connect(self._on_state)
        self.worker.captions_updated.connect(self._on_captions)
        self.worker.error.connect(self._on_error)
        self.worker.input_level.connect(self._on_input_level)
        self.worker.output_level.connect(self._on_output_level)
        self.worker.output_pose.connect(self._on_output_pose)
        self.worker.guard_debug_updated.connect(self._on_guard_debug)

    def _report_render_backend(self) -> None:
        try:
            summary = self.head.render_backend_summary()
        except Exception:
            return
        prefix = "Renderer: GPU" if self.head.is_gpu_renderer_active() else "Renderer: fallback"
        self._append_terminal_line(f"SYS: {prefix} | {summary}")

    def _open_settings_dialog(self) -> None:
        try:
            d = SettingsDialog(self.settings, parent=self)
            if d.exec():
                d.apply()
                self.settings.save()
        except Exception:
            import logging

            logging.getLogger("kajovochat").exception("settings_dialog_failed")
            try:
                QMessageBox.critical(self, "Nastavení", "Nepodařilo se otevřít nastavení. Podrobnosti jsou v logu.")
            except Exception:
                pass

    def _save_api_key(self) -> None:
        key = self.api_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "API klíč", "Nejdřív zadejte OpenAI API klíč.")
            return
        self.settings.openai_api_key = key
        self.settings.save()
        self._append_terminal_line("SYS: OpenAI API klíč byl bezpečně uložen lokálně.")
        QMessageBox.information(
            self,
            "API klíč",
            "Klíč byl uložen lokálně. Ve Windows se chrání přes DPAPI, na ostatních platformách přes systémový keyring, pokud je dostupný.",
        )

    def _delete_api_key(self) -> None:
        self.api_key_input.clear()
        self.settings.openai_api_key = ""
        self.settings.save()
        self._append_terminal_line("SYS: OpenAI API klíč byl smazán.")
        QMessageBox.information(self, "API klíč", "OpenAI API klíč byl smazán.")

    def _apply_audio_profile(self, profile: dict[str, float], *, source: str) -> None:
        self.settings.audio_guard_profile = {
            key: float(value)
            for key, value in profile.items()
            if key in DEFAULT_AUDIO_GUARD_PROFILE
        }
        self.settings.audio_guard_profile = self.settings.normalized_audio_guard_profile()
        self.settings.save()
        self.sig_reload_guard_profile.emit()
        self._append_terminal_line(
            f"SELFTEST APPLY [{source}]: "
            + ", ".join(f"{key}={value:.3f}" for key, value in self.settings.audio_guard_profile.items())
        )

    def _calibrate_audio_guard(self, *, trigger: str, show_dialog: bool, restart_after: bool) -> dict[str, object]:
        self.status_label.setText("Probíhá automatická kalibrace reproduktoru a mikrofonu.")
        self._append_terminal_line(f"SELFTEST START [{trigger}]")
        result = run_audio_guard_selftest()

        lines = []
        for item in result["checks"]:
            mark = "OK" if item["ok"] else ("WARN" if item.get("non_blocking") else "FAIL")
            lines.append(f"{mark} {item['name']}: {item['detail']}")
            self._append_terminal_line(f"SELFTEST {mark}: {item['name']} | {item['detail']}")

        profile = result.get("profile")
        calibration = result.get("calibration") if isinstance(result.get("calibration"), dict) else {}
        if isinstance(profile, dict) and profile:
            self._apply_audio_profile(profile, source=trigger)
        if calibration:
            in_dev, _ = pick_audio_device("input", None)
            out_dev, _ = pick_audio_device("output", None)
            try:
                input_name = str(sd.query_devices(in_dev, "input").get("name", "default")) if in_dev is not None else "default"
            except Exception:
                input_name = "default"
            try:
                output_name = str(sd.query_devices(out_dev, "output").get("name", "default")) if out_dev is not None else "default"
            except Exception:
                output_name = "default"
            self.settings.audio_guard_calibration = {
                **calibration,
                "profile": dict(profile) if isinstance(profile, dict) else {},
                "input_device_name": input_name,
                "output_device_name": output_name,
                "device_fingerprint": calibration.get("device_fingerprint") or build_device_fingerprint(in_dev, out_dev, 24000),
            }
            self.settings.save()

        if restart_after and result.get("ok"):
            self._handsfree_running = True
            self.head.set_running(True)
            self.status_label.setText("Kalibrace hotová, startuji chat.")
            self._sync_start_stop_button()
            self.sig_start_handsfree.emit()
        else:
            self.status_label.setText("Audio kalibrace doběhla.")

        if show_dialog:
            QMessageBox.information(self, "Audio selftest", "Audio guard selftest\n" + "\n".join(lines))
        return result

    def _save_defaults(self) -> None:
        self.settings.save()
        self._append_terminal_line("SYS: Aktuální nastavení bylo uloženo.")
        QMessageBox.information(self, "SAVE", "Aktuální nastavení bylo uloženo jako výchozí.")

    def _clear_session(self) -> None:
        try:
            self.sig_request_stop.emit()
        except Exception:
            pass
        self._handsfree_running = False
        self._sync_start_stop_button()
        self.head.set_running(False)
        self.head.set_error_text("")
        self.head.set_lipsync_snapshot(_closed_pose_snapshot())
        self.head.set_terminal_text("")
        self.status_label.setText("Relace byla vyčištěna.")
        if not self.settings.openai_api_key:
            self.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte.")

    @Slot()
    def _toggle_handsfree(self) -> None:
        if self._handsfree_running:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.head.set_running(False)
            self.status_label.setText("Hlasový chat zastaven.")
            self._append_terminal_line("SYS: Hlasový chat byl zastaven.")
            self._sync_start_stop_button()
            return

        if not self.settings.openai_api_key and self.api_key_input.text().strip():
            self.settings.openai_api_key = self.api_key_input.text().strip()
            self.settings.save()
            self._append_terminal_line("SYS: OpenAI API klíč byl uložen před startem relace.")

        if not self.settings.openai_api_key:
            self.status_label.setText("Nejdřív uložte OpenAI API klíč.")
            QMessageBox.warning(self, "API klíč", "Chybí OpenAI API klíč.")
            return

        if not isinstance(self.settings.audio_guard_calibration, dict) or not self.settings.audio_guard_calibration:
            self._append_terminal_line("SYS: Chybí uložený audio profil, spouštím krátký preflight.")
            self._calibrate_audio_guard(trigger="preflight", show_dialog=False, restart_after=False)

        self._handsfree_running = True
        self.head.set_running(True)
        self.status_label.setText("Hands-free relace se spouští.")
        self._append_terminal_line("SYS: Hands-free relace se spouští.")
        self._sync_start_stop_button()
        self.sig_start_handsfree.emit()

    @Slot(float)
    def _on_input_level(self, lvl: float) -> None:
        self.head.set_input_level(lvl)

    @Slot(float)
    def _on_output_level(self, lvl: float) -> None:
        self.head.set_output_level(lvl)

    @Slot(object)
    def _on_output_pose(self, snapshot: object) -> None:
        self.head.set_lipsync_snapshot(snapshot)

    @Slot(str)
    def _on_state(self, s: str) -> None:
        self.head.set_state(s)
        if s == "error":
            self._handsfree_running = False
            self.head.set_running(False)
            self._sync_start_stop_button()
        else:
            self.head.set_error_text("")
        self.status_label.setText(f"Stav relace: {s}")

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.head.set_terminal_text(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.head.set_error_text(msg)
        self._handsfree_running = False
        self.head.set_running(False)
        self._sync_start_stop_button()
        self.status_label.setText("Došlo k chybě relace.")
        self._append_terminal_line(f"SYS: ERROR {msg}")

    @Slot(object)
    def _on_guard_debug(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        telemetry = data.get("telemetry", {}) if isinstance(data.get("telemetry"), dict) else {}
        profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
        state = str(data.get("state", "-"))
        audio_mode = str(data.get("audio_mode", "notebook_builtin"))
        aec_aware = "on" if bool(data.get("aec_aware")) else "off"
        learning_mode = "on" if bool(data.get("learning_mode")) else "off"
        input_name = str(data.get("input_device_name", "-"))
        output_name = str(data.get("output_device_name", "-"))
        self.guard_debug_label.setText(
            "GUARD: živá telemetrie\n"
            f"stav={state}  samples={int(telemetry.get('samples', 0) or 0)}  "
            f"drop_rate={float(telemetry.get('drop_rate', 0.0) or 0.0):.3f}  "
            f"similarity={float(telemetry.get('avg_similarity', 0.0) or 0.0):.3f}  "
            f"voice={float(telemetry.get('avg_voice_likelihood', 0.0) or 0.0):.3f}  "
            f"aec={float(telemetry.get('avg_aec_quality', 0.0) or 0.0):.3f}\n"
            f"echo_soft={float(profile.get('echo_similarity_soft', 0.0) or 0.0):.3f}  "
            f"echo_drop={float(profile.get('echo_similarity_drop', 0.0) or 0.0):.3f}  "
            f"barge_in={float(profile.get('barge_in_min_input_level', 0.0) or 0.0):.3f}  "
            f"playback={float(profile.get('playback_activity_level', 0.0) or 0.0):.3f}\n"
            f"mode={audio_mode}  aec={aec_aware}  learning={learning_mode}  latency={int(data.get('calibration', {}).get('latency_samples', 0) or 0)}\n"
            f"mic={input_name[:42]} | spk={output_name[:42]}"
        )

    def _run_audio_test(self) -> None:
        restart_after = self._handsfree_running
        if restart_after:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.head.set_running(False)
            self._sync_start_stop_button()
            self._append_terminal_line("SYS: Chat byl dočasně zastaven kvůli ruční překalibraci.")
        result = self._calibrate_audio_guard(trigger="manual", show_dialog=True, restart_after=restart_after)
        if restart_after and not result.get("ok"):
            self._handsfree_running = True
            self.head.set_running(True)
            self._sync_start_stop_button()
            self.sig_start_handsfree.emit()
            self._append_terminal_line("SYS: Překalibrace nebyla ideální, chat byl obnoven s posledním profilem.")

    def _sync_start_stop_button(self) -> None:
        self.btn_start_stop.setText("Stop" if self._handsfree_running else "Start")

    def _append_terminal_line(self, line: str) -> None:
        current = list(getattr(self.head, "_terminal_lines", []))
        current.append((line or "").strip())
        self.head.set_terminal_text("\n".join(current[-10:]))

    def closeEvent(self, event) -> None:
        try:
            self.worker.request_stop()
        except Exception:
            pass
        try:
            self._thread.quit()
            self._thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    settings = AppSettings.load()
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        install_app_logging(log_dir=settings.ensure_log_dir(), session_tag=session_tag)
    except Exception:
        pass
    asset_issues = verify_asset_manifest()
    if asset_issues:
        raise RuntimeError("Integrita assetů selhala: " + "; ".join(asset_issues))

    app = QApplication(sys.argv)
    app.setStyleSheet(app_stylesheet())
    w = MainWindow(settings)
    if not settings.openai_api_key:
        QTimer.singleShot(0, lambda: w.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte."))
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
