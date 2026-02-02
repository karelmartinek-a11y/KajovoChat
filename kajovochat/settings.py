from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from appdirs import user_config_dir


APP_NAME = "KajovoChat"
ORG_NAME = "Kajovo"


def _config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, ORG_NAME))


def _config_path() -> Path:
    return _config_dir() / "settings.json"


def _mask_key(key: str) -> str:
    # Not security—only avoids accidental shoulder-surfing.
    return key[::-1]


def _unmask_key(masked: str) -> str:
    return masked[::-1]


# ---- Language / formality ----

LANGUAGE_CHOICES = [
    ("auto", "Auto"),
    ("cs", "Čeština (cs)"),
    ("en", "Angličtina (en)"),
    ("de", "Němčina (de)"),
    ("sk", "Slovenština (sk)"),
    ("fr", "Francouzština (fr)"),
]

LANG_CODE_TO_PROMPT = {
    "cs": "Odpovídej česky.",
    "sk": "Odpovídej slovensky.",
    "de": "Antworte auf Deutsch.",
    "en": "Answer in English.",
    "fr": "Réponds en français.",
}


def language_label(code: str) -> str:
    for c, lbl in LANGUAGE_CHOICES:
        if c == code:
            return lbl
    return code


# ---- Response shaping ----

STYLE_PROMPTS = {
    "obsáhlé": "Odpovídej obsáhle, strukturovaně a s příklady, ale bez zbytečné omáčky.",
    "věcné": "Odpovídej věcně a prakticky. Vyhni se zbytečné omáčce.",
    "exaktní": "Odpovídej exaktně. Používej jasné definice a přesné kroky. Kde je nejistota, výslovně ji uveď.",
    "strohé": "Odpovídej stručně a přímo, bez úvodu a bez vysvětlování, pokud to není nutné.",
}

LENGTH_PROMPTS = {
    "krátké": "Délka odpovědi: krátká (max cca 4–6 vět, pokud to stačí).",
    "normální": "Délka odpovědi: normální.",
    "dlouhé": "Délka odpovědi: dlouhá (podrobně, ale stále přehledně).",
}

DETAIL_PROMPTS = {
    "stručná": "Buď stručný. Pokud si nejsi jistý, raději se doptáš jednou otázkou.",
    "detailní": "Buď detailnější a strukturovaný. U důležitých věcí přidej krátké odůvodnění.",
}

FORMALITY_PROMPTS = {
    ("cs", "vykání"): "V češtině používej výhradně vykání (Vy).",
    ("cs", "tykání"): "V češtině používej tykání (ty).",
    ("sk", "vykání"): "V slovenčině používaj výhradne vykanie (Vy).",
    ("sk", "tykání"): "V slovenčině používaj tykanie.",
    ("de", "vykání"): "In Deutsch verwende die höfliche Anrede (Sie).",
    ("de", "tykání"): "In Deutsch verwende das Du (du).",
    ("fr", "vykání"): "En français, utilise le vouvoiement.",
    ("fr", "tykání"): "En français, utilise le tutoiement.",
    ("en", "vykání"): "Use a polite, professional tone (no slang).",
    ("en", "tykání"): "Use a friendly tone, but stay respectful.",
}


# ---- TTS ----

# Konzervativní seznam oficiálně podporovaných hlasů Realtime TTS (2026-02-02).
TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]

# Heuristické preference podle jazyka.
LANG_TO_PREFERRED_VOICES = {
    "cs": ["alloy", "echo", "shimmer"],
    "sk": ["alloy", "echo", "shimmer"],
    "de": ["sage", "alloy", "ash"],
    "en": ["alloy", "ash", "verse"],
    "fr": ["coral", "marin", "alloy"],
}


def normalize_language_code(value: str) -> str:
    v = (value or "").strip().lower()
    legacy = {
        "česky": "cs",
        "slovensky": "sk",
        "německy": "de",
        "anglicky": "en",
        "francouzsky": "fr",
        "auto": "auto",
    }
    if v in legacy:
        return legacy[v]
    if v in {"cs", "en", "de", "sk", "fr", "auto"}:
        return v
    # unknown: keep as-is (will behave as auto)
    return "auto"


@dataclass
class AppSettings:
    # UI / behavior
    response_style: str = "věcné"       # obsáhlé, věcné, exaktní, strohé
    response_length: str = "normální"   # krátké, normální, dlouhé
    response_detail: str = "stručná"    # stručná, detailní
    language: str = "auto"              # auto/cs/en/de/sk/fr
    formality: str = "vykání"           # vykání/tykání
    log_dir: str = str((Path.home() / "Documents" / "KajovoChatLogs").resolve())

    # OpenAI
    openai_api_key_masked: str = ""
    chat_model: str = "gpt-4o-mini"
    # Realtime voice
    realtime_model: str = "gpt-realtime"
    # STT is fixed to Whisper
    stt_model: str = "whisper-1"
    # TTS defaults (keep editable)
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_speed: float = 1.0

    # LLM params
    temperature: float = 0.3
    max_output_tokens: int = 512

    # audio
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_samplerate: int = 16000
    tts_samplerate: int = 24000

    # VAD
    vad_rms_threshold: float = 0.012  # base threshold (will be auto-adjusted after calibration in hands-free)
    vad_silence_ms: int = 900
    vad_calibration_s: float = 0.7
    vad_multiplier: float = 3.0
    max_record_seconds: int = 25

    @property
    def openai_api_key(self) -> str:
        return _unmask_key(self.openai_api_key_masked) if self.openai_api_key_masked else ""

    @openai_api_key.setter
    def openai_api_key(self, key: str) -> None:
        self.openai_api_key_masked = _mask_key(key.strip()) if key else ""

    def ensure_log_dir(self) -> Path:
        p = Path(self.log_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self) -> None:
        _config_dir().mkdir(parents=True, exist_ok=True)
        _config_path().write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "AppSettings":
        p = _config_path()
        if not p.exists():
            s = cls()
            s.ensure_log_dir()
            s.save()
            return s

        data = json.loads(p.read_text(encoding="utf-8"))

        # migration from older keys
        if "voice_language" in data and "language" not in data:
            data["language"] = normalize_language_code(data.get("voice_language", "auto"))
        if "voice_gender" in data and "formality" not in data:
            # old setting had no formality; default to vykání.
            data["formality"] = "vykání"
        if "tts_voice_female" in data and "tts_voice" not in data:
            # preserve prior default: use female voice as generic voice
            data["tts_voice"] = data.get("tts_voice_female") or "nova"

        # enforce fixed STT model regardless of config drift
        data["stt_model"] = "whisper-1"

        s = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        s.language = normalize_language_code(s.language)
        # normalize TTS hlas na aktuální podporovaný seznam
        legacy_map = {
            "fable": "alloy",
            "nova": "alloy",
            "onyx": "alloy",
        }
        v = (s.tts_voice or "").strip()
        v = legacy_map.get(v, v)
        if v not in TTS_VOICES:
            pref = LANG_TO_PREFERRED_VOICES.get(s.language, [])
            v = pref[0] if pref else TTS_VOICES[0]
        elif s.language in LANG_TO_PREFERRED_VOICES and v not in LANG_TO_PREFERRED_VOICES[s.language]:
            v = LANG_TO_PREFERRED_VOICES[s.language][0]
        s.tts_voice = v
        s.ensure_log_dir()
        return s


def build_system_prompt(settings: AppSettings, resolved_language: str) -> str:
    lang = resolved_language if resolved_language in LANG_CODE_TO_PROMPT else "cs"
    lang_prompt = LANG_CODE_TO_PROMPT.get(lang, "Odpovídej česky.")
    style = STYLE_PROMPTS.get(settings.response_style, STYLE_PROMPTS["věcné"])
    length = LENGTH_PROMPTS.get(settings.response_length, LENGTH_PROMPTS["normální"])
    detail = DETAIL_PROMPTS.get(settings.response_detail, DETAIL_PROMPTS["stručná"])
    form = FORMALITY_PROMPTS.get((lang, settings.formality), "")
    parts = [lang_prompt, form, style, length, detail]
    return "\n".join([p for p in parts if p]).strip() + "\n"
