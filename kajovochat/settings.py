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


@dataclass
class AppSettings:
    # UI / behavior
    response_style: str = "věcné"     # obsáhlé, věcné, exaktní, strohé
    response_length: str = "normální" # krátké, normální, dlouhé
    voice_language: str = "česky"     # česky, slovensky, německy, anglicky, francouzsky
    voice_gender: str = "ženský"      # ženský, mužský
    log_dir: str = str((Path.home() / "Documents" / "KajovoChatLogs").resolve())

    # OpenAI
    openai_api_key_masked: str = ""
    chat_model: str = "gpt-4o-mini"
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice_female: str = "nova"
    tts_voice_male: str = "onyx"

    # audio
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_samplerate: int = 16000
    tts_samplerate: int = 24000
    vad_rms_threshold: float = 0.012  # tweak per mic
    vad_silence_ms: int = 900
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
        s = cls(**data)
        s.ensure_log_dir()
        return s


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

LANGUAGE_PROMPTS = {
    "česky": "Odpovídej česky.",
    "slovensky": "Odpovídej slovensky.",
    "německy": "Antworte auf Deutsch.",
    "anglicky": "Answer in English.",
    "francouzsky": "Réponds en français.",
}
