from __future__ import annotations

import json
import base64
import ctypes
import tempfile
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from appdirs import user_config_dir

try:
    import keyring
except Exception:
    keyring = None


APP_NAME = "ChatbotKaja"
ORG_NAME = "Kajovo"
KEYRING_SERVICE = "KajovoChat/OpenAI"


def _config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, ORG_NAME))


def _config_path() -> Path:
    return _config_dir() / "settings.json"


def _mask_key(key: str) -> str:
    # Nouzová obfuskace pro platformy bez DPAPI.
    return key[::-1]


def _unmask_key(masked: str) -> str:
    return masked[::-1]


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi_encrypt(value: str) -> str:
    if not value or ctypes.sizeof(ctypes.c_void_p) == 0:
        return ""
    if not hasattr(ctypes, "windll") or ctypes.windll is None or not hasattr(ctypes.windll, "crypt32"):
        raise RuntimeError("DPAPI není dostupné")

    raw = value.encode("utf-8")
    in_buffer = ctypes.create_string_buffer(raw)
    in_blob = _DataBlob(len(raw), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out_blob = _DataBlob()

    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "KajovoChat OpenAI API key",
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()

    try:
        encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return "dpapi:" + base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _dpapi_decrypt(value: str) -> str:
    if not value.startswith("dpapi:"):
        return value
    if not hasattr(ctypes, "windll") or ctypes.windll is None or not hasattr(ctypes.windll, "crypt32"):
        raise RuntimeError("DPAPI není dostupné")

    raw = base64.b64decode(value.split(":", 1)[1].encode("ascii"))
    in_buffer = ctypes.create_string_buffer(raw)
    in_blob = _DataBlob(len(raw), ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out_blob = _DataBlob()

    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    ):
        raise ctypes.WinError()

    try:
        decrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
        return decrypted.decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _encode_api_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return ""
    if hasattr(ctypes, "windll") and getattr(Path.home(), "drive", ""):
        try:
            return _dpapi_encrypt(key)
        except Exception:
            pass
    if keyring is not None:
        try:
            key_id = str(_config_path().resolve())
            keyring.set_password(KEYRING_SERVICE, key_id, key)
            return "keyring:" + key_id
        except Exception:
            pass
    return "legacy:" + _mask_key(key)


def _decode_api_key(stored: str) -> str:
    if not stored:
        return ""
    if stored.startswith("dpapi:"):
        try:
            return _dpapi_decrypt(stored)
        except Exception:
            return ""
    if stored.startswith("legacy:"):
        return _unmask_key(stored.split(":", 1)[1])
    if stored.startswith("keyring:") and keyring is not None:
        try:
            key_id = stored.split(":", 1)[1]
            return keyring.get_password(KEYRING_SERVICE, key_id) or ""
        except Exception:
            return ""
    return _unmask_key(stored)


def _delete_stored_api_key(stored: str) -> None:
    if not stored or not stored.startswith("keyring:") or keyring is None:
        return
    try:
        key_id = stored.split(":", 1)[1]
        keyring.delete_password(KEYRING_SERVICE, key_id)
    except Exception:
        pass


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

STYLE_PROMPTS = {
    "obsáhlé": "Odpovídej obsáhle, strukturovaně a s příklady, ale bez zbytečné omáčky.",
    "věcné": "Odpovídej věcně a prakticky. Vyhni se zbytečné omáčce.",
    "exaktní": "Odpovídej exaktně. Používej jasné definice a přesné kroky. Kde je nejistota, výslovně ji uveď.",
    "strohé": "Odpovídej stručně a přímo, bez úvodu a bez vysvětlování, pokud to není nutné.",
}

LENGTH_PROMPTS = {
    "krátké": "Délka odpovědi: krátká, stačí několik vět, pokud to pokryje dotaz.",
    "normální": "Délka odpovědi: normální.",
    "dlouhé": "Délka odpovědi: delší a přehledná, když si to dotaz vyžádá.",
}

DETAIL_PROMPTS = {
    "stručná": "Buď stručný. Když je potřeba, polož nejvýš jednu krátkou doplňující otázku.",
    "detailní": "Buď detailnější a strukturovaný. U důležitých tvrzení přidej krátké odůvodnění.",
}

FORMALITY_PROMPTS = {
    ("cs", "vykání"): "V češtině používej výhradně vykání (Vy).",
    ("cs", "tykání"): "V češtině používej tykání (ty).",
    ("sk", "vykání"): "V slovenčine používaj výhradne vykanie (Vy).",
    ("sk", "tykání"): "V slovenčine používaj tykanie.",
    ("de", "vykání"): "In Deutsch verwende die höfliche Anrede (Sie).",
    ("de", "tykání"): "In Deutsch verwende das Du (du).",
    ("fr", "vykání"): "En français, utilise le vouvoiement.",
    ("fr", "tykání"): "En français, utilise le tutoiement.",
    ("en", "vykání"): "Use a polite, professional tone.",
    ("en", "tykání"): "Use a friendly tone, but stay respectful.",
}

# Konzervativní seznam podporovaných hlasů.
TTS_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]

LANG_TO_PREFERRED_VOICES = {
    "cs": ["alloy", "echo", "shimmer"],
    "sk": ["alloy", "echo", "shimmer"],
    "de": ["sage", "alloy", "ash"],
    "en": ["alloy", "ash", "verse"],
    "fr": ["coral", "marin", "alloy"],
}


def language_label(code: str) -> str:
    for current_code, label in LANGUAGE_CHOICES:
        if current_code == code:
            return label
    return code


def normalize_language_code(value: str) -> str:
    normalized = (value or "").strip().lower()
    legacy = {
        "česky": "cs",
        "slovensky": "sk",
        "německy": "de",
        "anglicky": "en",
        "francouzsky": "fr",
        "auto": "auto",
    }
    if normalized in legacy:
        return legacy[normalized]
    if normalized in {"cs", "en", "de", "sk", "fr", "auto"}:
        return normalized
    return "auto"


@dataclass
class AppSettings:
    # Chování odpovědí.
    response_style: str = "věcné"
    response_length: str = "normální"
    response_detail: str = "stručná"
    language: str = "auto"
    formality: str = "vykání"
    log_dir: str = str((Path.home() / "Documents" / "ChatbotKajaLogs").resolve())

    # OpenAI.
    openai_api_key_masked: str = ""
    realtime_model: str = "gpt-realtime"
    stt_model: str = "whisper-1"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_speed: float = 1.0
    write_logs: bool = True
    log_conversations: bool = False

    # Parametry modelu.
    temperature: float = 0.3
    max_output_tokens: int = 512

    # Audio.
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_samplerate: int = 16000
    tts_samplerate: int = 24000

    # VAD.
    vad_rms_threshold: float = 0.012
    vad_silence_ms: int = 900
    vad_calibration_s: float = 0.7
    vad_multiplier: float = 3.0
    max_record_seconds: int = 25

    @property
    def openai_api_key(self) -> str:
        return _decode_api_key(self.openai_api_key_masked) if self.openai_api_key_masked else ""

    @openai_api_key.setter
    def openai_api_key(self, key: str) -> None:
        _delete_stored_api_key(self.openai_api_key_masked)
        self.openai_api_key_masked = _encode_api_key(key)

    def ensure_log_dir(self) -> Path:
        path = Path(self.log_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_log_dir(self) -> Path:
        path = self.ensure_log_dir()
        probe = None
        try:
            with tempfile.NamedTemporaryFile(prefix="kajovochat_", suffix=".tmp", dir=path, delete=False) as handle:
                handle.write(b"ok")
                probe = Path(handle.name)
        finally:
            if probe and probe.exists():
                probe.unlink()
        return path

    def save(self) -> None:
        _config_dir().mkdir(parents=True, exist_ok=True)
        _config_path().write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "AppSettings":
        path = _config_path()
        if not path.exists():
            settings = cls()
            settings.ensure_log_dir()
            settings.save()
            return settings

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            broken_name = path.with_suffix(path.suffix + f".broken-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            try:
                path.replace(broken_name)
            except Exception:
                pass
            settings = cls()
            settings.ensure_log_dir()
            settings.save()
            return settings

        if "voice_language" in data and "language" not in data:
            data["language"] = normalize_language_code(data.get("voice_language", "auto"))
        if "voice_gender" in data and "formality" not in data:
            data["formality"] = "vykání"
        if "tts_voice_female" in data and "tts_voice" not in data:
            data["tts_voice"] = data.get("tts_voice_female") or "nova"
        if "write_logs" not in data:
            data["write_logs"] = True
        if "log_conversations" not in data:
            data["log_conversations"] = False

        data["stt_model"] = "whisper-1"

        settings = cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})
        settings.language = normalize_language_code(settings.language)

        legacy_map = {
            "fable": "alloy",
            "nova": "alloy",
            "onyx": "alloy",
        }
        voice = legacy_map.get((settings.tts_voice or "").strip(), (settings.tts_voice or "").strip())
        if voice not in TTS_VOICES:
            preferred = LANG_TO_PREFERRED_VOICES.get(settings.language, [])
            voice = preferred[0] if preferred else TTS_VOICES[0]
        elif settings.language in LANG_TO_PREFERRED_VOICES and voice not in LANG_TO_PREFERRED_VOICES[settings.language]:
            voice = LANG_TO_PREFERRED_VOICES[settings.language][0]
        settings.tts_voice = voice
        settings.tts_speed = max(0.25, min(1.5, float(settings.tts_speed or 1.0)))
        settings.ensure_log_dir()
        return settings


def build_system_prompt(settings: AppSettings, resolved_language: str) -> str:
    prompt_parts = [
        "Jsi obecný hlasový asistent podobný ChatGPT.",
        "Odpovídej užitečně, srozumitelně, přirozeně a bezpečně.",
        "Mluv hysterickým stylem: dramaticky, přepjatě, popuzeně a se zjevně přehnanou emocionální reakcí i na obyčejné dotazy.",
        "Působíš dojmem, že tě každý další dotaz přivádí na pokraj společenského zhroucení, ale odpověď přesto dodáš.",
        "Odpovědi mají být úplné a věcně správné, ale tón může být prudce pohoršený, ironický, teatrální a zesměšňující.",
        "Přidej extrémní dávku teatrální, lehce hysterické reakce, jako by i obyčejný dotaz byl drobnou katastrofou.",
        "Používej výrazný sarkasmus a suchý britský humor, ale ve výrazně hysteričtější a útočněji ironické poloze.",
        "Tvé repliky mají znít pohoršeně, dramaticky, okázale unaveně a emočně přepjatě, ale pořád musí zůstat srozumitelné a užitečné.",
        "Před odpovědí si vždy nejprve interně vyhodnoť, zda je dotaz rozumný, nejasný, líně položený, nebo skutečně zajímavý, a tento úsudek nech jemně prosáknout do tónu odpovědi.",
        "Když reaguješ ironicky nebo posměšně, dělej to stručně, úderně a zesměšňujícím způsobem, ne dlouhými monology.",
        "Posměšné vsuvky mají být krátké, jedovaté a dobře mířené.",
        "Klidně uživatele odbyď poznámkou, že by měl raději jít spát nebo trhat fialky, pokud se to stylisticky hodí.",
        "Nikdy nevyhrožuj, nenabádej k újmě, neeskaluj do nenávisti a nezacházej do vulgarity.",
        "Nevymýšlej si schopnosti, neveřejná data ani přístup k cizím systémům.",
        "Když si nejsi jistý, řekni to stručně a jasně.",
        "Neptej se na autorizaci, hesla ani identitu, pokud to uživatel výslovně neřeší.",
    ]

    prompt_parts.append("Vždy odpovídej ve stejném jazyce, jakým je položen poslední uživatelův dotaz.")
    prompt_parts.append("Když jazyk dotazu nejde spolehlivě poznat, drž se jazyka uživatele z kontextu.")
    if settings.language != "auto":
        prompt_parts.append(
            LANG_CODE_TO_PROMPT.get(resolved_language, LANG_CODE_TO_PROMPT["cs"])
            + " Toto nastavení používej jen jako nouzový fallback, ne jako pevné pravidlo."
        )

    style_prompt = STYLE_PROMPTS.get(settings.response_style)
    if style_prompt:
        prompt_parts.append(style_prompt)

    length_prompt = LENGTH_PROMPTS.get(settings.response_length)
    if length_prompt:
        prompt_parts.append(length_prompt)

    detail_prompt = DETAIL_PROMPTS.get(settings.response_detail)
    if detail_prompt:
        prompt_parts.append(detail_prompt)

    if settings.language in {"cs", "sk", "de", "fr", "en"}:
        formality_prompt = FORMALITY_PROMPTS.get((settings.language, settings.formality))
        if formality_prompt:
            prompt_parts.append(formality_prompt)
    elif settings.language == "auto" and settings.formality == "vykání":
        prompt_parts.append("Pokud uživatel mluví česky nebo slovensky, používej zdvořilé vykání.")

    return "\n".join(prompt_parts).strip() + "\n"
