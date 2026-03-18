from __future__ import annotations

import base64
import ctypes
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from appdirs import user_config_dir

try:
    import keyring
except Exception:
    keyring = None


APP_NAME = "ChatbotKaja"
ORG_NAME = "Kajovo"
KEYRING_SERVICE = "KajovoChat/OpenAI"

ANSWER_LANGUAGE_MODE_CHOICES = [
    ("follow_input", "Odpovídat jazykem uživatele"),
    ("fixed", "Vždy odpovídat zvoleným jazykem"),
]

LANGUAGE_CHOICES = [
    ("cs", "Čeština"),
    ("en", "Angličtina"),
    ("de", "Němčina"),
    ("sk", "Slovenština"),
    ("fr", "Francouzština"),
]

RESPONSE_STYLE_CHOICES = [
    ("stručný", "Stručný"),
    ("vědecký_s_analýzou", "Vědecký s analýzou"),
    ("normální", "Normální"),
]

LANG_CODE_TO_PROMPT = {
    "cs": "Odpovídej česky.",
    "sk": "Odpovídej slovensky.",
    "de": "Antworte auf Deutsch.",
    "en": "Answer in English.",
    "fr": "Réponds en français.",
}

STYLE_PROMPTS = {
    "stručný": "Odpovídej stručně, přímo a bez zbytečných odboček.",
    "vědecký_s_analýzou": (
        "Odpovídej analyticky a strukturovaně. Pracuj explicitně s předpoklady, nejistotou a důvody závěrů."
    ),
    "normální": "Odpovídej přirozeně, užitečně a věcně jako běžný hlasový asistent.",
}


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


def normalize_fixed_language(value: str) -> str:
    normalized = normalize_language_code(value)
    if normalized == "auto":
        return "cs"
    if normalized in {code for code, _ in LANGUAGE_CHOICES}:
        return normalized
    return "cs"


def normalize_answer_language_mode(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"follow_input", "fixed"}:
        return normalized
    return "follow_input"


def normalize_response_style(value: str) -> str:
    normalized = (value or "").strip().lower()
    mapping = {
        "stručný": "stručný",
        "stručny": "stručný",
        "vědecký_s_analýzou": "vědecký_s_analýzou",
        "vedecky_s_analyzou": "vědecký_s_analýzou",
        "normální": "normální",
        "normalni": "normální",
    }
    return mapping.get(normalized, "normální")


def _migrate_response_style(data: dict) -> str:
    if "response_style" in data and data["response_style"] in {code for code, _ in RESPONSE_STYLE_CHOICES}:
        return normalize_response_style(str(data["response_style"]))

    legacy_style = str(data.get("response_style", "")).strip().lower()
    legacy_length = str(data.get("response_length", "")).strip().lower()
    legacy_detail = str(data.get("response_detail", "")).strip().lower()

    if legacy_style in {"strohé", "strohe"} or legacy_length == "krátké":
        return "stručný"
    if legacy_style in {"exaktní", "exaktní", "exaktni"} or legacy_detail == "detailní":
        return "vědecký_s_analýzou"
    return "normální"


@dataclass
class AppSettings:
    answer_language_mode: str = "follow_input"
    fixed_answer_language: str = "cs"
    response_style: str = "normální"
    log_dir: str = str((Path.home() / "Documents" / "ChatbotKajaLogs").resolve())
    openai_api_key_masked: str = ""

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

        if "answer_language_mode" not in data:
            legacy_language = normalize_language_code(str(data.get("language", "auto")))
            if legacy_language != "auto":
                data["answer_language_mode"] = "fixed"
                data["fixed_answer_language"] = legacy_language
            else:
                data["answer_language_mode"] = "follow_input"
        if "fixed_answer_language" not in data:
            data["fixed_answer_language"] = normalize_fixed_language(data.get("language", "cs"))
        data["response_style"] = _migrate_response_style(data)

        settings = cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})
        settings.answer_language_mode = normalize_answer_language_mode(settings.answer_language_mode)
        settings.fixed_answer_language = normalize_fixed_language(settings.fixed_answer_language)
        settings.response_style = normalize_response_style(settings.response_style)
        settings.ensure_log_dir()
        return settings


def build_system_prompt(settings: AppSettings, resolved_language: str) -> str:
    prompt_parts = [
        "Jsi obecný hlasový asistent podobný ChatGPT.",
        "Odpovídej užitečně, srozumitelně, přirozeně a bezpečně.",
        "Nevymýšlej si schopnosti, neveřejná data ani přístup k cizím systémům.",
        "Když si nejsi jistý, řekni to stručně a jasně.",
        "Neptej se na autorizaci, hesla ani identitu, pokud to uživatel výslovně neřeší.",
    ]

    if settings.answer_language_mode == "fixed":
        output_language = normalize_fixed_language(settings.fixed_answer_language)
        prompt_parts.append(
            LANG_CODE_TO_PROMPT.get(output_language, LANG_CODE_TO_PROMPT["cs"])
            + " Odpovídej takto vždy bez ohledu na jazyk vstupního dotazu."
        )
    else:
        follow_language = normalize_fixed_language(resolved_language)
        prompt_parts.append("Odpovídej ve stejném jazyce, jakým mluví nebo píše uživatel.")
        prompt_parts.append(
            LANG_CODE_TO_PROMPT.get(follow_language, LANG_CODE_TO_PROMPT["cs"])
            + " Toto použij jen jako fallback, když jazyk vstupu nejde spolehlivě poznat."
        )

    style_prompt = STYLE_PROMPTS.get(settings.response_style)
    if style_prompt:
        prompt_parts.append(style_prompt)

    return "\n".join(prompt_parts).strip() + "\n"
