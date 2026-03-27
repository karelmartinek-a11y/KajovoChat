from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from ..settings import (
    ANSWER_LANGUAGE_MODE_CHOICES,
    LANGUAGE_CHOICES,
    RESPONSE_FORMALITY_CHOICES,
    RESPONSE_LENGTH_CHOICES,
    RESPONSE_STYLE_CHOICES,
    VOICE_CHOICES,
    AppSettings,
    build_system_prompt,
    language_label,
    normalize_answer_language_mode,
    normalize_fixed_language,
    normalize_response_formality,
    normalize_response_length,
    normalize_response_style,
    normalize_voice,
    voice_label,
)
from ..theme import Theme

_REALTIME_MODEL = "gpt-realtime"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_TOOL_RUN_PROGRAM = "spust_program"
_TOOL_SET_VOICE = "nastav_hlas"
_TOOL_SET_LANGUAGE = "nastav_jazyk_odpovedi"
_TOOL_SET_STYLE = "nastav_styl_odpovedi"
_TOOL_SET_LENGTH = "nastav_delku_odpovedi"
_TOOL_SET_FORMALITY = "nastav_formalnost_odpovedi"
_ALLOWED_PROGRAMS: dict[str, tuple[str, ...]] = {
    "powershell": ("powershell.exe", "pwsh", "powershell"),
}


@dataclass
class BrowserCaptureProfile:
    acoustic_profile: str = "far_field"
    input_label: str = ""
    output_label: str = ""
    echo_cancellation: bool | None = None
    noise_suppression: bool | None = None
    auto_gain_control: bool | None = None


def _normalize_acoustic_profile(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return "near_field" if normalized == "near_field" else "far_field"


def _guess_language(settings: AppSettings) -> str:
    candidate = (settings.fixed_answer_language or "cs").strip().lower()
    return candidate if candidate in {"cs", "en", "de", "sk", "fr"} else "cs"


def _browser_capture_profile(payload: dict[str, Any] | None) -> BrowserCaptureProfile:
    payload = payload or {}
    browser_audio = payload.get("browser_audio") if isinstance(payload.get("browser_audio"), dict) else {}
    return BrowserCaptureProfile(
        acoustic_profile=_normalize_acoustic_profile(payload.get("acoustic_profile")),
        input_label=str(payload.get("input_label") or "").strip(),
        output_label=str(payload.get("output_label") or "").strip(),
        echo_cancellation=(browser_audio.get("echoCancellation") if isinstance(browser_audio, dict) else None),
        noise_suppression=(browser_audio.get("noiseSuppression") if isinstance(browser_audio, dict) else None),
        auto_gain_control=(browser_audio.get("autoGainControl") if isinstance(browser_audio, dict) else None),
    )


def _choice_payload(choices: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"value": value, "label": label} for value, label in choices]


def _browser_tools() -> list[dict[str, Any]]:
    voices = [value for value, _ in VOICE_CHOICES]
    languages = [value for value, _ in LANGUAGE_CHOICES]
    language_modes = [value for value, _ in ANSWER_LANGUAGE_MODE_CHOICES]
    styles = [value for value, _ in RESPONSE_STYLE_CHOICES]
    lengths = [value for value, _ in RESPONSE_LENGTH_CHOICES]
    formalities = [value for value, _ in RESPONSE_FORMALITY_CHOICES]
    return [
        {
            "type": "function",
            "name": _TOOL_RUN_PROGRAM,
            "description": (
                "Spustí lokální program na počítači, kde běží Chatbot Kája. "
                "Použij jen tehdy, když uživatel výslovně požádá o spuštění programu. "
                "Nikdy nehadej parametr programu; když chybí, nejdřív se doptáš. "
                "Pro testování je povolený jen program powershell."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program": {"type": "string", "enum": ["powershell"]}
                },
                "required": ["program"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": _TOOL_SET_VOICE,
            "description": "Změní druh hlasu asistenta pro další relaci. Pokud už relace běží, klient ji restartuje.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hlas": {"type": "string", "enum": voices}
                },
                "required": ["hlas"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": _TOOL_SET_LANGUAGE,
            "description": "Nastaví jazyk odpovědí. Režim follow_input znamená stejný jazyk jako uživatel, fixed znamená vždy zvolený jazyk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rezim": {"type": "string", "enum": language_modes},
                    "jazyk": {"type": "string", "enum": languages},
                },
                "required": ["rezim"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": _TOOL_SET_STYLE,
            "description": "Nastaví hlavní styl odpovědi asistenta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "styl": {"type": "string", "enum": styles}
                },
                "required": ["styl"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": _TOOL_SET_LENGTH,
            "description": "Nastaví preferovanou délku odpovědí asistenta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delka": {"type": "string", "enum": lengths}
                },
                "required": ["delka"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": _TOOL_SET_FORMALITY,
            "description": "Nastaví míru formálnosti odpovědí asistenta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formalnost": {"type": "string", "enum": formalities}
                },
                "required": ["formalnost"],
                "additionalProperties": False,
            },
        },
    ]


def build_browser_session(settings: AppSettings, capture: BrowserCaptureProfile) -> dict[str, Any]:
    resolved_language = _guess_language(settings)
    instructions = (
        build_system_prompt(settings, resolved_language).rstrip()
        + "\nK dispozici máš nástroje pro změnu nastavení hlasového asistenta i pro testovací spuštění programu. "
        + "Programový nástroj spust_program použij jen při výslovné žádosti uživatele a nikdy nehadej chybějící parametr. "
        + "Pro testování je povolený jen program powershell. "
        + "Nastavení hlasu, jazyka, stylu, délky a formálnosti můžeš měnit jen pomocí dostupných nástrojů a povolených hodnot. "
        + "Když uživatel chce vědět, jaké volby má k dispozici, vyjmenuj jen skutečně povolené hodnoty z těchto nástrojů.\n"
    )
    return {
        "type": "realtime",
        "model": _REALTIME_MODEL,
        "instructions": instructions,
        "tools": _browser_tools(),
        "tool_choice": "auto",
        "audio": {
            "input": {
                "turn_detection": {
                    "type": "semantic_vad",
                    "eagerness": "low",
                    "create_response": True,
                    "interrupt_response": True,
                },
                "noise_reduction": {"type": capture.acoustic_profile},
                "transcription": {
                    "model": "gpt-4o-transcribe",
                    "language": resolved_language,
                },
            },
            "output": {
                "voice": settings.voice,
                "speed": 1.0,
            },
        },
    }


class KajovoWebApp:
    def __init__(self, settings: AppSettings, root_dir: Path) -> None:
        self.settings = settings
        self.root_dir = root_dir
        self.static_dir = root_dir / "kajovochat" / "web" / "static"
        self.assets_dir = root_dir / "kajovochat" / "resources" / "assets"
        self.theme = Theme()

    def build_state(self) -> dict[str, Any]:
        return {
            "app_name": "Chatbot Kája",
            "has_openai_api_key": bool(self.settings.openai_api_key),
            "voice": self.settings.voice,
            "voice_label": voice_label(self.settings.voice),
            "response_style": self.settings.response_style,
            "response_length": self.settings.response_length,
            "response_formality": self.settings.response_formality,
            "answer_language_mode": self.settings.answer_language_mode,
            "fixed_answer_language": self.settings.fixed_answer_language,
            "fixed_answer_language_label": language_label(self.settings.fixed_answer_language),
            "options": {
                "voices": _choice_payload(VOICE_CHOICES),
                "language_modes": _choice_payload(ANSWER_LANGUAGE_MODE_CHOICES),
                "languages": _choice_payload(LANGUAGE_CHOICES),
                "response_styles": _choice_payload(RESPONSE_STYLE_CHOICES),
                "response_lengths": _choice_payload(RESPONSE_LENGTH_CHOICES),
                "response_formalities": _choice_payload(RESPONSE_FORMALITY_CHOICES),
            },
            "theme": {
                "brand_blue": self.theme.brand_blue,
                "brand_yellow": self.theme.brand_yellow,
                "navy": self.theme.navy,
                "bg": self.theme.bg,
                "surface": self.theme.surface,
                "surface_2": self.theme.surface_2,
                "text": self.theme.text,
                "text_muted": self.theme.text_muted,
                "border": self.theme.border,
            },
        }

    def save_api_key(self, api_key: str) -> None:
        key = (api_key or "").strip()
        self.settings.openai_api_key = key
        self.settings.save()

    def delete_api_key(self) -> None:
        self.settings.openai_api_key = ""
        self.settings.save()

    def update_preferences(self, payload: dict[str, Any]) -> dict[str, Any]:
        restart_required = False
        changes: list[str] = []

        if "voice" in payload:
            voice = normalize_voice(str(payload.get("voice") or ""))
            if voice != self.settings.voice:
                self.settings.voice = voice
                restart_required = True
                changes.append(f"hlas={voice}")

        if "answer_language_mode" in payload:
            mode = normalize_answer_language_mode(str(payload.get("answer_language_mode") or ""))
            if mode != self.settings.answer_language_mode:
                self.settings.answer_language_mode = mode
                restart_required = True
                changes.append(f"rezim_jazyka={mode}")

        if "fixed_answer_language" in payload:
            language = normalize_fixed_language(str(payload.get("fixed_answer_language") or ""))
            if language != self.settings.fixed_answer_language:
                self.settings.fixed_answer_language = language
                restart_required = True
                changes.append(f"jazyk={language}")

        if "response_style" in payload:
            style = normalize_response_style(str(payload.get("response_style") or ""))
            if style != self.settings.response_style:
                self.settings.response_style = style
                restart_required = True
                changes.append(f"styl={style}")

        if "response_length" in payload:
            length = normalize_response_length(str(payload.get("response_length") or ""))
            if length != self.settings.response_length:
                self.settings.response_length = length
                restart_required = True
                changes.append(f"delka={length}")

        if "response_formality" in payload:
            formality = normalize_response_formality(str(payload.get("response_formality") or ""))
            if formality != self.settings.response_formality:
                self.settings.response_formality = formality
                restart_required = True
                changes.append(f"formalnost={formality}")

        self.settings.save()
        return {
            "ok": True,
            "restart_required": restart_required,
            "changes": changes,
            "state": self.build_state(),
        }

    def launch_program(self, program: str) -> dict[str, Any]:
        normalized = (program or "").strip().lower()
        if normalized not in _ALLOWED_PROGRAMS:
            raise ValueError("Povolený parametr programu je jen powershell.")

        executable = None
        for candidate in _ALLOWED_PROGRAMS[normalized]:
            resolved = shutil.which(candidate)
            if resolved:
                executable = resolved
                break
        if executable is None:
            raise RuntimeError("Program powershell se na tomto počítači nepodařilo najít.")

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
            creationflags |= int(getattr(subprocess, "CREATE_NEW_CONSOLE"))
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP"))

        kwargs: dict[str, Any] = {
            "cwd": os.path.expanduser("~"),
        }
        if os.name != "nt":
            kwargs.update({
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
                "close_fds": True,
            })
        if creationflags:
            kwargs["creationflags"] = creationflags

        process = subprocess.Popen([executable, "-NoExit"], **kwargs)
        return {
            "ok": True,
            "program": normalized,
            "pid": int(process.pid),
            "executable": executable,
        }

    def apply_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == _TOOL_RUN_PROGRAM:
            result = self.launch_program(str(args.get("program") or ""))
            result["message"] = f"Program {result['program']} byl spuštěn."
            return {"ok": True, "result": result, "restart_required": False}
        if name == _TOOL_SET_VOICE:
            payload = self.update_preferences({"voice": args.get("hlas")})
            payload["message"] = f"Hlas je nastavený na {voice_label(payload['state']['voice'])}."
            return payload
        if name == _TOOL_SET_LANGUAGE:
            mode = normalize_answer_language_mode(str(args.get("rezim") or ""))
            payload_in: dict[str, Any] = {"answer_language_mode": mode}
            if mode == "fixed":
                if not args.get("jazyk"):
                    raise ValueError("Pro režim fixed je potřeba zadat i parametr jazyk.")
                payload_in["fixed_answer_language"] = args.get("jazyk")
            payload = self.update_preferences(payload_in)
            if payload['state']['answer_language_mode'] == 'fixed':
                payload["message"] = f"Jazyk odpovědí je nastavený na {payload['state']['fixed_answer_language_label']}."
            else:
                payload["message"] = "Jazyk odpovědí je nastavený podle jazyka uživatele."
            return payload
        if name == _TOOL_SET_STYLE:
            payload = self.update_preferences({"response_style": args.get("styl")})
            payload["message"] = f"Styl odpovědí je nastavený na {payload['state']['response_style']}."
            return payload
        if name == _TOOL_SET_LENGTH:
            payload = self.update_preferences({"response_length": args.get("delka")})
            payload["message"] = f"Délka odpovědí je nastavená na {payload['state']['response_length']}."
            return payload
        if name == _TOOL_SET_FORMALITY:
            payload = self.update_preferences({"response_formality": args.get("formalnost")})
            payload["message"] = f"Formálnost odpovědí je nastavená na {payload['state']['response_formality']}."
            return payload
        raise ValueError(f"Nepodporovaný nástroj: {name}")

    def create_realtime_call(self, sdp_offer: str, capture_payload: dict[str, Any] | None) -> str:
        api_key = self.settings.openai_api_key.strip()
        if not api_key:
            raise ValueError("Nejdřív ulož OpenAI API klíč.")
        if not sdp_offer.strip():
            raise ValueError("Chybí SDP nabídka z browseru.")
        capture = _browser_capture_profile(capture_payload)
        session = build_browser_session(self.settings, capture)
        files = {
            "sdp": (None, sdp_offer),
            "session": (None, json.dumps(session, ensure_ascii=False)),
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
            )
        if response.status_code >= 400:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise RuntimeError(f"OpenAI realtime call selhal: {detail}")
        return response.text


class KajovoRequestHandler(BaseHTTPRequestHandler):
    server_version = "KajovoChatHTTP/1.0"

    @property
    def app(self) -> KajovoWebApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("Neplatné JSON tělo požadavku.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON tělo musí být objekt.")
        return payload

    def _serve_file(self, path: Path, fallback_content_type: str = "application/octet-stream") -> None:
        if not path.exists() or not path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Soubor nebyl nalezen."})
            return
        content_type, _ = mimetypes.guess_type(str(path))
        self._send_bytes(HTTPStatus.OK, path.read_bytes(), content_type or fallback_content_type)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._serve_file(self.app.static_dir / "index.html", "text/html; charset=utf-8")
            return
        if self.path == "/app.js":
            self._serve_file(self.app.static_dir / "app.js", "application/javascript; charset=utf-8")
            return
        if self.path == "/app.css":
            self._serve_file(self.app.static_dir / "app.css", "text/css; charset=utf-8")
            return
        if self.path == "/assets/logo_chatbot_kaja.png":
            self._serve_file(self.app.assets_dir / "logo_chatbot_kaja.png", "image/png")
            return
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "status": "ready"})
            return
        if self.path == "/api/settings":
            self._send_json(HTTPStatus.OK, {"ok": True, "state": self.app.build_state()})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Neznámá cesta."})

    def do_POST(self) -> None:
        try:
            if self.path == "/api/settings/openai-key":
                payload = self._read_json()
                self.app.save_api_key(str(payload.get("api_key") or ""))
                self._send_json(HTTPStatus.OK, {"ok": True, "state": self.app.build_state()})
                return
            if self.path == "/api/settings/delete-openai-key":
                self.app.delete_api_key()
                self._send_json(HTTPStatus.OK, {"ok": True, "state": self.app.build_state()})
                return
            if self.path == "/api/settings/preferences":
                payload = self._read_json()
                result = self.app.update_preferences(payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if self.path == "/api/realtime/call":
                payload = self._read_json()
                answer_sdp = self.app.create_realtime_call(str(payload.get("sdp") or ""), payload)
                self._send_bytes(HTTPStatus.OK, answer_sdp.encode("utf-8"), "application/sdp")
                return
            if self.path == "/api/tools/run-program":
                payload = self._read_json()
                result = self.app.launch_program(str(payload.get("program") or ""))
                self._send_json(HTTPStatus.OK, {"ok": True, "result": result})
                return
            if self.path == "/api/tools/execute":
                payload = self._read_json()
                result = self.app.apply_tool(str(payload.get("name") or ""), payload.get("arguments") or {})
                self._send_json(HTTPStatus.OK, result)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Neznámá cesta."})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except RuntimeError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"Interní chyba: {exc}"})


class KajovoHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], app: KajovoWebApp):
        super().__init__(server_address, handler_class)
        self.app = app


def _pick_port(host: str, preferred_port: int) -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if probe.connect_ex((host, preferred_port)) != 0:
            return preferred_port
    for port in range(preferred_port + 1, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if probe.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Nepodařilo se najít volný TCP port pro lokální webové rozhraní.")


def run_server(*, open_browser: bool = True, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> int:
    root_dir = Path(__file__).resolve().parents[2]
    settings = AppSettings.load()
    app = KajovoWebApp(settings=settings, root_dir=root_dir)
    resolved_port = _pick_port(host, port)
    server = KajovoHTTPServer((host, resolved_port), KajovoRequestHandler, app)
    url = f"http://{host}:{resolved_port}/"
    print(f"[INFO] Chatbot Kája Web běží na {url}")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chatbot Kája - webový voice frontend přes browser")
    parser.add_argument("--no-browser", action="store_true", help="Neotevírat browser automaticky.")
    parser.add_argument("--host", default=_DEFAULT_HOST, help="Host pro lokální HTTP server.")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="Preferovaný TCP port pro lokální HTTP server.")
    args = parser.parse_args(argv)
    return run_server(open_browser=not args.no_browser, host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
