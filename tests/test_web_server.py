from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from kajovochat.settings import AppSettings
from kajovochat.web.server import (
    BrowserCaptureProfile,
    KajovoHTTPServer,
    KajovoRequestHandler,
    KajovoWebApp,
    build_browser_session,
)


def test_build_browser_session_uses_semantic_vad_transcription_and_setting_tools() -> None:
    settings = AppSettings(voice="cedar", response_length="podrobná", response_formality="formální")
    capture = BrowserCaptureProfile(acoustic_profile="near_field")
    session = build_browser_session(settings, capture)
    assert session["model"] == "gpt-realtime"
    assert session["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
    assert session["audio"]["input"]["turn_detection"]["eagerness"] == "low"
    assert session["audio"]["input"]["noise_reduction"]["type"] == "near_field"
    assert session["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert session["audio"]["output"]["voice"] == "cedar"
    tool_names = {tool["name"] for tool in session["tools"]}
    assert all("strict" not in tool for tool in session["tools"])
    assert tool_names == {
        "spust_program",
        "nastav_hlas",
        "nastav_jazyk_odpovedi",
        "nastav_styl_odpovedi",
        "nastav_delku_odpovedi",
        "nastav_formalnost_odpovedi",
    }


def test_http_routes_return_state_index_and_options() -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])
    server = KajovoHTTPServer(("127.0.0.1", 0), KajovoRequestHandler, app)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/api/settings", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["state"]["voice"] == "marin"
        assert any(option["value"] == "cedar" for option in payload["state"]["options"]["voices"])
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Druh hlasu" in html
        assert "Styl odpovědí" in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_launch_program_accepts_only_powershell(monkeypatch) -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])

    monkeypatch.setattr("kajovochat.web.server.shutil.which", lambda name: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")

    calls: list[tuple[list[str], dict[str, object]]] = []

    class DummyProcess:
        pid = 4321

    def fake_popen(cmd: list[str], **kwargs: object) -> DummyProcess:
        calls.append((cmd, dict(kwargs)))
        return DummyProcess()

    monkeypatch.setattr("kajovochat.web.server.subprocess.Popen", fake_popen)

    result = app.launch_program("powershell")
    assert result["ok"] is True
    assert result["program"] == "powershell"
    assert result["pid"] == 4321
    assert calls[0][0][0].lower().endswith("powershell.exe")
    assert calls[0][0][1] == "-NoExit"


def test_launch_program_uses_visible_windows_console(monkeypatch) -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])

    monkeypatch.setattr("kajovochat.web.server.os.name", "nt")
    monkeypatch.setattr("kajovochat.web.server.shutil.which", lambda name: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    monkeypatch.setattr("kajovochat.web.server.subprocess.CREATE_NEW_CONSOLE", 0x00000010, raising=False)
    monkeypatch.setattr("kajovochat.web.server.subprocess.CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)

    calls: list[tuple[list[str], dict[str, object]]] = []

    class DummyProcess:
        pid = 5678

    def fake_popen(cmd: list[str], **kwargs: object) -> DummyProcess:
        calls.append((cmd, dict(kwargs)))
        return DummyProcess()

    monkeypatch.setattr("kajovochat.web.server.subprocess.Popen", fake_popen)

    result = app.launch_program("powershell")
    assert result["pid"] == 5678
    assert calls[0][0] == ["C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoExit"]
    assert calls[0][1]["creationflags"] == 0x00000010 | 0x00000200
    assert "stdout" not in calls[0][1]
    assert "stderr" not in calls[0][1]


def test_run_program_route_rejects_invalid_program() -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])
    server = KajovoHTTPServer(("127.0.0.1", 0), KajovoRequestHandler, app)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        req = urllib.request.Request(
            f"http://{host}:{port}/api/tools/run-program",
            data=json.dumps({"program": "calc"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read().decode("utf-8"))
            assert "powershell" in payload["error"]
        else:
            raise AssertionError("Expected HTTPError for invalid program")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_preferences_route_persists_voice_and_language() -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])
    server = KajovoHTTPServer(("127.0.0.1", 0), KajovoRequestHandler, app)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        req = urllib.request.Request(
            f"http://{host}:{port}/api/settings/preferences",
            data=json.dumps({
                "voice": "cedar",
                "answer_language_mode": "fixed",
                "fixed_answer_language": "en",
                "response_style": "stručný",
                "response_length": "krátká",
                "response_formality": "formální",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["restart_required"] is True
        assert payload["state"]["voice"] == "cedar"
        assert payload["state"]["fixed_answer_language"] == "en"
        assert payload["state"]["response_length"] == "krátká"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_apply_tool_changes_voice_and_requires_restart() -> None:
    app = KajovoWebApp(settings=AppSettings(), root_dir=Path(__file__).resolve().parents[1])
    payload = app.apply_tool("nastav_hlas", {"hlas": "cedar"})
    assert payload["ok"] is True
    assert payload["restart_required"] is True
    assert payload["state"]["voice"] == "cedar"
    assert "Cedar" in payload["message"]
