from __future__ import annotations

from fastapi.testclient import TestClient

from kajovochat.settings import AppSettings
from kajovochat.webapp.configuration import build_realtime_session_config, infer_noise_reduction
from kajovochat.webapp.server import create_app


def _settings() -> AppSettings:
    settings = AppSettings(
        answer_language_mode="follow_input",
        fixed_answer_language="cs",
        response_style="normální",
        realtime_voice="marin",
    )
    settings.save = lambda: None  # type: ignore[method-assign]
    return settings


def test_infer_noise_reduction_prefers_headset_near_field() -> None:
    assert infer_noise_reduction("wired_headset") == "near_field"
    assert infer_noise_reduction("bluetooth_headset") == "near_field"
    assert infer_noise_reduction("notebook_builtin") == "far_field"


def test_build_realtime_session_config_uses_browser_webrtc_profile() -> None:
    settings = _settings()
    session = build_realtime_session_config(
        settings,
        {"audio_topology": "notebook_builtin", "browser_language": "cs-CZ"},
    )

    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime"
    assert session["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert session["audio"]["input"]["transcription"]["language"] == "cs"
    assert session["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
    assert session["audio"]["input"]["turn_detection"]["eagerness"] == "low"
    assert session["audio"]["input"]["noise_reduction"]["type"] == "far_field"
    assert session["audio"]["output"]["voice"] == "marin"
    assert session["output_modalities"] == ["audio"]


def test_build_realtime_session_config_uses_fixed_language_and_headset_profile() -> None:
    settings = _settings()
    settings.answer_language_mode = "fixed"
    settings.fixed_answer_language = "de"
    settings.realtime_voice = "cedar"

    session = build_realtime_session_config(
        settings,
        {"audio_topology": "wired_headset", "browser_language": "cs-CZ"},
    )

    assert session["audio"]["input"]["transcription"]["language"] == "de"
    assert session["audio"]["input"]["noise_reduction"]["type"] == "near_field"
    assert session["audio"]["output"]["voice"] == "cedar"


def test_create_app_serves_browser_frontend() -> None:
    app = create_app(_settings())
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "Chatbot Kája" in response.text
    assert "WebRTC" in response.text


def test_runtime_selftest_reports_missing_api_key() -> None:
    app = create_app(_settings())
    client = TestClient(app)

    response = client.post("/api/selftest/runtime", json={"audio_topology": "notebook_builtin"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["checks"][0]["name"] == "api_key"


def test_client_secret_route_uses_session_builder(monkeypatch) -> None:
    settings = _settings()
    settings.openai_api_key_masked = "legacy:yek"
    app = create_app(settings)
    client = TestClient(app)

    captured = {}

    async def fake_mint(api_key: str, session_config: dict):
        captured["api_key"] = api_key
        captured["session_config"] = session_config
        return {
            "client_secret": {"value": "epk_test", "expires_at": 123456},
            "expires_at": 123456,
            "session": session_config,
        }

    monkeypatch.setattr("kajovochat.webapp.server._mint_client_secret", fake_mint)

    response = client.post(
        "/api/realtime/client-secret",
        json={"audio_topology": "wired_headset", "browser_language": "sk-SK"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["client_secret"]["value"] == "epk_test"
    assert captured["api_key"] == "key"
    assert captured["session_config"]["audio"]["input"]["noise_reduction"]["type"] == "near_field"
