from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..settings import AppSettings, normalize_answer_language_mode, normalize_fixed_language, normalize_response_style, normalize_realtime_voice
from .configuration import build_realtime_session_config, public_settings_payload

OPENAI_CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"


class ApiKeyPayload(BaseModel):
    api_key: str = Field(default="")


class PreferencesPayload(BaseModel):
    answer_language_mode: str = Field(default="follow_input")
    fixed_answer_language: str = Field(default="cs")
    response_style: str = Field(default="normální")
    realtime_voice: str = Field(default="marin")


class ClientHintsPayload(BaseModel):
    audio_topology: str | None = Field(default=None)
    browser_language: str | None = Field(default=None)


class WebAppState:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.server = None


async def _mint_client_secret(api_key: str, session_config: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(20.0, connect=10.0, read=20.0, write=20.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.post(
            OPENAI_CLIENT_SECRETS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"session": session_config},
        )
    if response.status_code >= 400:
        detail = "Nepodařilo se vytvořit dočasný klientský klíč pro Realtime API."
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message") or payload.get("message")
            if message:
                detail = str(message)
        except Exception:
            pass
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="OpenAI API klíč je neplatný nebo chybí oprávnění pro Realtime API.")
        raise HTTPException(status_code=502, detail=detail)
    data = response.json()
    return data


async def _request_shutdown(app: FastAPI) -> None:
    await asyncio.sleep(0.15)
    server = getattr(app.state, "server", None)
    if server is not None:
        server.should_exit = True


def create_app(settings: AppSettings | None = None) -> FastAPI:
    state = WebAppState(settings or AppSettings.load())

    app = FastAPI(title="Chatbot Kája WebRTC", version="0.2.0")
    app.state.runtime = state

    static_dir = Path(__file__).resolve().parent / "static"
    assets_dir = Path(__file__).resolve().parent.parent / "resources" / "assets"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(public_settings_payload(app.state.runtime.settings))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        settings_obj = app.state.runtime.settings
        return {
            "ok": True,
            "has_api_key": bool(settings_obj.openai_api_key),
            "entrypoint": "browser_webrtc",
        }

    @app.post("/api/settings/api-key")
    async def save_api_key(payload: ApiKeyPayload) -> dict[str, Any]:
        key = (payload.api_key or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="API klíč je prázdný.")
        settings_obj = app.state.runtime.settings
        settings_obj.openai_api_key = key
        settings_obj.save()
        return {"ok": True, "has_api_key": True}

    @app.delete("/api/settings/api-key")
    async def delete_api_key() -> dict[str, Any]:
        settings_obj = app.state.runtime.settings
        settings_obj.openai_api_key = ""
        settings_obj.save()
        return {"ok": True, "has_api_key": False}

    @app.post("/api/settings/preferences")
    async def save_preferences(payload: PreferencesPayload) -> JSONResponse:
        settings_obj = app.state.runtime.settings
        settings_obj.answer_language_mode = normalize_answer_language_mode(payload.answer_language_mode)
        settings_obj.fixed_answer_language = normalize_fixed_language(payload.fixed_answer_language)
        settings_obj.response_style = normalize_response_style(payload.response_style)
        settings_obj.realtime_voice = normalize_realtime_voice(payload.realtime_voice)
        settings_obj.save()
        return JSONResponse(public_settings_payload(settings_obj))

    @app.post("/api/realtime/client-secret")
    async def mint_realtime_client_secret(payload: ClientHintsPayload) -> JSONResponse:
        settings_obj = app.state.runtime.settings
        api_key = settings_obj.openai_api_key
        if not api_key:
            raise HTTPException(status_code=400, detail="Nejdřív uložte OpenAI API klíč.")
        hints = payload.model_dump(exclude_none=True)
        session_config = build_realtime_session_config(settings_obj, hints)
        data = await _mint_client_secret(api_key, session_config)
        return JSONResponse(data)

    @app.post("/api/selftest/runtime")
    async def runtime_selftest(payload: ClientHintsPayload) -> dict[str, Any]:
        settings_obj = app.state.runtime.settings
        api_key = settings_obj.openai_api_key
        if not api_key:
            return {
                "ok": False,
                "checks": [
                    {"name": "api_key", "ok": False, "detail": "Chybí uložený OpenAI API klíč."},
                ],
            }
        hints = payload.model_dump(exclude_none=True)
        session_config = build_realtime_session_config(settings_obj, hints)
        data = await _mint_client_secret(api_key, session_config)
        expires_at = None
        if isinstance(data, dict):
            expires_at = data.get("expires_at") or data.get("client_secret", {}).get("expires_at")
        return {
            "ok": True,
            "checks": [
                {"name": "api_key", "ok": True, "detail": "OpenAI API klíč je uložený."},
                {"name": "client_secret", "ok": True, "detail": "Server dokáže vytvořit dočasný Realtime client secret."},
                {
                    "name": "session_profile",
                    "ok": True,
                    "detail": (
                        "Session používá WebRTC browser profil, semantic_vad low, gpt-4o-transcribe a automatickou near/far field noise reduction."
                    ),
                },
            ],
            "expires_at": expires_at,
        }

    @app.post("/api/shutdown")
    async def shutdown(request: Request) -> dict[str, Any]:
        asyncio.create_task(_request_shutdown(request.app))
        return {"ok": True}

    return app
