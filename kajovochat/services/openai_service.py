from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError


StreamCallback = Callable[[str], None]


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None  # e.g. "cs", "en", ...


class InvalidApiKeyError(RuntimeError):
    pass


class OpenAIService:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()
        self.client = OpenAI(api_key=self.api_key)

    def list_models(self) -> List[str]:
        models = self.client.models.list()
        out = []
        for m in getattr(models, "data", []) or []:
            mid = getattr(m, "id", None)
            if mid:
                out.append(mid)
        out.sort()
        return out

    @staticmethod
    def filter_chat_models(models: List[str]) -> List[str]:
        # Conservative filter: likely text chat-capable models.
        allow = []
        for m in models:
            ml = m.lower()
            if ml.startswith(("gpt", "o")) and "whisper" not in ml and "tts" not in ml and "audio" not in ml and "transcribe" not in ml:
                allow.append(m)
        return allow or models

    def _with_retry(self, fn, *, op: str):
        # Transient errors: max 1 retry with exponential backoff.
        # Invalid key: no retry.
        try:
            return fn()
        except AuthenticationError as e:
            raise InvalidApiKeyError("Neplatný API key.") from e
        except (APITimeoutError, APIConnectionError) as e:
            time.sleep(0.35)
            try:
                return fn()
            except Exception as e2:
                raise RuntimeError(f"{op} selhalo (timeout/connection).") from e2
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in (502, 503):
                time.sleep(0.35)
                try:
                    return fn()
                except Exception as e2:
                    raise RuntimeError(f"{op} selhalo ({status}).") from e2
            raise

    def transcribe_wav(self, wav_bytes: bytes, *, language_hint: Optional[str] = None) -> TranscriptionResult:
        # STT is fixed to Whisper.
        bio = io.BytesIO(wav_bytes)
        bio.name = "audio.wav"

        kwargs = {}
        if language_hint:
            kwargs["language"] = language_hint

        def _call():
            return self.client.audio.transcriptions.create(model="whisper-1", file=bio, **kwargs)

        tr = self._with_retry(_call, op="Přepis (Whisper)")
        text = getattr(tr, "text", "") or ""

        # language is not always present; best-effort
        lang = getattr(tr, "language", None)
        if isinstance(lang, str):
            lang = lang.strip().lower() or None
        else:
            lang = None

        return TranscriptionResult(text=text, language=lang)

    def chat_stream(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: List[dict],
        temperature: float = 0.3,
        max_output_tokens: int = 512,
        on_delta: Optional[StreamCallback] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        payload = [{"role": "system", "content": system_prompt}] + messages

        def _call_stream():
            return self.client.chat.completions.create(
                model=model,
                messages=payload,
                temperature=float(temperature),
                max_tokens=int(max_output_tokens),
                stream=True,
            )

        stream = self._with_retry(_call_stream, op="Chat")

        full = ""
        for event in stream:
            if cancel and cancel():
                break
            choice = event.choices[0]
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                full += text
                if on_delta:
                    on_delta(text)

        return full.strip()

    def tts_pcm16(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        speed: float = 1.0,
        response_format: str = "pcm",
    ) -> bytes:
        if not text.strip():
            return b""

        def _call():
            return self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=float(speed),
            )

        resp = self._with_retry(_call, op="TTS")
        data = None
        if hasattr(resp, "read"):
            data = resp.read()
        elif hasattr(resp, "content"):
            data = resp.content
        elif isinstance(resp, (bytes, bytearray)):
            data = bytes(resp)
        return data or b""
