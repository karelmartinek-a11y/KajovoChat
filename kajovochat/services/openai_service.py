from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI


StreamCallback = Callable[[str], None]


@dataclass
class ModelInfo:
    id: str


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
        # If filter became too strict, return original.
        return allow or models

    def transcribe_wav(self, wav_bytes: bytes, model: str = "whisper-1", language_hint: Optional[str] = None) -> str:
        bio = io.BytesIO(wav_bytes)
        # OpenAI SDK expects a file-like with a name
        bio.name = "audio.wav"
        kwargs = {}
        if language_hint:
            # e.g. "cs", "sk", "de", "en", "fr"
            kwargs["language"] = language_hint
        tr = self.client.audio.transcriptions.create(model=model, file=bio, **kwargs)
        return getattr(tr, "text", "") or ""

    def chat_stream(
        self,
        model: str,
        system_prompt: str,
        messages: List[dict],
        on_delta: Optional[StreamCallback] = None,
    ) -> str:
        # Prefer Responses API if available, fallback to Chat Completions.
        try:
            payload = [{"role": "system", "content": system_prompt}] + messages
            stream = self.client.chat.completions.create(
                model=model,
                messages=payload,
                stream=True,
            )
            full = ""
            for event in stream:
                choice = event.choices[0]
                delta = getattr(choice, "delta", None)
                text = getattr(delta, "content", None) if delta else None
                if text:
                    full += text
                    if on_delta:
                        on_delta(text)
            return full.strip()
        except Exception:
            # Fallback: non-streaming
            payload = [{"role": "system", "content": system_prompt}] + messages
            r = self.client.chat.completions.create(model=model, messages=payload)
            txt = r.choices[0].message.content or ""
            if on_delta and txt:
                on_delta(txt)
            return txt.strip()

    def tts_pcm16(
        self,
        text: str,
        model: str,
        voice: str,
        response_format: str = "pcm",
    ) -> bytes:
        if not text.strip():
            return b""
        resp = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        )
        # SDK returns a binary response-like object
        data = None
        if hasattr(resp, "read"):
            data = resp.read()
        elif hasattr(resp, "content"):
            data = resp.content
        elif isinstance(resp, (bytes, bytearray)):
            data = bytes(resp)
        return data or b""
