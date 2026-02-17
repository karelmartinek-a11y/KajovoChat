from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from ..services.openai_service import OpenAIService
from ..services.audio_service import AudioPlayer

_DEFAULT_TEXT = {
    "cs": "Ahoj, tady je ukázka vybraného hlasu. Jak vám mohu pomoci?",
    "sk": "Ahoj, toto je ukážka vybraného hlasu. Ako vám môžem pomôcť?",
    "en": "Hello. This is a short preview of the selected voice. How can I help you?",
    "de": "Hallo. Das ist eine kurze Vorschau der ausgewählten Stimme. Wie kann ich helfen?",
    "fr": "Bonjour. Voici un court aperçu de la voix sélectionnée. Comment puis-je aider ?",
}

def _eprint(msg: str) -> None:
    try:
        sys.stderr.write(msg.rstrip() + "\n")
        sys.stderr.flush()
    except Exception:
        pass

def _oprint(msg: str) -> None:
    try:
        sys.stdout.write(msg.rstrip() + "\n")
        sys.stdout.flush()
    except Exception:
        pass

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--output_device", type=int, default=-1)
    p.add_argument("--lang", default="cs")
    p.add_argument("--text", default="")
    args = p.parse_args(argv)

    api_key = (os.getenv("KAJOVOCHAT_API_KEY") or "").strip()
    if not api_key:
        _eprint("Chybí API key (KAJOVOCHAT_API_KEY).")
        return 2

    text = (args.text or "").strip() or _DEFAULT_TEXT.get((args.lang or "cs").strip().lower(), _DEFAULT_TEXT["cs"])
    out_dev = None if args.output_device is None or int(args.output_device) < 0 else int(args.output_device)

    svc: Optional[OpenAIService] = None
    player: Optional[AudioPlayer] = None
    try:
        svc = OpenAIService(api_key)
        pcm = svc.tts_pcm16(text=text, model=args.model, voice=args.voice, speed=float(args.speed), response_format="pcm")
        if not pcm:
            _eprint("Ukázku se nepodařilo vygenerovat (prázdná audio odpověď).")
            return 3

        player = AudioPlayer(samplerate=24000, device=out_dev)
        player.play_pcm16(pcm)

        _oprint(json.dumps({"ok": True, "bytes": len(pcm)}, ensure_ascii=False))
        return 0
    except Exception as e:
        _eprint(str(e))
        # also print a compact traceback marker for debugging
        try:
            tb = traceback.format_exc(limit=12)
            _eprint(tb)
        except Exception:
            pass
        return 1
    finally:
        try:
            if player:
                player.stop()
        except Exception:
            pass
        try:
            if svc:
                svc.close()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())
