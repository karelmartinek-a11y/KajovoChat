from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict


class RealtimeLogWriter:
    """
    Almost-real-time append-only logger (text + jsonl).
    Designed to minimize data loss on crash: frequent flushes.
    """

    def __init__(self, log_dir: Path, session_name: str) -> None:
        self.log_dir = log_dir
        self.session_name = session_name
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.txt_path = self.log_dir / f"{session_name}.txt"
        self.jsonl_path = self.log_dir / f"{session_name}.jsonl"

        # line-buffering
        self._txt_f = open(self.txt_path, "a", encoding="utf-8", buffering=1)
        self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)

        self._thread.start()

    def append(self, record: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # best-effort: drop on overload
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait({"type": "___close___"})
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        try:
            self._txt_f.flush()
            self._jsonl_f.flush()
        except Exception:
            pass
        try:
            self._txt_f.close()
            self._jsonl_f.close()
        except Exception:
            pass

    def _run(self) -> None:
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                item = None

            if item:
                if item.get("type") == "___close___":
                    break

                line = item.get("text_line")
                if line:
                    self._txt_f.write(line.rstrip() + "\n")

                self._jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            if time.time() - last_flush > 0.5:
                try:
                    self._txt_f.flush()
                    self._jsonl_f.flush()
                except Exception:
                    pass
                last_flush = time.time()
