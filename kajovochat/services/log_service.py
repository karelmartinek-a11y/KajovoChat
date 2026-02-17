from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict


class RealtimeLogWriter:
    """Deterministic JSONL logger + human-readable TXT sidecar."""

    def __init__(self, log_dir: Path, session_name: str) -> None:
        self.log_dir = log_dir
        self.session_name = session_name

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._seq_lock = threading.Lock()
        self._seq = 0

        self.txt_path = self.log_dir / f"{session_name}.txt"
        self.jsonl_path = self.log_dir / f"{session_name}.jsonl"

        # line-buffering
        self._txt_f = open(self.txt_path, "a", encoding="utf-8", buffering=1)
        self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)

        self._thread.start()

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def append(self, record: Dict[str, Any]) -> None:
        if "seq" not in record:
            record["seq"] = self.next_seq()
        record.setdefault("ts_wall", time.time())
        record.setdefault("ts_mono_ns", time.monotonic_ns())
        record.setdefault("session", self.session_name)

        try:
            self._q.put_nowait(record)
        except queue.Full:
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

                # Human-readable line for every record.
                line = item.get("text_line")
                if not line:
                    try:
                        ts = item.get("ts_wall")
                        if isinstance(ts, (int, float)):
                            ts_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
                        else:
                            ts_s = ""
                        t = (item.get("type") or "").upper()
                        msg = item.get("message") or item.get("event") or ""
                        if isinstance(msg, (dict, list)):
                            msg = json.dumps(msg, ensure_ascii=False, sort_keys=True)
                        line = f"{ts_s} seq={item.get('seq')} {t} {msg}".strip()
                    except Exception:
                        line = f"seq={item.get('seq')} {item.get('type')}"

                try:
                    if line:
                        self._txt_f.write(str(line).rstrip() + "\n")
                except Exception:
                    pass

                try:
                    self._jsonl_f.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                except Exception:
                    pass

            if time.time() - last_flush > 0.35:
                try:
                    self._txt_f.flush()
                    self._jsonl_f.flush()
                except Exception:
                    pass
                last_flush = time.time()
