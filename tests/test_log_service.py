from __future__ import annotations

import tempfile
from pathlib import Path

from kajovochat.services.log_service import RealtimeLogWriter


def test_log_writer_writes_files() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        log_dir = Path(temp_dir)
        writer = RealtimeLogWriter(log_dir, "session_test", queue_size=8)
        writer.append({"type": "user", "text": "ahoj"})
        writer.append({"type": "assistant", "text": "nazdar"})
        writer.close()

        txt = (log_dir / "session_test.txt").read_text(encoding="utf-8")
        jsonl = (log_dir / "session_test.jsonl").read_text(encoding="utf-8")

        assert "USER" in txt
        assert '"type":"assistant"' in jsonl
        assert writer.last_error == ""


def test_log_writer_records_write_error() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        writer = RealtimeLogWriter(Path(temp_dir), "session_test_error", queue_size=8)
        writer._txt_f.close()
        writer.append({"type": "user", "text": "ahoj"})
        writer.close()

        assert writer.last_error
