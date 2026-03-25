class AudioProcessor:
    """Minimal test stub for environments without native webrtc-audio-processing wheels."""

    def __init__(self, *, enable_aec=True, enable_ns=False, enable_agc=False, enable_vad=False):
        self.enable_aec = bool(enable_aec)
        self.enable_ns = bool(enable_ns)
        self.enable_agc = bool(enable_agc)
        self.enable_vad = bool(enable_vad)
        self._delay_ms = 0

    def set_stream_format(self, in_rate, in_channels, out_rate, out_channels):
        self._stream_format = (int(in_rate), int(in_channels), int(out_rate), int(out_channels))

    def set_reverse_stream_format(self, rate, channels):
        self._reverse_stream_format = (int(rate), int(channels))

    def set_stream_delay(self, delay_ms):
        self._delay_ms = max(0, int(delay_ms))

    def process_reverse_stream(self, frame: bytes) -> bytes:
        return bytes(frame)

    def process_stream(self, frame: bytes) -> bytes:
        return bytes(frame)
