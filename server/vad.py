import time
from typing import Optional

from pysilero_vad import SileroVoiceActivityDetector

SPEECH_THRESHOLD = 0.5
SILENCE_MS = 700
CHUNK_MS = 32


class VAD:
    """Detect speech boundaries in incoming 16kHz PCM16 audio frames."""

    def __init__(self):
        self.model = SileroVoiceActivityDetector()
        self.buffer = b""
        self.speaking = False
        self.silence_ms = 0
        self.last_speech_at = None

    def process(self, frame: bytes) -> Optional[str]:
        self.buffer += frame
        event = None
        chunk_bytes = self.model.chunk_bytes()

        while len(self.buffer) >= chunk_bytes:
            chunk, self.buffer = self.buffer[:chunk_bytes], self.buffer[chunk_bytes:]
            is_speech = self.model.process_chunk(chunk) > SPEECH_THRESHOLD
            if is_speech:
                self.silence_ms = 0
                self.last_speech_at = time.perf_counter()
                if not self.speaking:
                    self.speaking = True
                    event = "speech_start"
            elif self.speaking:
                self.silence_ms += CHUNK_MS
                if self.silence_ms >= SILENCE_MS:
                    self.speaking = False
                    event = "speech_end"
        return event

    def reset(self):
        self.model.reset()
        self.buffer = b""
        self.speaking = False
        self.silence_ms = 0
        self.last_speech_at = None
