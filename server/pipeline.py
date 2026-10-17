import asyncio
import re
import time

from fastapi import WebSocket

from . import llm, stt, tts
from .trace import Trace
from .vad import VAD

SENTENCE_END = re.compile(r"[.!?]\s")


async def sentences(text_stream, trace: Trace):
    """Group streamed text pieces into whole sentences, so TTS gets natural chunks."""
    buffer = ""
    async for piece in text_stream:
        if "llm_first_token" not in trace.marks:
            trace.mark("llm_first_token")
        buffer += piece
        while match := SENTENCE_END.search(buffer):
            yield buffer[: match.end()].strip()
            buffer = buffer[match.end():]
    if buffer.strip():
        yield buffer.strip()


async def forward_audio(tts_ws, ws: WebSocket, trace: Trace):
    """Copy TTS audio to the browser as soon as each chunk arrives."""
    async for chunk in tts.audio_chunks(tts_ws):
        if "tts_first_byte" not in trace.marks:
            trace.mark("tts_first_byte")
        await ws.send_bytes(chunk)
    trace.mark("audio_sent")


class Session:
    """One browser connection with continuous STT and model-based turn detection."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.messages = []
        self.vad = VAD()
        self.state = "listening"
        self.stt_ws = None
        self.chunk = b""
        self.reader = None

    async def start(self):
        """Open the continuous STT connection for this browser session."""
        self.stt_ws = await stt.connect()
        self.reader = asyncio.create_task(self.read_events())

    async def on_audio(self, frame: bytes):
        """Send continuous audio to Flux in its preferred 80 ms chunks."""
        self.vad.process(frame)
        self.chunk += frame
        if len(self.chunk) >= len(frame) * stt.CHUNK_FRAMES:
            await stt.send_audio(self.stt_ws, self.chunk)
            self.chunk = b""

    async def read_events(self):
        """Use Flux turn events to update the session and publish live text."""
        async for event, transcript in stt.events(self.stt_ws):
            if self.state == "listening" and event == "StartOfTurn":
                self.state = "user_speaking"
                await self.ws.send_json({"type": "speech_start"})

            if self.state != "user_speaking":
                continue

            final = event == "EndOfTurn"
            await self.ws.send_json({"type": "transcript", "text": transcript, "final": final})
            if final and transcript:
                await self.end_of_turn(transcript)

    async def end_of_turn(self, transcript: str):
        """Start the response as soon as Flux returns its final turn transcript."""
        self.state = "agent_turn"
        last_speech_at = self.vad.last_speech_at
        turn_detection_ms = round((time.perf_counter() - last_speech_at) * 1000) if last_speech_at else 0
        await self.ws.send_json({"type": "speech_end", "turn_detection_ms": turn_detection_ms})
        asyncio.create_task(self.run_turn(transcript, turn_detection_ms))

    async def playback_done(self):
        """Reset local speech timing after the browser finishes playback."""
        self.vad.reset()
        self.state = "listening"

    async def run_turn(self, transcript: str, turn_detection_ms: int):
        """Run the existing streamed LLM/TTS response pipeline for a completed turn."""
        trace = Trace()
        trace.mark("stt_done")
        self.messages.append({"role": "user", "content": transcript})
        await self.ws.send_json({"type": "audio_start", "sample_rate": tts.SAMPLE_RATE})
        tts_ws = await tts.connect()
        forward = asyncio.create_task(forward_audio(tts_ws, self.ws, trace))

        reply = ""
        async for sentence in sentences(llm.chat_stream(self.messages), trace):
            if not reply:
                trace.mark("llm_first_sentence")
            reply += sentence + " "
            await tts.speak(tts_ws, sentence)
        trace.mark("llm_done")

        await tts.close(tts_ws)
        await forward
        self.messages.append({"role": "assistant", "content": reply.strip()})
        trace.save()

        await self.ws.send_json({
            "type": "turn_done",
            "transcript": transcript,
            "reply": reply.strip(),
            "timings": {
                "turn_detection": turn_detection_ms,
                "stt": trace.marks["stt_done"],
                "llm_first_token": trace.duration("stt_done", "llm_first_token"),
                "llm_first_sentence": trace.duration("llm_first_token", "llm_first_sentence"),
                "tts_first_byte": trace.duration("llm_first_sentence", "tts_first_byte"),
                "first_audio": turn_detection_ms + trace.marks["tts_first_byte"],
                "total": turn_detection_ms + trace.marks["audio_sent"],
            },
        })
