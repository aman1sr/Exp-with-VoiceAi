import asyncio
import re

from fastapi import WebSocket

from . import llm, stt, tts
from .trace import Trace

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
    """One browser connection: stream audio to STT, then reply to each utterance."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.messages = []
        self.stt_ws = None
        self.finals = []
        self.reader = None

    async def start_listening(self):
        """Open STT before the browser starts sending microphone frames."""
        self.stt_ws = await stt.connect()
        self.finals = []
        self.reader = asyncio.create_task(self.read_transcripts())

    async def read_transcripts(self):
        """Collect final transcript segments and publish live interim text."""
        async for text, is_final in stt.transcripts(self.stt_ws):
            if is_final:
                self.finals.append(text)
            live = " ".join(self.finals if is_final else [*self.finals, text])
            await self.ws.send_json({"type": "transcript", "text": live, "final": is_final})

    async def on_audio(self, frame: bytes):
        """Forward each microphone frame to Deepgram while the user is speaking."""
        if self.stt_ws:
            await stt.send_audio(self.stt_ws, frame)

    async def end_of_speech(self):
        """Close STT, then run the existing streamed LLM/TTS response pipeline."""
        trace = Trace()
        await stt.finish(self.stt_ws)
        await self.reader
        self.stt_ws = None
        transcript = " ".join(self.finals)
        trace.mark("stt_done")
        if not transcript:
            await self.ws.send_json({"type": "error", "message": "No speech detected"})
            return

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
                "stt": trace.marks["stt_done"],
                "llm_first_token": trace.duration("stt_done", "llm_first_token"),
                "llm_first_sentence": trace.duration("llm_first_token", "llm_first_sentence"),
                "tts_first_byte": trace.duration("llm_first_sentence", "tts_first_byte"),
                "first_audio": trace.marks["tts_first_byte"],
                "total": trace.marks["audio_sent"],
            },
        })
