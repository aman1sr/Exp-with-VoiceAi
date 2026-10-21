import asyncio
import json
import logging
import random
import re
import time

from fastapi import WebSocket

from . import llm, stt, tools, tts
from .speech_text import clean_for_speech
from .trace import Trace
from .vad import VAD

SENTENCE_END = re.compile(r"[.!?]\s")
log = logging.getLogger("uvicorn.error")
BARGE_IN_MIN_MS = 300
BACKCHANNELS = {
    "okay", "ok", "yeah", "yes", "yep", "right", "sure", "alright", "hmm", "hm", "hum", "mhm",
    "mhmm", "mmhmm", "mm", "uh", "huh", "uhhuh", "um", "ah", "oh", "got", "it", "i", "see",
    "cool", "nice", "great", "true",
}
STOP_WORDS = {"stop", "wait", "pause", "enough", "shh"}
MS_PER_BYTE = 1 / 48
MS_PER_WORD = 300
FILLERS = ["Let me check that for you.", "One moment, looking that up.", "Sure, give me a second."]
IDLE_SECONDS = 8
HOLD_SECONDS = 30
KEEP_TURNS = 6
CANNED = {
    "greeting": "Hi! I am your voice assistant. What would you like to know?",
    "still_there": "Are you still there?",
    "goodbye": "It seems you have gone. Goodbye!",
    "take_your_time": "Sure, take your time.",
}
HOLD_ON = re.compile(
    r"\b(hold on|one moment|one second|give me a (second|minute|moment)|just a (second|minute|moment))\b",
    re.I,
)
canned_audio = {}


def real_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower().replace("-", ""))
    return {word for word in words if word not in BACKCHANNELS and len(word) > 1}


async def sentences(text_stream, trace: Trace):
    """Group streamed text pieces into whole sentences, so TTS gets natural chunks."""
    buffer = ""
    async for piece in text_stream:
        if "llm_first_token" not in trace.marks:
            trace.mark("llm_first_token")
        if not isinstance(piece, str):
            yield piece
            continue
        buffer += piece
        while match := SENTENCE_END.search(buffer):
            yield buffer[: match.end()].strip()
            buffer = buffer[match.end():]
    if buffer.strip():
        yield buffer.strip()


async def forward_audio(tts_ws, ws: WebSocket, trace: Trace, sentence_ends: list):
    """Forward audio chunks and record the rendered end position of each sentence."""
    audio_ms = 0
    async for kind, payload in tts.messages(tts_ws):
        if kind == "audio":
            if "tts_first_byte" not in trace.marks:
                trace.mark("tts_first_byte")
            audio_ms += len(payload) * MS_PER_BYTE
            await ws.send_bytes(payload)
        else:
            sentence_ends.append(audio_ms)
    trace.mark("audio_sent")


async def chain(first, rest):
    try:
        yield await first
    except StopAsyncIteration:
        return
    async for piece in rest:
        yield piece


async def prepare():
    for key, text in CANNED.items():
        canned_audio[key] = await tts.synthesize(text)


class Session:
    """One browser connection with continuous STT and barge-in handling."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.messages = []
        self.vad = VAD()
        self.state = "listening"
        self.stt_ws = None
        self.chunk = b""
        self.reader = None
        self.turn = None
        self.next_tts = None
        self.speculation = None
        self.paused = False
        self.user_talking = False
        self.over_agent = False
        self.heard = set()
        self.played = None
        self.sentences = []
        self.sentence_ends = []
        self.idle_task = None
        self.idle_seconds = IDLE_SECONDS
        self.prompted = False
        self.compacting = False

    async def start(self):
        """Play a cached greeting while opening connections needed for the first turn."""
        if not canned_audio:
            await prepare()
        await self.say_canned("greeting")

        self.stt_ws = await stt.connect()
        self.reader = asyncio.create_task(self.read_events())
        self.next_tts = asyncio.create_task(tts.connect())
        asyncio.create_task(llm.warm_up())

    async def say_canned(self, key: str):
        """Play a pre-rendered line as an agent turn without waiting on TTS."""
        text, audio = CANNED[key], canned_audio[key]
        self.state = "agent_turn"
        self.turn = None
        self.paused, self.heard = False, set()
        self.sentences, self.sentence_ends = [text], [len(audio) * MS_PER_BYTE]
        self.messages.append({"role": "assistant", "content": text})
        await self.ws.send_json({"type": "audio_start", "sample_rate": tts.SAMPLE_RATE})
        await self.ws.send_bytes(audio)
        await self.ws.send_json({"type": "turn_done", "greeting": key == "greeting", "canned": key})

    async def idle_watch(self):
        """Check in after silence, then end a session that remains inactive."""
        await asyncio.sleep(self.idle_seconds)
        self.idle_task = None
        if self.prompted:
            await self.say_canned("goodbye")
            await asyncio.sleep(len(canned_audio["goodbye"]) * MS_PER_BYTE / 1000)
            await self.ws.close()
        else:
            self.prompted = True
            await self.say_canned("still_there")

    async def tts_connection(self):
        tts_ws = await self.next_tts
        self.next_tts = asyncio.create_task(tts.connect())
        if tts_ws.close_code is not None:
            tts_ws = await tts.connect()
        return tts_ws

    async def on_audio(self, frame: bytes):
        """Forward every microphone frame and pause playback while checking for an interruption."""
        event = self.vad.process(frame)
        self.chunk += frame
        if len(self.chunk) >= len(frame) * stt.CHUNK_FRAMES:
            await stt.send_audio(self.stt_ws, self.chunk)
            self.chunk = b""

        if self.state == "agent_turn":
            if not self.paused and self.vad.speaking and self.vad.speech_ms >= BARGE_IN_MIN_MS:
                self.paused = True
                await self.ws.send_json({"type": "pause_audio", "after_ms": self.vad.speech_ms})
            elif self.paused and event == "speech_end" and not self.heard:
                await self.resume()

    async def read_events(self):
        """Use Flux turn events to update the session and publish live text."""
        async for event, transcript in stt.events(self.stt_ws):
            if event == "StartOfTurn":
                self.user_talking, self.over_agent, self.heard = True, self.state == "agent_turn", set()
            elif event == "EndOfTurn":
                self.user_talking = False

            if self.state == "listening" and event == "StartOfTurn":
                self.state = "user_speaking"
                if self.idle_task:
                    self.idle_task.cancel()
                    self.idle_task = None
                await self.ws.send_json({"type": "speech_start"})

            if self.state == "agent_turn" and transcript:
                real = self.heard = real_words(transcript)
                if len(real) >= 2 or (real and event == "EndOfTurn"):
                    await self.interrupt()
                elif event == "EndOfTurn":
                    await self.resume()

            if self.state != "user_speaking":
                continue

            final = event == "EndOfTurn"
            await self.ws.send_json({"type": "transcript", "text": transcript, "final": final})
            if event == "EagerEndOfTurn" and transcript:
                self.speculate(transcript)
            elif event == "TurnResumed":
                self.drop_speculation()
            elif final and transcript:
                real = real_words(transcript)
                if (self.over_agent and not real) or (real and real <= STOP_WORDS):
                    self.drop_speculation()
                    self.state = "listening"
                    self.idle_task = asyncio.create_task(self.idle_watch())
                    continue
                await self.end_of_turn(transcript)

    def speculate(self, transcript: str):
        self.drop_speculation()
        stream = llm.chat_stream([*self.messages, {"role": "user", "content": transcript}])
        first = asyncio.create_task(stream.__anext__())
        self.speculation = (transcript, stream, first)

    def drop_speculation(self):
        if self.speculation:
            self.speculation[2].cancel()
            self.speculation = None

    def take_speculation(self, transcript: str):
        if not self.speculation or self.speculation[0] != transcript:
            self.drop_speculation()
            return None
        _, stream, first = self.speculation
        self.speculation = None
        return chain(first, stream)

    async def resume(self):
        if self.paused:
            self.paused = False
            await self.ws.send_json({"type": "resume_audio"})

    async def interrupt(self):
        """Cancel the current reply and save only the portion the user heard."""
        if self.turn is not None and not self.turn.done():
            self.turn.cancel()
        self.paused = False
        self.state = "user_speaking"
        self.played = asyncio.get_running_loop().create_future()
        await self.ws.send_json({"type": "clear_audio"})
        self.keep_heard(await self.played)

    async def on_played(self, played_ms: float):
        if self.played is not None and not self.played.done():
            self.played.set_result(played_ms)

    def keep_heard(self, played_ms: float):
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        heard_sentences = [
            sentence for sentence, end_ms in zip(self.sentences, self.sentence_ends)
            if end_ms <= played_ms
        ]
        if len(heard_sentences) < len(self.sentences):
            start_ms = self.sentence_ends[len(heard_sentences) - 1] if heard_sentences else 0
            words = self.sentences[len(heard_sentences)].split()
            if len(self.sentence_ends) > len(heard_sentences):
                span_ms = self.sentence_ends[len(heard_sentences)] - start_ms
            else:
                span_ms = len(words) * MS_PER_WORD
            fraction = min(1, (played_ms - start_ms) / span_ms) if span_ms else 0
            partial = " ".join(words[:round(len(words) * fraction)])
            if partial:
                heard_sentences.append(partial + "...")
        if heard_sentences:
            self.messages.append({"role": "assistant", "content": " ".join(heard_sentences)})

    async def end_of_turn(self, transcript: str):
        """Start the response as soon as Flux returns its final turn transcript."""
        self.state = "agent_turn"
        last_speech_at = self.vad.last_speech_at
        turn_detection_ms = round((time.perf_counter() - last_speech_at) * 1000) if last_speech_at else 0
        self.vad.reset()
        self.paused, self.heard = False, set()
        await self.ws.send_json({"type": "speech_end", "turn_detection_ms": turn_detection_ms})
        self.prompted = False
        self.idle_seconds = IDLE_SECONDS

        last_reply = (self.messages[-1].get("content") or "") if self.messages else ""
        if len(transcript.split()) >= 4 and transcript.lower().rstrip(".?!") in last_reply.lower():
            self.state = "listening"
            self.idle_task = asyncio.create_task(self.idle_watch())
            return
        if HOLD_ON.search(transcript):
            self.idle_seconds = HOLD_SECONDS
            self.messages.append({"role": "user", "content": transcript})
            await self.say_canned("take_your_time")
            return
        self.turn = asyncio.create_task(self.run_turn(transcript, turn_detection_ms))

    async def playback_done(self):
        """Return to listening, unless Flux has already detected the next user turn."""
        self.vad.reset()
        self.state = "user_speaking" if self.user_talking else "listening"
        if self.state == "listening":
            self.idle_task = asyncio.create_task(self.idle_watch())
        if not self.compacting:
            asyncio.create_task(self.compact_history())

    async def compact_history(self):
        """Summarize older user turns to keep long conversations within the context window."""
        user_turns = [index for index, message in enumerate(self.messages) if message["role"] == "user"]
        if len(user_turns) <= KEEP_TURNS:
            return
        self.compacting = True
        cut = user_turns[-KEEP_TURNS]
        history = list(self.messages)
        boundary = history[cut]
        try:
            summary = await llm.summarize(history[:cut])
            current_cut = next(
                (index for index, message in enumerate(self.messages) if message is boundary),
                None,
            )
            if current_cut is not None:
                self.messages = [
                    {"role": "system", "content": "Earlier in this conversation: " + summary},
                    *self.messages[current_cut:],
                ]
        except Exception:
            log.exception("conversation history summarization failed")
        finally:
            self.compacting = False

    async def run_turn(self, transcript: str, turn_detection_ms: int):
        """Run streamed LLM/TTS, executing requested tools between model rounds."""
        trace = Trace()
        trace.mark("stt_done")
        self.messages.append({"role": "user", "content": transcript})
        await self.ws.send_json({"type": "audio_start", "sample_rate": tts.SAMPLE_RATE})
        self.sentences = []
        self.sentence_ends = []
        tts_ws = None
        forward = None
        stream = self.take_speculation(transcript)
        speculated = stream is not None
        if not speculated:
            stream = llm.chat_stream(self.messages)
        tool_ms = 0
        written = []

        try:
            tts_ws = await self.tts_connection()
            forward = asyncio.create_task(forward_audio(tts_ws, self.ws, trace, self.sentence_ends))
            while True:
                calls = []
                text = ""
                async for item in sentences(stream, trace):
                    if isinstance(item, llm.ToolCall):
                        calls.append(item)
                        continue
                    if "llm_first_sentence" not in trace.marks:
                        trace.mark("llm_first_sentence")
                    spoken = clean_for_speech(item)
                    if spoken != item:
                        written.append(item)
                    item = spoken
                    if not item:
                        continue
                    self.sentences.append(item)
                    text += item + " "
                    await tts.speak(tts_ws, item)
                if not calls:
                    break

                filler = random.choice(FILLERS)
                if "llm_first_sentence" not in trace.marks:
                    trace.mark("llm_first_sentence")
                self.sentences.append(filler)
                await tts.speak(tts_ws, filler)

                results = []
                for call in calls:
                    started = time.perf_counter()
                    result = await tools.run(call.name, json.loads(call.arguments or "{}"))
                    tool_ms += round((time.perf_counter() - started) * 1000)
                    results.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    })
                self.messages += [
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [call.as_message() for call in calls],
                    },
                    *results,
                ]
                stream = llm.chat_stream(self.messages)
            trace.mark("llm_done")

            await tts.close(tts_ws)
            await forward
        except asyncio.CancelledError:
            if forward is not None:
                forward.cancel()
                try:
                    await forward
                except (asyncio.CancelledError, Exception):
                    pass
            if tts_ws is not None:
                await tts_ws.close()
            return
        except Exception:
            log.exception("turn failed")
            if forward is not None:
                forward.cancel()
                try:
                    await forward
                except (asyncio.CancelledError, Exception):
                    pass
            if tts_ws is not None:
                await tts_ws.close()
            await self.ws.send_json({"type": "error", "message": "Something went wrong, please try again"})
            return

        reply = " ".join(self.sentences)
        self.messages.append({"role": "assistant", "content": reply})
        trace.marks["turn_detection"] = turn_detection_ms
        trace.marks["first_audio"] = turn_detection_ms + trace.marks["tts_first_byte"]
        trace.save()

        await self.ws.send_json({
            "type": "turn_done",
            "transcript": transcript,
            "reply": reply,
            "provider": llm.last_provider,
            "speculated": speculated,
            "tool_ms": tool_ms,
            "written": " ".join(written),
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
