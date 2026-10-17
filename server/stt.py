import json
import os

import websockets

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


async def connect():
    """Open live transcription for PCM16 16kHz mono audio."""
    params = "model=nova-3&encoding=linear16&sample_rate=16000&channels=1&interim_results=true&smart_format=true"
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    return await websockets.connect(f"{DEEPGRAM_URL}?{params}", additional_headers=headers)


async def send_audio(stt_ws, frame: bytes):
    await stt_ws.send(frame)


async def finish(stt_ws):
    """Signal that the current utterance is complete and request final results."""
    await stt_ws.send(json.dumps({"type": "CloseStream"}))


async def transcripts(stt_ws):
    """Yield non-empty transcript text and whether each result is final."""
    async for message in stt_ws:
        data = json.loads(message)
        if data["type"] != "Results":
            continue
        text = data["channel"]["alternatives"][0]["transcript"]
        if text:
            yield text, data["is_final"]
