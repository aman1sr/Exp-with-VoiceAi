import json
import os

import websockets

# Flux transcribes speech and determines when the user's turn is complete.
DEEPGRAM_URL = "wss://api.deepgram.com/v2/listen"
EOT_THRESHOLD = 0.7
CHUNK_FRAMES = 4


async def connect():
    """Open one continuous PCM16 16kHz mono Flux session."""
    params = f"model=flux-general-en&encoding=linear16&sample_rate=16000&eot_threshold={EOT_THRESHOLD}"
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    return await websockets.connect(f"{DEEPGRAM_URL}?{params}", additional_headers=headers)


async def send_audio(stt_ws, chunk: bytes):
    await stt_ws.send(chunk)


async def events(stt_ws):
    """Yield Flux turn events and their current transcript."""
    async for message in stt_ws:
        data = json.loads(message)
        if data["type"] == "TurnInfo":
            yield data["event"], data["transcript"]
