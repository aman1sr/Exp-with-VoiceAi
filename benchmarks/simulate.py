"""Simulated user for benchmarking without a microphone.

Generates the question as speech with Deepgram TTS (16kHz, like our mic), streams it to the
server one 20ms frame at a time, and prints every event with a timestamp.

    python benchmarks/simulate.py "What is the capital of India?"

Run the server first: uvicorn server.main:app
"""
import asyncio
import json
import os
import sys
import time

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

QUESTION = next((arg for arg in sys.argv[1:] if not arg.startswith("--")), "What is the capital of India?")
SERVER = "ws://127.0.0.1:8000/ws"


def silence(ms):
    return bytes(32 * ms)


async def speech(text):
    params = {
        "model": "aura-2-thalia-en",
        "encoding": "linear16",
        "sample_rate": 16000,
        "container": "none",
    }
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.deepgram.com/v1/speak",
            params=params,
            headers=headers,
            json={"text": text},
        )
        response.raise_for_status()
        return response.content


async def stream(ws, pcm):
    start = time.perf_counter()
    for number, offset in enumerate(range(0, len(pcm), 640)):
        await ws.send(pcm[offset:offset + 640])
        await asyncio.sleep(max(0, start + (number + 1) * 0.02 - time.perf_counter()))


async def main():
    pcm = await speech(QUESTION)
    audio = silence(500) + pcm + silence(1500)
    estimated_speech_end = 500 + len(pcm) / 32

    async with websockets.connect(SERVER, max_size=None) as ws:
        start = time.perf_counter()
        first_audio = None

        async def read_events():
            nonlocal first_audio
            async for message in ws:
                elapsed = (time.perf_counter() - start) * 1000
                if isinstance(message, bytes):
                    if first_audio is None:
                        first_audio = elapsed
                        print(
                            f"  first audio @ {first_audio:.0f} ms"
                            f"  ->  V2V from estimated speech end: {first_audio - estimated_speech_end:.0f} ms"
                        )
                    continue

                data = json.loads(message)
                if data["type"] in ("speech_start", "speech_end"):
                    print(f"  {data['type']} @ {elapsed:.0f} ms")
                elif data["type"] == "transcript":
                    kind = "final" if data["final"] else "interim"
                    print(f"  {kind} transcript @ {elapsed:.0f} ms: {data['text']}")
                elif data["type"] == "audio_start":
                    print(f"  audio sample rate: {data['sample_rate']} Hz")
                elif data["type"] == "turn_done":
                    print("  transcript:", data["transcript"])
                    print("  reply:", data["reply"])
                    print("  timings:", data["timings"])
                    return
                elif data["type"] == "error":
                    print("  error:", data["message"])
                    return

        reader = asyncio.create_task(read_events())
        await stream(ws, audio)
        await reader


asyncio.run(main())
