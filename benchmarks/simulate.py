"""Simulate speech, interruptions, and follow-up turns without a microphone.

    python benchmarks/simulate.py "What is the capital of India?"
    python benchmarks/simulate.py "Tell me about the moon." --interrupt "Wait, tell me about Mars instead."
    python benchmarks/simulate.py "Tell me about the moon." --interrupt "Okay."
    python benchmarks/simulate.py "Cancel order 1234." --then "Yes, cancel it."

Run the server first: uvicorn server.main:app
"""
import argparse
import asyncio
import json
import os
import time

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="What is the capital of India?")
    parser.add_argument("--interrupt")
    parser.add_argument("--then", dest="follow_up")
    args = parser.parse_args()

    question_pcm = await speech(args.question)
    interrupt_pcm = await speech(args.interrupt) if args.interrupt else None
    follow_up_pcm = await speech(args.follow_up) if args.follow_up else None

    async with websockets.connect(SERVER, max_size=None) as ws:
        started_at = time.perf_counter()
        turns = asyncio.Queue()
        greeting_done = asyncio.Event()
        first_audio = asyncio.Event()
        interruption_cleared = asyncio.Event()
        greeting_audio_bytes = 0
        greeting_finished = False
        reply_audio_started_at = None
        pause_started_at = None
        turn_number = 0
        cleared_after_turn = 0

        def elapsed_ms():
            return (time.perf_counter() - started_at) * 1000

        async def read_events():
            nonlocal greeting_audio_bytes, greeting_finished, reply_audio_started_at
            nonlocal pause_started_at, turn_number, cleared_after_turn
            async for message in ws:
                elapsed = elapsed_ms()
                if isinstance(message, bytes):
                    if not greeting_finished:
                        greeting_audio_bytes += len(message)
                        continue
                    if reply_audio_started_at is None:
                        reply_audio_started_at = time.perf_counter()
                        first_audio.set()
                        print(f"  first audio @ {elapsed:.0f} ms")
                    continue

                data = json.loads(message)
                kind = data["type"]
                if kind == "speech_start":
                    print(f"  speech_start @ {elapsed:.0f} ms")
                elif kind == "speech_end":
                    print(f"  speech_end @ {elapsed:.0f} ms (turn detection {data['turn_detection_ms']} ms)")
                elif kind == "transcript":
                    label = "final" if data["final"] else "interim"
                    print(f"  {label} transcript @ {elapsed:.0f} ms: {data['text']}")
                elif kind == "audio_start":
                    reply_audio_started_at = None
                    first_audio.clear()
                    print(f"  audio sample rate: {data['sample_rate']} Hz")
                elif kind == "pause_audio":
                    pause_started_at = time.perf_counter()
                    print(f"  pause_audio @ {elapsed:.0f} ms")
                elif kind == "resume_audio":
                    pause_started_at = None
                    print(f"  resume_audio @ {elapsed:.0f} ms")
                elif kind == "clear_audio":
                    played_ms = max(0, ((pause_started_at or time.perf_counter()) - reply_audio_started_at) * 1000) if reply_audio_started_at else 0
                    await ws.send(json.dumps({"type": "played", "ms": played_ms}))
                    cleared_after_turn = turn_number
                    interruption_cleared.set()
                    print(f"  clear_audio @ {elapsed:.0f} ms, heard about {played_ms:.0f} ms")
                elif kind == "turn_done":
                    if data.get("greeting"):
                        greeting_finished = True
                        greeting_done.set()
                        print(f"  greeting ready @ {elapsed:.0f} ms")
                        continue
                    turn_number += 1
                    print("  transcript:", data["transcript"])
                    print("  reply:", data["reply"])
                    print("  timings:", data["timings"])
                    await turns.put((turn_number, data))
                elif kind == "error":
                    print("  error:", data["message"])

        reader = asyncio.create_task(read_events())

        async def send_question(pcm):
            await stream(ws, silence(600) + pcm + silence(2500))

        await greeting_done.wait()
        await asyncio.sleep(greeting_audio_bytes / 48000)
        await ws.send(json.dumps({"type": "playback_done"}))
        await send_question(question_pcm)
        await first_audio.wait()

        if interrupt_pcm is not None:
            await asyncio.sleep(0.8)
            print(f"  interrupting @ {elapsed_ms():.0f} ms: {args.interrupt}")
            await send_question(interrupt_pcm)
            try:
                await asyncio.wait_for(interruption_cleared.wait(), timeout=8)
            except asyncio.TimeoutError:
                print("  no interruption confirmed; agent likely treated it as a backchannel")
            if interruption_cleared.is_set():
                first_turn = cleared_after_turn
                while True:
                    next_turn, _ = await turns.get()
                    if next_turn > first_turn:
                        break
            else:
                first_turn, _ = await turns.get()
        else:
            first_turn, _ = await turns.get()
        if args.follow_up:
            await asyncio.sleep(2)
            await ws.send(json.dumps({"type": "playback_done"}))
            print(f"  follow-up @ {elapsed_ms():.0f} ms: {args.follow_up}")
            await send_question(follow_up_pcm)
            while True:
                next_turn, _ = await turns.get()
                if next_turn > first_turn:
                    break

        reader.cancel()
        try:
            await reader
        except asyncio.CancelledError:
            pass


asyncio.run(main())
