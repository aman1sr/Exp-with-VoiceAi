import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
MODELS = [
    ("anthropic/claude-haiku-4.5", None),
    ("openai/gpt-4.1-mini", None),
    ("google/gemini-2.5-flash-lite", None),
    ("meta-llama/llama-3.3-70b-instruct", None),
    ("meta-llama/llama-3.3-70b-instruct", ["groq"]),
    ("openai/gpt-oss-120b", ["groq"]),
]
PROMPT = [{"role": "user", "content": "Say hello in one short sentence."}]


async def first_token_ms(client, model, providers):
    body = {"model": model, "messages": PROMPT, "max_tokens": 40, "stream": True}
    if providers:
        body["provider"] = {"order": providers, "allow_fallbacks": True}
    started = time.perf_counter()
    async with client.stream("POST", URL, headers=HEADERS, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line[6:])["choices"][0]["delta"]
                if delta.get("content"):
                    return round((time.perf_counter() - started) * 1000)
    return None


async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        for model, providers in MODELS:
            runs = [await first_token_ms(client, model, providers) for _ in range(3)]
            print(f"{model:40s} {str(providers or 'auto'):12s} {runs} ms")


asyncio.run(main())
