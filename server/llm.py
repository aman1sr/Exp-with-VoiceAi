import json
import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
PROVIDERS = [provider for provider in os.environ.get("LLM_PROVIDERS", "groq").split(",") if provider]
client = httpx.AsyncClient(timeout=60)
last_provider = None

SYSTEM_PROMPT = """You are a friendly voice assistant. Your reply will be spoken aloud.
Keep answers short, one or two sentences. Use plain spoken language.
No markdown, no lists, no emojis. Say numbers in words, like "forty five dollars"."""


async def warm_up():
    async for _ in chat_stream([{"role": "user", "content": "Hi"}], max_tokens=1):
        pass


async def close():
    await client.aclose()


async def chat_stream(messages: list[dict], max_tokens: int = 150):
    """Send the conversation so far and yield the reply one piece of text at a time."""
    global last_provider
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if PROVIDERS:
        body["provider"] = {"order": PROVIDERS, "allow_fallbacks": True}
    async with client.stream("POST", OPENROUTER_URL, headers=headers, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            last_provider = data.get("provider", last_provider)
            delta = data["choices"][0]["delta"]
            if delta.get("content"):
                yield delta["content"]
