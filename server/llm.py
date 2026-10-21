import json
import os
from dataclasses import dataclass

import httpx

from . import tools

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
PROVIDERS = [provider for provider in os.environ.get("LLM_PROVIDERS", "groq").split(",") if provider]
client = httpx.AsyncClient(timeout=60)
last_provider = None

SYSTEM_PROMPT = """You are a friendly voice assistant. Your reply will be spoken aloud.
Keep answers short, one or two sentences. Use plain spoken language.
No markdown, no lists, no emojis. Say numbers in words, like "forty five dollars".
Use the tools to look things up instead of guessing. Before cancelling an order, say back the
order id and ask the user to confirm, and call cancel_order only after they say yes.
If a tool returns an error, say sorry in plain words and offer what you can do instead.
Never read out error codes or technical details."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def as_message(self):
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


async def warm_up():
    async for _ in chat_stream([{"role": "user", "content": "Hi"}], max_tokens=1):
        pass


async def close():
    await client.aclose()


async def summarize(messages: list[dict]) -> str:
    """Compress older conversation turns while preserving useful names and details."""
    transcript = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in messages
        if message.get("content")
    )
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "Summarize this conversation in three short sentences, keeping names, numbers, "
            "order ids and anything the user asked for:\n\n" + transcript,
        }],
        "max_tokens": 150,
    }
    if PROVIDERS:
        body["provider"] = {"order": PROVIDERS, "allow_fallbacks": True}
    response = await client.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json=body,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def chat_stream(messages: list[dict], max_tokens: int = 150):
    """Yield text pieces, followed by any complete streamed tool calls."""
    global last_provider
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "max_tokens": max_tokens,
        "stream": True,
        "tools": tools.DEFINITIONS,
    }
    calls = {}
    if PROVIDERS:
        body["provider"] = {"order": PROVIDERS, "allow_fallbacks": True}
    async with client.stream("POST", OPENROUTER_URL, headers=headers, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            last_provider = data.get("provider", last_provider)
            if not data.get("choices"):
                continue
            delta = data["choices"][0]["delta"]
            if delta.get("content"):
                yield delta["content"]
            for part in delta.get("tool_calls") or []:
                function = part.get("function") or {}
                index = part["index"]
                if index not in calls:
                    calls[index] = ToolCall(part.get("id") or "", function.get("name") or "")
                else:
                    if part.get("id"):
                        calls[index].id = part["id"]
                    calls[index].name += function.get("name") or ""
                calls[index].arguments += function.get("arguments") or ""
    for call in calls.values():
        yield call
