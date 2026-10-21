# Real-Time Voice AI Agent

A browser-based voice agent built step by step, one phase per branch, from a 5 second
batch pipeline down to under 1 second voice-to-voice. Every phase adds one idea, measures
it, and leaves a small readable diff.

Design reference: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)

Built for teaching at **Outcome School** by **Amit Shekhar** as part of the: [AI and Machine Learning Program](https://outcomeschool.com/program/ai-and-machine-learning)

## How to run

Needs Python 3.11+, Chrome, a [Deepgram](https://deepgram.com) API key (Flux turn detection,
speech to text, and text to speech) and an [OpenRouter](https://openrouter.ai) API key (LLM).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env      # creates .env only if it doesn't already exist
# add DEEPGRAM_API_KEY and OPENROUTER_API_KEY to .env
uvicorn server.main:app --reload
```

Open http://localhost:8000 in Chrome, click **Start** and allow the microphone. P7 pre-renders a greeting and warms provider connections; Flux starts the LLM speculatively at EagerEndOfTurn and discards that request if speech resumes. P8 adds tool calling for weather and sample order lookup/cancellation, with spoken filler while a tool runs. P9 adds silence check-ins, a longer wait after “hold on,” echo filtering, spoken-text cleanup, and conversation-history summaries. The browser still pauses playback while Flux decides whether to interrupt or resume.

Optional `LLM_MODEL` and `LLM_PROVIDERS` settings select an OpenRouter model and inference provider. Compare first-token latency with `python benchmarks/llm_shootout.py`; summarize run traces with `python benchmarks/analyze.py`.

Try asking for the weather in a city, the status of order `1234` or `5678`, or to cancel an order. The agent should confirm the order id before requesting cancellation. Run `python benchmarks/simulate.py --idle` to exercise the silence check-in and automatic goodbye without a microphone.
