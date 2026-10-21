import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "traces.jsonl"
with open(path) as trace_file:
    turns = [json.loads(line) for line in trace_file if line.strip()]

stages = [
    ("turn_detection", "Turn detection (last word -> decision)"),
    ("first_audio", "First audio, from real end of speech"),
    ("llm_first_token", "  decision -> LLM first token"),
    ("llm_first_sentence", "  decision -> LLM first sentence"),
    ("tts_first_byte", "  decision -> TTS first byte"),
    ("audio_sent", "  decision -> last audio sent"),
]


def percentile(values, percentile):
    values = sorted(values)
    return values[min(len(values) - 1, round(percentile / 100 * (len(values) - 1)))]


print(f"{len(turns)} turns from {path}\n")
print(f"{'stage':40s} {'p50':>6s} {'p95':>6s} {'p99':>6s}")
for key, label in stages:
    values = [turn[key] for turn in turns if key in turn]
    if values:
        p50 = percentile(values, 50)
        p95 = percentile(values, 95)
        p99 = percentile(values, 99)
        print(f"{label:40s} {p50:6.0f} {p95:6.0f} {p99:6.0f}")
