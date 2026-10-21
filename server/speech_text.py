import re

MARKDOWN = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"[*_#>]+"), ""),
    (re.compile(r"^\s*[-•]\s+", re.M), ""),
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def spoken_date(match):
    day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
    return f"{day} {MONTHS[month - 1]} {year}" if 1 <= month <= 12 else match.group(0)


NORMALIZE = [
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), spoken_date),
    (re.compile(r"\$(\d[\d,]*(?:\.\d+)?)"), r"\1 dollars"),
    (re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)"), r"\1 rupees"),
    (re.compile(r"€(\d[\d,]*(?:\.\d+)?)"), r"\1 euros"),
    (re.compile(r"(\d)%"), r"\1 percent"),
    (re.compile(r"\s{2,}"), " "),
]


def clean_for_speech(text: str) -> str:
    """Remove common formatting and normalize written forms before TTS."""
    for pattern, replacement in MARKDOWN + NORMALIZE:
        text = pattern.sub(replacement, text)
    return text.strip()
