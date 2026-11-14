"""Generation limits and preliminary prompt misuse prevention.

These are deliberately conservative heuristics, not a complete safety solution.
"""
import re
from .errors import GenerationError

MAX_PROMPT_CHARS = 800
MIN_SECONDS = 0.5
MAX_SECONDS = 15.0

VOICE_PATTERNS = (
    r"\b(?:clone|imitate|impersonate)\s+(?:the\s+)?(?:voice|speaker|person)\b",
    r"\b(?:voice\s+of|sound\s+like|make\s+\w+\s+say)\b",
    r"\b(?:speech|spoken|narrate|narration|text.to.speech|tts)\b",
    r"(?:आवाज़|आवाज)\s+(?:की\s+)?(?:नकल|क्लोन)",
)

def validate_request(prompt: str, duration_seconds: float) -> str:
    if not isinstance(prompt, str):
        raise GenerationError("prompt must be a string")
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        raise GenerationError(f"prompt must contain 1–{MAX_PROMPT_CHARS} characters")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in prompt):
        raise GenerationError("prompt contains invalid control characters")
    if not isinstance(duration_seconds, (int, float)) or not MIN_SECONDS <= duration_seconds <= MAX_SECONDS:
        raise GenerationError(f"duration_seconds must be between {MIN_SECONDS} and {MAX_SECONDS}")
    if any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in VOICE_PATTERNS):
        raise GenerationError("Voice generation/impersonation is not available in this prototype")
    return prompt
