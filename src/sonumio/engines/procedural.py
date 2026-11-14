"""A deterministic prompt-to-audio DSP baseline (NOT an AI model)."""
import numpy as np
from ..audio import AudioResult, SAMPLE_RATE
from ..errors import UnsupportedPromptError

TOKENS = {
    "rain": ("rain", "rainfall", "बारिश", "वर्षा", "बरसात"),
    "thunder": ("thunder", "storm", "बिजली", "गरज", "तूफान"),
    "wind": ("wind", "breeze", "हवा", "आंधी"),
    "forest": ("forest", "bird", "birds", "jungle", "जंगल", "पक्षी", "चिड़िया"),
    "water": ("ocean", "sea", "wave", "waves", "समुद्र", "लहर"),
    "drone": ("ambient", "drone", "synth", "cinematic", "संगीत", "धुन"),
}

def _smooth_noise(rng: np.random.Generator, n: int, kernel: int) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    k = max(1, kernel)
    if k == 1:
        return noise
    filtered = np.convolve(noise, np.ones(k, dtype=np.float32) / k, mode="same")
    return filtered.astype(np.float32)

def _rain(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    n, sr = len(t), SAMPLE_RATE
    hiss = rng.standard_normal(n).astype(np.float32) * 0.055
    impulses = np.zeros(n, dtype=np.float32)
    indices = rng.integers(0, n, size=int(110 * len(t) / sr))
    np.add.at(impulses, indices, rng.uniform(0.1, 0.6, size=len(indices)))
    drops = np.convolve(impulses, np.exp(-np.arange(180) / 38).astype(np.float32), mode="full")[:n]
    return hiss + 0.08 * drops

def _thunder(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    n = len(t)
    rumble = _smooth_noise(rng, n, 140)
    rumble /= max(float(np.max(np.abs(rumble))), 1e-6)
    center = 0.37 * t[-1] if n > 1 else 0.0
    decay = np.where(t >= center, np.exp(-(t - center) / 1.1), 0.0)
    attack = np.minimum((t - center) * 18, 1).clip(0, 1)
    return 0.50 * rumble * (attack * decay).astype(np.float32)

def _wind(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t)
    return 0.75 * _smooth_noise(rng, len(t), 60) * gust.astype(np.float32)

def _forest(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    n, sr = len(t), SAMPLE_RATE
    chirps = np.zeros(n, dtype=np.float32)
    for position in np.arange(0.35, t[-1], 0.7):
        start = int(position * sr)
        length = min(int(0.19 * sr), n - start)
        if length <= 0:
            continue
        local_t = np.arange(length, dtype=np.float32) / sr
        freq = rng.uniform(900, 1900)
        phase = 2 * np.pi * (freq * local_t + 1800 * local_t ** 2)
        window = np.sin(np.pi * np.linspace(0, 1, length)) ** 2
        chirps[start:start + length] += (0.12 * np.sin(phase) * window).astype(np.float32)
    return chirps

def _water(rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 0.24 * t - 0.9)) ** 2
    return _smooth_noise(rng, len(t), 12) * envelope.astype(np.float32) * 1.8

def _drone(t: np.ndarray) -> np.ndarray:
    wave = (np.sin(2 * np.pi * 110 * t) + 0.55 * np.sin(2 * np.pi * 164.81 * t)
            + 0.25 * np.sin(2 * np.pi * 220.0 * t))
    lfo = 0.38 + 0.08 * np.sin(2 * np.pi * 0.4 * t)
    return (wave * lfo).astype(np.float32)

def recognized_sound_classes(prompt: str) -> list[str]:
    """Cheap validation that shares the generator's supported vocabulary."""
    lower = prompt.lower()
    recognized = [kind for kind, words in TOKENS.items() if any(word in lower for word in words)]
    if not recognized:
        raise UnsupportedPromptError("No procedural sound matches this prompt; try rain, thunder, forest, wind, ocean, or ambient.")
    return recognized

def generate(prompt: str, duration_seconds: float, seed: int = 0) -> AudioResult:
    recognized = recognized_sound_classes(prompt)
    rng = np.random.default_rng(seed)
    frames = round(SAMPLE_RATE * duration_seconds)
    t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
    layers = {"rain": _rain, "thunder": _thunder, "wind": _wind, "forest": _forest, "water": _water}
    output = np.zeros(frames, dtype=np.float32)
    for kind in recognized:
        output += _drone(t) if kind == "drone" else layers[kind](rng, t)
    edge = min(int(0.025 * SAMPLE_RATE), frames // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= min(1.0, 0.92 / peak)
    return AudioResult(output, SAMPLE_RATE, "procedural-sfx-v0", seed)
