"""Experimental limited-vocabulary *multi-sound* conditioning for waveform DDPM.

Not a language model or open-vocabulary text-to-audio. It recognizes one or two
specified sound tags (English/Hindi) and trains embeddings on DSP-made mixtures.
This is a research-only step towards compositional conditioning, NOT Sonumio-1.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import math
import re

import numpy as np
import torch

from research.wave_diffusion import (
    CATEGORIES, CLIP_FRAMES, SAMPLE_RATE, DiffusionSchedule,
    TinyWaveDenoiser, _downsample_24k_to_8k,
)
from sonumio.engines.procedural import TOKENS, generate

WORD_KEYS = {"ocean": "water", "ambient": "drone"}
# Voice and speech requests are not eligible for the sound-effects-only model.
EXCLUDED = ("voice", "speech", "clone", "impersonat", "narrat", "आवाज़", "आवाज", "बोलने", "नैरेशन")


def parse_sound_prompt(text: str) -> torch.Tensor:
    """Return [6] binary condition vector; reject unsupported / risky prompts."""
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 800:
        raise ValueError("prompt must be non-empty text of at most 800 characters")
    lower = text.casefold()
    if any(word in lower for word in EXCLUDED):
        raise ValueError("speech/voice requests are excluded from this sound-effects model")
    labels = []
    for idx, category in enumerate(CATEGORIES):
        words = TOKENS[WORD_KEYS.get(category, category)]
        if any((re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", lower)
                if word.isascii() else word in lower) for word in words):
            labels.append(idx)
    if not 1 <= len(labels) <= 2:
        raise ValueError("prompt needs one or two known sound categories: " + ", ".join(CATEGORIES))
    result = torch.zeros(len(CATEGORIES), dtype=torch.float32)
    result[labels] = 1
    return result


def mixed_dataset(seeds: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Single sounds + all 15 two-sound compositions from DSP, deterministic."""
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be unique 32-bit nonnegative integers")
    combos = [(x,) for x in range(len(CATEGORIES))] + list(combinations(range(len(CATEGORIES)), 2))
    examples, labels = [], []
    digest = sha256()
    for ids in combos:
        prompt = " and ".join(CATEGORIES[index] for index in ids)
        condition = parse_sound_prompt(prompt)
        for seed in seeds:
            audio = generate(prompt, 0.5, seed).samples
            wave = _downsample_24k_to_8k(audio)[:CLIP_FRAMES].copy()
            examples.append(torch.from_numpy(wave).view(1, -1))
            labels.append(condition)
            digest.update(prompt.encode("utf-8") + seed.to_bytes(4, "little") + wave.tobytes())
    return torch.stack(examples), torch.stack(labels), digest.hexdigest()


@torch.no_grad()
def sample_mixed(schedule: DiffusionSchedule, model: TinyWaveDenoiser,
                 prompt: str, *, seed: int = 0, guidance: float = 1.0,
                 frames: int = CLIP_FRAMES) -> torch.Tensor:
    """DDPM generation with multi-label embeddings and bounded CFG."""
    vector = parse_sound_prompt(prompt)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError("guidance must be finite in [0, 4]")
    if not 256 <= frames <= 32_768 or frames % 16:
        raise ValueError("frames must be a multiple of 16, 256..32768")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, 1, frames), device=device, generator=generator)
    condition = vector.to(device)[None, :]
    empty = torch.zeros_like(condition)
    model.eval()
    for step in range(schedule.steps - 1, -1, -1):
        t = torch.tensor([step], device=device)
        prediction = model(x, t, condition)
        if guidance != 1:
            unconditional = model(x, t, empty)
            prediction = unconditional + guidance * (prediction - unconditional)
        alpha, bar, beta = schedule.alphas[step], schedule.alpha_bars[step], schedule.betas[step]
        mean = (x - beta / torch.sqrt(1 - bar) * prediction) / alpha.sqrt()
        if step:
            prev = schedule.alpha_bars[step - 1]
            variance = beta * (1 - prev) / (1 - bar)
            x = mean + variance.sqrt() * torch.randn(x.shape, device=device, generator=generator)
        else:
            x = mean
    return x.clamp(-1, 1).cpu()[0, 0]
