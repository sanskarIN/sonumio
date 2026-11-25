"""Versioned event-aware first-party synthetic crop source.

A generic early crop can miss the thunder attack or all forest chirps.
Use prespecified category-specific windows of a two-second render and
seeded offsets within those windows. Reject silent crops, never add
identity noise just to defeat duplicate detection. Synthetic crops are
not independent real recordings.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch

from research.crop_codec_packet import CropCodecError
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, SAMPLE_RATE, _downsample_24k_to_8k
from sonumio.engines.procedural import generate

PROTOCOL = "sonumio-event-aware-crop-v2"
DURATION = 2.0
# Start indices at 8 kHz. Thunder begins near 5920, and the first
# forest chirp spans roughly 2800..4320. The window must include the
# event, not the preceding silent interval. Inclusive bounds.
WINDOWS = {
    "rain": (400, 12000),
    "thunder": (6500, 8500),
    "wind": (400, 12000),
    "forest": (2400, 3000),
    "ocean": (400, 12000),
    "ambient": (400, 12000),
}
MIN_CENTERED_RMS = 1e-4


def render_crop(category, seed):
    if category not in WINDOWS or type(seed) is not int or not 0 <= seed < 2**32:
        raise CropCodecError("Invalid synthetic crop source coordinates")
    low, high = WINDOWS[category]
    # Separate offset domain from the procedural engine RNG.
    offset_rng = np.random.default_rng(np.random.SeedSequence([seed, CATEGORIES.index(category), 0x534C02]))
    offset = int(offset_rng.integers(low, high+1))
    source = generate(category, DURATION, seed)
    wave = _downsample_24k_to_8k(source.samples)
    if wave.shape != (int(DURATION*SAMPLE_RATE),):
        raise CropCodecError("Invalid synthetic source geometry")
    crop = np.ascontiguousarray(wave[offset:offset+CLIP_FRAMES], dtype=np.float32)
    if crop.shape != (CLIP_FRAMES,) or not np.isfinite(crop).all():
        raise CropCodecError("Invalid event-aware waveform crop")
    centered = crop.astype(np.float64)-float(crop.mean(dtype=np.float64))
    if float(np.sqrt(np.mean(centered**2))) < MIN_CENTERED_RMS:
        raise CropCodecError("Synthetic crop missed active source: "+category)
    return crop, offset


def dataset(seeds: tuple[int, ...]):
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 8
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise CropCodecError("Invalid synthetic crop source seeds")
    clips, labels = [], []
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0")
    for label, category in enumerate(CATEGORIES):
        for seed in seeds:
            crop, offset = render_crop(category, seed)
            trace.update(struct.pack("<III", label, seed, offset))
            trace.update(np.asarray(crop, dtype="<f4").tobytes())
            clips.append(torch.from_numpy(crop.copy()).view(1, -1))
            labels.append(label)
    return torch.stack(clips), torch.tensor(labels, dtype=torch.long), trace.hexdigest()
