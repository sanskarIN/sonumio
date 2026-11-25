"""Versioned first-party crop source with explicit seed-dependent offsets.

The legacy short waveform dataset uses the first samples for every seed.
Some procedural sources are seed-invariant, so distinct seeds can yield
identical PCM. This separate recipe uses the longer source and seeded
window already established by latent_audio. Exact duplicates still fail
closed; different offsets do not imply independent real recordings.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np
import torch

from research.crop_codec_packet import CropCodecError
from research.latent_audio import render_seeded_waveform
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES

PROTOCOL = "sonumio-seeded-window-crop-v1"


def dataset(seeds: tuple[int, ...]):
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= 8
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise CropCodecError("Invalid synthetic crop source seeds")
    clips, labels = [], []
    trace = hashlib.sha256(PROTOCOL.encode("ascii") + b"\0")
    for label in range(len(CATEGORIES)):
        for seed in seeds:
            crop = np.ascontiguousarray(
                render_seeded_waveform((label,), seed)[:CLIP_FRAMES], dtype=np.float32
            )
            if crop.shape != (CLIP_FRAMES,) or not np.isfinite(crop).all():
                raise CropCodecError("Invalid seeded waveform crop")
            trace.update(struct.pack("<II", label, seed))
            trace.update(np.asarray(crop, dtype="<f4").tobytes())
            clips.append(torch.from_numpy(crop.copy()).view(1, -1))
            labels.append(label)
    return torch.stack(clips), torch.tensor(labels, dtype=torch.long), trace.hexdigest()
