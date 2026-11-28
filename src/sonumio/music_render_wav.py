"""Independent canonical stereo PCM16 verification for rendered artifacts."""
from __future__ import annotations

import struct
import hashlib
import numpy as np

from .audio import SAMPLE_RATE
from .errors import GenerationError

MAX_FRAMES = 120 * SAMPLE_RATE


def inspect_pcm16(data, frames):
    if (type(frames) is not int or not 1 <= frames <= MAX_FRAMES
            or type(data) is not bytes or len(data) != 44 + 4 * frames):
        raise GenerationError("Invalid canonical render WAV length")
    if (data[:4] != b"RIFF" or data[8:16] != b"WAVEfmt " or data[36:40] != b"data"
            or struct.unpack_from("<I", data, 4)[0] != len(data)-8
            or struct.unpack_from("<IHHIIHH", data, 16) !=
               (16, 1, 2, SAMPLE_RATE, SAMPLE_RATE*4, 4, 16)
            or struct.unpack_from("<I", data, 40)[0] != frames*4):
        raise GenerationError("Invalid canonical render WAV header")
    samples = np.frombuffer(data, dtype="<i2", offset=44).reshape(-1, 2).astype(np.int64)
    return {
        "frames": frames, "channels": 2, "sample_rate": SAMPLE_RATE, "bits_per_sample": 16,
        "pcm_sha256": hashlib.sha256(data[44:]).hexdigest(),
        "minimum": [int(v) for v in samples.min(axis=0)],
        "maximum": [int(v) for v in samples.max(axis=0)],
        "sum_squares": [int(v) for v in (samples*samples).sum(axis=0)],
        "zero_samples": [int(v) for v in (samples==0).sum(axis=0)],
        "clipped_samples": [int(v) for v in ((samples==-32768)|(samples==32767)).sum(axis=0)],
    }
