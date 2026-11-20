"""Packet-boundary-invariant local triangular PCM dithering (TPDF).

This module deliberately defines a NEW stream-specific random draw order,
rather than silently claiming compatibility with the legacy one-shot
`quantize_pcm(..., dither=True)` routine. It is not a perceptual noise
shaper, watermark, cryptographic randomness, mastering effect or model.
"""
from __future__ import annotations

import numpy as np

from .contracts import as_channel_first, check_audio
from .dither import QuantizationSettings, pack_pcm
from .errors import AudioContractError, AudioLimitError

# Bound RNG, float64 and packed-PCM working allocations even for independent
# callers that do not use the 65,536-frame WavPacketWriter contract.
MAX_DITHER_SAMPLES_PER_CALL = 65_536 * 8
STREAM_TPDF_VERSION = "pcg64-frame-major-interleaved-pairs-v1"


class StreamDitherQuantizer:
    """Consume exactly two PCG64 uniforms per interleaved PCM sample.

    PCM traversal is frame-major: frame 0 channel 0, frame 0 channel 1, ...
    and then frame 1. For each sample, the first random uniform is subtracted
    by the second. One persistent generator is owned by this quantizer.
    Consecutive packet partitions are bit-for-bit identical to single-call
    processing under the same seed, settings, NumPy/PCG64 algorithm and PCM.

    This class is synchronous, single-owner state; never share across output
    streams or threads, or reuse after an export has been aborted.
    """

    def __init__(self, settings: QuantizationSettings):
        if not isinstance(settings, QuantizationSettings) or not settings.dither:
            raise AudioContractError(
                "StreamDitherQuantizer requires enabled, explicitly seeded TPDF"
            )
        # Explicit PCG64 bit generator, not default_rng whose default family
        # can change across versions. The uniform conversion still depends on
        # NumPy's documented implementation; CI checks supported versions.
        self.settings = settings
        self._rng = np.random.Generator(np.random.PCG64(settings.seed))
        self.samples_processed = 0

    def quantize(self, samples: np.ndarray) -> np.ndarray:
        """Return channel-first signed int64 PCM for a single bounded batch."""
        # All validation happens before consuming RNG state.
        data = as_channel_first(check_audio(samples), float64=True)
        sample_count = int(data.size)
        if sample_count > MAX_DITHER_SAMPLES_PER_CALL:
            raise AudioLimitError("Streaming TPDF batch exceeds sample cap")
        # Use double-precision calculations, matching other PCM quantizers.
        scale = 1 << (self.settings.bits - 1)
        gain = 10.0 ** (self.settings.headroom_db / 20.0)
        interleaved = data.T.reshape(-1)
        scaled = np.clip(interleaved * gain, -1.0, 1.0) * scale
        # Pair adjacent uniforms per individual sample. Drawing a single
        # 2*N vector (instead of all N positive then N negative values)
        # makes the draw order independent of packet size.
        pairs = self._rng.random((sample_count, 2))
        scaled += pairs[:, 0] - pairs[:, 1]
        rounded = np.clip(np.rint(scaled), -scale, scale - 1)
        ints = rounded.astype(np.int64).reshape(data.shape[1], data.shape[0]).T
        self.samples_processed += sample_count
        return ints

    def encode(self, samples: np.ndarray) -> bytes:
        """Pack one dithered batch into interleaved integer PCM bytes."""
        return pack_pcm(self.quantize(samples), bits=self.settings.bits)
