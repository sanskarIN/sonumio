"""Bounded aligned-waveform overlap gate for rights-reviewed audio splits.

A deterministic sign-sketch retrieves plausible candidates; all possible matches
are verified using the entire centered, unit-normalized waveform. The gate can
detect many gain, DC-offset or polarity edits of an *aligned* model input.
It cannot prove dataset independence or detect arbitrary time shifts, edits,
resampling, remixing, or semantically similar recordings.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from research.dataset_manifest import ManifestError

BANDS = 4
BITS_PER_BAND = 16
PROBE_COUNT = BANDS * BITS_PER_BAND
MAX_CANDIDATES_PER_CLIP = 512
CORRELATION_THRESHOLD = 0.9995


@dataclass(frozen=True)
class _IndexedClip:
    samples: np.ndarray
    split: str
    line: int
    piece: int
    mean: float
    norm: float


class AlignedLeakageGuard:
    """Fail-closed bounded candidate check for cross-split aligned copies.

    Holds references to source float32 clips, not a duplicated waveform corpus.
    Callers must not mutate submitted input arrays during this audit.
    """

    def __init__(self, *, frames: int, threshold: float = CORRELATION_THRESHOLD,
                 max_candidates: int = MAX_CANDIDATES_PER_CLIP):
        if type(frames) is not int or frames < PROBE_COUNT:
            raise ValueError("frames must be an integer >= 64")
        if type(threshold) is not float or not 0.99 <= threshold < 1.0:
            raise ValueError("threshold must be a float in [0.99, 1.0)")
        if type(max_candidates) is not int or not 1 <= max_candidates <= 4096:
            raise ValueError("max_candidates must be an integer in [1, 4096]")
        self.frames = frames
        self.threshold = threshold
        self.max_candidates = max_candidates
        self._positions = np.asarray(
            [(37 + 977 * i) % frames for i in range(PROBE_COUNT)],
            dtype=np.intp,
        )
        if len(set(self._positions.tolist())) != PROBE_COUNT:
            raise ValueError("probe pattern requires distinct positions")
        self._entries: list[_IndexedClip] = []
        self._bands: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.compared_pairs = 0

    def _sketch(self, samples: np.ndarray, mean: float) -> tuple[int, ...]:
        signs = samples[self._positions].astype(np.float64) > mean
        return tuple(
            sum(int(bit) << j for j, bit in enumerate(
                signs[k * BITS_PER_BAND:(k + 1) * BITS_PER_BAND]))
            for k in range(BANDS)
        )

    def add(self, samples: np.ndarray, *, split: Literal["train", "validation", "test"],
            line: int, piece: int) -> None:
        if (not isinstance(samples, np.ndarray)
                or samples.shape != (self.frames,)
                or samples.dtype != np.dtype("float32")
                or not np.isfinite(samples).all()):
            raise ManifestError("aligned overlap audit requires finite float32 model-input clips")
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split for aligned overlap audit")
        mean = float(np.mean(samples, dtype=np.float64))
        centered = samples.astype(np.float64) - mean
        norm = float(np.linalg.norm(centered))
        if norm <= 1e-10:
            raise ManifestError("nearly constant model-input clip cannot pass overlap audit")
        keys = self._sketch(samples, mean)
        candidates: set[int] = set()
        mask = (1 << BITS_PER_BAND) - 1
        for band, bits in enumerate(keys):
            for code in (bits, bits ^ mask):
                for idx in self._bands.get((band, code), ()):
                    if self._entries[idx].split != split:
                        candidates.add(idx)
                        if len(candidates) > self.max_candidates:
                            raise ManifestError(
                                "aligned overlap audit exceeded bounded candidate budget; "
                                "split source recordings into smaller reviewed batches"
                            )
        for idx in sorted(candidates):
            previous = self._entries[idx]
            other = previous.samples.astype(np.float64) - previous.mean
            correlation = float(np.dot(centered, other) / (norm * previous.norm))
            self.compared_pairs += 1
            if abs(correlation) >= self.threshold:
                raise ManifestError(
                    "aligned gain/DC/polarity-invariant model-input overlap "
                    f"with line {previous.line} ({previous.split} split, "
                    f"clip {previous.piece}); full-clip |correlation| "
                    f"{abs(correlation):.6f} >= {self.threshold:.6f}"
                )
        idx = len(self._entries)
        self._entries.append(_IndexedClip(samples, split, line, piece, mean, norm))
        for band, code in enumerate(keys):
            self._bands[(band, code)].append(idx)

    @property
    def indexed_clips(self) -> int:
        return len(self._entries)
