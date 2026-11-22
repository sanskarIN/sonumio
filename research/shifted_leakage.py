"""Narrow, bounded time-shifted waveform leakage check for reviewed model clips.

This is a second gate, complementing exact hashes and aligned correlation. It
indexes short, high-variation derivative-sign fingerprints from earlier clips
and probes every possible frame offset of the new clip. A fingerprint hit is
NEVER enough to reject audio: the corresponding full overlapping waveforms
must have a high centered absolute correlation.

Scope: integer offsets of at most 256 frames at the fixed 8 kHz model-input
rate, and approximately gain/DC/polarity-preserving copies whose sign anchors
remain recognizable. The detector intentionally ignores low-complexity
anchors and cannot certify absence of leakage, arbitrary edits, time warps,
resampling differences, remixing, or semantically related recordings.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from research.dataset_manifest import ManifestError

_SPLITS = ("train", "validation", "test")
_SIGN_BITS = 64
_ANCHOR_STRIDE = 128
_MIN_TURNS = 8
_TRIPLES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_MASK48 = (1 << 48) - 1
DEFAULT_MAX_SHIFT = 256
DEFAULT_THRESHOLD = 0.9995
DEFAULT_MAX_CANDIDATES = 256
DEFAULT_MAX_BUCKET_ITEMS = 4096


@dataclass(frozen=True)
class _Clip:
    samples: np.ndarray
    split: str
    line: int
    piece: int


def _sign_fingerprints(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four polarity-invariant 48-bit keys at every 65-frame window.

    Three-of-four 16-bit groups give limited resilience when a noisy edit
    changes derivative signs in *one* group. Explicit little-endian packing
    avoids host-byte-order-dependent index keys. A matching key alone has no
    statistical or legal meaning; a full-window correlation must also pass.
    """
    rising = np.diff(samples.astype(np.float64, copy=False)) > 0.0
    windows = np.lib.stride_tricks.sliding_window_view(rising, _SIGN_BITS)
    turns = np.count_nonzero(windows[:, 1:] != windows[:, :-1], axis=1)
    active = turns >= _MIN_TURNS
    packed = np.packbits(windows, axis=1, bitorder="little")
    words = packed.view("<u2").reshape(-1, 4).astype(np.uint64, copy=False)
    keys = np.empty((len(words), len(_TRIPLES)), dtype=np.uint64)
    for column, (a, b, c) in enumerate(_TRIPLES):
        value = (
            (words[:, a] << np.uint64(32))
            | (words[:, b] << np.uint64(16))
            | words[:, c]
        )
        keys[:, column] = np.minimum(value, value ^ np.uint64(_MASK48))
    return keys, active


def _overlap_correlation(previous: np.ndarray, current: np.ndarray, lag: int) -> float:
    """Compare only *real* overlapping frames, never a zero-padded edge."""
    length = len(previous)
    if lag >= 0:
        a = previous[:length - lag]
        b = current[lag:]
    else:
        a = previous[-lag:]
        b = current[:length + lag]
    a = a.astype(np.float64) - float(np.mean(a, dtype=np.float64))
    b = b.astype(np.float64) - float(np.mean(b, dtype=np.float64))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-10:
        return 0.0
    return float(np.dot(a, b) / norm)


class ShiftedLeakageGuard:
    """Fail-closed cross-split audit for near-aligned time-shifted copies.

    The previous input arrays are referenced, not copied. Callers must not
    mutate input clips while using this audit. Each new recording has bounded
    candidate comparisons; individual posting lists have an explicit limit.
    The index is partitioned by split to avoid train-only quadratic scans.
    """

    def __init__(
        self,
        *,
        frames: int,
        max_shift_frames: int = DEFAULT_MAX_SHIFT,
        threshold: float = DEFAULT_THRESHOLD,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_bucket_items: int = DEFAULT_MAX_BUCKET_ITEMS,
    ) -> None:
        if type(frames) is not int or frames < 1024:
            raise ValueError("frames must be an integer >= 1024")
        if (type(max_shift_frames) is not int
                or not 1 <= max_shift_frames <= frames // 4):
            raise ValueError("max_shift_frames must be an integer in [1, frames//4]")
        if type(threshold) is not float or not 0.99 <= threshold < 1.0:
            raise ValueError("threshold must be a float in [0.99, 1.0)")
        if type(max_candidates) is not int or not 1 <= max_candidates <= 4096:
            raise ValueError("max_candidates must be an integer in [1, 4096]")
        if type(max_bucket_items) is not int or not 1 <= max_bucket_items <= 65536:
            raise ValueError("max_bucket_items must be an integer in [1, 65536]")
        self.frames = frames
        self.max_shift_frames = max_shift_frames
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.max_bucket_items = max_bucket_items
        self._entries: list[_Clip] = []
        self._buckets: dict[tuple[str, int, int], list[tuple[int, int]]] = defaultdict(list)
        self._split_counts: Counter[str] = Counter()
        self.compared_pairs = 0
        self.indexed_anchors = 0

    @property
    def indexed_clips(self) -> int:
        return len(self._entries)

    def add(
        self,
        samples: np.ndarray,
        *,
        split: Literal["train", "validation", "test"],
        line: int,
        piece: int,
    ) -> None:
        if (not isinstance(samples, np.ndarray)
                or samples.shape != (self.frames,)
                or samples.dtype != np.dtype("float32")
                or not np.isfinite(samples).all()):
            raise ManifestError(
                "shifted overlap audit requires finite float32 model-input clips"
            )
        if split not in _SPLITS:
            raise ValueError("invalid split for shifted overlap audit")

        keys, active = _sign_fingerprints(samples)
        if any(self._split_counts[other] for other in _SPLITS if other != split):
            considered: set[tuple[int, int]] = set()
            for q in np.flatnonzero(active):
                q = int(q)
                for column in range(len(_TRIPLES)):
                    code = int(keys[q, column])
                    for other in _SPLITS:
                        if other == split:
                            continue
                        for idx, p in self._buckets.get((other, column, code), ()):
                            lag = q - p
                            if abs(lag) > self.max_shift_frames:
                                continue
                            candidate = (idx, lag)
                            if candidate in considered:
                                continue
                            considered.add(candidate)
                            if len(considered) > self.max_candidates:
                                raise ManifestError(
                                    "shifted overlap audit exceeded bounded candidate budget; "
                                    "split recordings into smaller manually reviewed batches"
                                )
                            previous = self._entries[idx]
                            correlation = _overlap_correlation(
                                previous.samples, samples, lag
                            )
                            self.compared_pairs += 1
                            if abs(correlation) >= self.threshold:
                                raise ManifestError(
                                    "shifted gain/DC/polarity-invariant model-input overlap "
                                    f"with line {previous.line} ({previous.split} split, "
                                    f"clip {previous.piece}); offset {lag} frames, "
                                    f"overlap {self.frames - abs(lag)} frames, "
                                    f"|correlation| {abs(correlation):.6f} "
                                    f">= {self.threshold:.6f}"
                                )

        # Check every impending posting limit before committing any keys, so
        # failed records cannot partially poison the in-memory audit index.
        postings = [
            (split, column, int(keys[p, column]), p)
            for p in range(0, len(active), _ANCHOR_STRIDE)
            if active[p]
            for column in range(len(_TRIPLES))
        ]
        additions = Counter((s, column, code) for s, column, code, _ in postings)
        for key, increment in additions.items():
            if len(self._buckets.get(key, ())) + increment > self.max_bucket_items:
                raise ManifestError(
                    "shifted overlap audit anchor bucket exceeds bounded posting budget; "
                    "reduce repeated source content or review smaller batches"
                )

        idx = len(self._entries)
        self._entries.append(_Clip(samples, split, line, piece))
        for s, column, code, p in postings:
            self._buckets[(s, column, code)].append((idx, p))
        self.indexed_anchors += len(postings) // len(_TRIPLES)
        self._split_counts[split] += 1
