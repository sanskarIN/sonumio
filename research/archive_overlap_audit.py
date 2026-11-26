"""Exhaustive bounded cross-split waveform overlap screening for private archives.

Unlike the ingestion sign-sketch gates, this diagnostic evaluates every pair
and every allowed integer lag. It is a numerical screening tool, not a
proof of dataset independence, rights, or semantic equivalence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = "sonumio-exhaustive-archive-overlap-v1"
SPLITS = ("train", "validation", "test")
DEFAULT_PAIR_BUDGET = 4096
MAX_PAIR_BUDGET = 100_000
DEFAULT_THRESHOLD = 0.9995
MIN_OVERLAP_ENERGY = 1e-8
SCREENING_MARGIN = 1e-7


@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    max_shift: int = 1024
    threshold: float = DEFAULT_THRESHOLD
    pair_budget: int = DEFAULT_PAIR_BUDGET

    def validate(self, frames: int) -> None:
        if type(frames) is not int or not 8 <= frames <= 16384:
            raise ValueError("Unsupported waveform frame count")
        if type(self.max_shift) is not int or not 0 <= self.max_shift <= frames // 2:
            raise ValueError("max_shift must be an integer between zero and half the clip")
        if (type(self.threshold) not in (int, float)
                or not math.isfinite(self.threshold)
                or not 0.99 <= self.threshold < 1):
            raise ValueError("threshold must be finite and in [0.99, 1)")
        if (type(self.pair_budget) is not int
                or not 1 <= self.pair_budget <= MAX_PAIR_BUDGET):
            raise ValueError("pair_budget must be an integer in [1, 100000]")


@dataclass(frozen=True, slots=True)
class _Prepared:
    spectrum: np.ndarray
    sums: np.ndarray
    squares: np.ndarray
    constant: bool


@dataclass(frozen=True, slots=True)
class PairResult:
    max_absolute_correlation: float | None
    defined_lags: int
    screened_match: bool


class CorrelationPlan:
    """Compute overlap-specific centered correlations with a linear FFT.

    A positive lag compares the start of the first clip with the later
    portion of the second. Prefix sums supply the mean and energy of each
    actual overlapping segment; padded samples never enter the denominator.
    """

    def __init__(self, frames: int, policy: OverlapPolicy):
        policy.validate(frames)
        self.frames = frames
        self.policy = policy
        self.fft_size = 1 << (2 * frames - 2).bit_length()
        self.lags = np.arange(-policy.max_shift, policy.max_shift + 1,
                              dtype=np.int64)
        self.a_start = np.maximum(-self.lags, 0)
        self.b_start = np.maximum(self.lags, 0)
        self.lengths = frames - np.abs(self.lags)
        self.a_end = self.a_start + self.lengths
        self.b_end = self.b_start + self.lengths
        self.fft_indices = self.lags % self.fft_size

    def prepare(self, waveform: np.ndarray) -> _Prepared:
        if (not isinstance(waveform, np.ndarray)
                or waveform.shape != (self.frames,)
                or waveform.dtype != np.dtype("float32")
                or not np.isfinite(waveform).all()):
            raise ValueError("Expected one finite float32 waveform")
        # Normalize globally in float64. Subsequent local centering makes
        # the actual lag correlations independent of gain, offset and sign.
        x = waveform.astype(np.float64)
        x -= np.mean(x)
        norm = float(np.linalg.norm(x))
        constant = norm == 0.0
        if not constant:
            x /= norm
        sums = np.concatenate(([0.0], np.cumsum(x)))
        squares = np.concatenate(([0.0], np.cumsum(x * x)))
        return _Prepared(np.fft.rfft(x, n=self.fft_size), sums, squares, constant)

    def correlations(self, a: _Prepared, b: _Prepared) -> np.ndarray:
        """Return one correlation per lag; NaN means numerically undefined."""
        result = np.full(len(self.lags), np.nan, dtype=np.float64)
        if a.constant or b.constant:
            return result
        cross = np.fft.irfft(np.conj(a.spectrum) * b.spectrum,
                             n=self.fft_size)[self.fft_indices]
        sa = a.sums[self.a_end] - a.sums[self.a_start]
        sb = b.sums[self.b_end] - b.sums[self.b_start]
        qa = a.squares[self.a_end] - a.squares[self.a_start]
        qb = b.squares[self.b_end] - b.squares[self.b_start]
        va = qa - sa * sa / self.lengths
        vb = qb - sb * sb / self.lengths
        valid = (va >= MIN_OVERLAP_ENERGY) & (vb >= MIN_OVERLAP_ENERGY)
        numerators = cross[valid] - sa[valid] * sb[valid] / self.lengths[valid]
        correlations = numerators / np.sqrt(va[valid] * vb[valid])
        if not np.isfinite(correlations).all():
            raise ValueError("Nonfinite numerical overlap result")
        result[valid] = np.clip(correlations, -1.0, 1.0)
        return result

    def compare(self, a: _Prepared, b: _Prepared) -> PairResult:
        values = self.correlations(a, b)
        defined = np.isfinite(values)
        count = int(np.count_nonzero(defined))
        maximum = float(np.max(np.abs(values[defined]))) if count else None
        # A small conservative margin makes a borderline result a review
        # candidate rather than a clearance. It is not a rigorous FFT bound.
        return PairResult(maximum, count, maximum is not None and
                          maximum >= self.policy.threshold - SCREENING_MARGIN)


def _pair_plan(counts: dict[str, int], policy: OverlapPolicy) -> list[tuple[str, str, int]]:
    if (set(counts) != set(SPLITS)
            or any(type(v) is not int or v < 0 for v in counts.values())):
        raise ValueError("Invalid archive split counts")
    plan = [(a, b, counts[a] * counts[b]) for a, b in combinations(SPLITS, 2)]
    if sum(n for _, _, n in plan) > policy.pair_budget:
        raise ValueError("Exhaustive pair budget exceeded; no partial audit performed")
    return plan


def _prepare_shard(shard, plan: CorrelationPlan) -> tuple[_Prepared, ...]:
    return tuple(plan.prepare(clip[0]) for clip in shard.clips)


def audit_archive_overlap(reader, policy: OverlapPolicy = OverlapPolicy()) -> dict:
    """Return aggregate diagnostics after reading a digest-verified archive.

    No crop IDs, captions, paths, individual scores, waveform or weight
    artifacts leave this function. At most two prepared shards are retained;
    later split shards are reread for each earlier split shard.
    """
    from research.corpus_archive import CorpusArchiveReader

    if not isinstance(reader, CorpusArchiveReader):
        raise ValueError("Expected a verified private archive reader")
    metadata = reader.report()
    plan = CorrelationPlan(metadata["clip_frames"], policy)
    comparisons = _pair_plan(metadata["split_counts"], policy)
    summaries = {}
    total = matches = undefined = 0
    for first, second, expected in comparisons:
        pairs = candidates = defined = 0
        highest = None
        if expected:
            for left_shard in reader.iter_shards(first):
                left = _prepare_shard(left_shard, plan)
                for right_shard in reader.iter_shards(second):
                    right = _prepare_shard(right_shard, plan)
                    for a in left:
                        for b in right:
                            outcome = plan.compare(a, b)
                            pairs += 1
                            candidates += int(outcome.screened_match)
                            defined += outcome.defined_lags
                            value = outcome.max_absolute_correlation
                            if value is not None:
                                highest = value if highest is None else max(highest, value)
                    del right
                del left
        if pairs != expected:
            raise ValueError("Archive pair coverage differs from declared plan")
        possible = pairs * len(plan.lags)
        missing = possible - defined
        summaries[first + "_vs_" + second] = {
            "expected_pairs": expected, "examined_pairs": pairs,
            "potential_overlap_pairs": candidates,
            "defined_lags": defined, "undefined_lags": missing,
            "max_absolute_correlation": highest,
        }
        total += pairs
        matches += candidates
        undefined += missing
    if reader.report()["archive_sha256"] != metadata["archive_sha256"]:
        raise ValueError("Archive identity changed during overlap audit")
    # A completely empty comparison cannot count as clearance.
    clear = total > 0 and matches == 0 and undefined == 0
    return {
        "protocol": PROTOCOL,
        "archive_sha256": metadata["archive_sha256"],
        "sample_rate": metadata["sample_rate"],
        "clip_frames": plan.frames,
        "policy": {
            "max_shift_frames": policy.max_shift, "threshold": policy.threshold,
            "screening_margin": SCREENING_MARGIN, "pair_budget": policy.pair_budget,
            "min_relative_centered_overlap_energy": MIN_OVERLAP_ENERGY,
        },
        "split_pairs": summaries,
        "examined_pairs": total, "potential_overlap_pairs": matches,
        "undefined_lags": undefined,
        "all_declared_cross_split_pairs_examined": True,
        "screening_clear": clear,
        "rights_or_dataset_independence_established": False,
        "audio_or_weights_saved": False,
    }


def main(argv: list[str] | None = None) -> int:
    from research.corpus_archive import CorpusArchiveReader
    from research.corpus_archive_format import require_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-shift", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--pair-budget", type=int, default=DEFAULT_PAIR_BUDGET)
    parser.add_argument("--require-clear", action="store_true")
    args = parser.parse_args(argv)
    try:
        digest = require_sha256(args.expected_sha256)
        policy = OverlapPolicy(args.max_shift, args.threshold, args.pair_budget)
        with CorpusArchiveReader(args.archive, expected_sha256=digest) as reader:
            report = audit_archive_overlap(reader, policy)
    except (ValueError, OSError, RuntimeError):
        # The archive reader can include sensitive local context in errors.
        # Never emit raw exception strings or a partial diagnostic report.
        parser.exit(2, "Archive overlap audit failed verification, policy or pair budget.\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 2 if args.require_clear and not report["screening_clear"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
