"""Deterministic bounded endpoint selection for original PCM audio loops.

The score is a signal-level seam proxy, not a claim of inaudibility.
Searches explicitly stay around caller-selected source positions, keep
multichannel observations linked, and never assume that a high waveform
correlation proves a musical phrase is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError

MAX_GRID_PAIRS = 50_000
MAX_PAIR_WORK = 90_000_000
MAX_REFINED_PAIRS = 16_384
MAX_RADIUS = 8192
MAX_WINDOWS_BYTES = 24 * 1024 * 1024
_EPS = 1e-12


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AudioContractError(f"{name} must be integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class LoopScoreWeights:
    sample_jump: float = 1.
    slope_jump: float = .5
    rms_mismatch: float = .5
    shape_mismatch: float = 1.
    spectrum_mismatch: float = .5
    silence_penalty: float = .25

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite_float(getattr(self, name), name, minimum=0, maximum=10)
        if sum(getattr(self, name) for name in self.__dataclass_fields__) <= 0:
            raise AudioContractError("At least one loop score weight must be positive")


@dataclass(frozen=True, slots=True)
class LoopSearchPlan:
    start_frame: int
    stop_frame: int
    start_radius: int = 128
    stop_radius: int = 128
    context_frames: int = 256
    stride: int = 8
    refine_count: int = 4
    results: int = 8
    minimum_period: int = 2
    maximum_period: int | None = None
    blend_frames: int = 0
    weights: LoopScoreWeights = LoopScoreWeights()

    def __post_init__(self):
        integer(self.start_frame, "start_frame", 0, 2**31-1)
        integer(self.stop_frame, "stop_frame", 1, 2**31-1)
        if self.stop_frame <= self.start_frame:
            raise AudioContractError("Loop endpoints are reversed or empty")
        integer(self.start_radius, "start_radius", 0, MAX_RADIUS)
        integer(self.stop_radius, "stop_radius", 0, MAX_RADIUS)
        integer(self.context_frames, "context_frames", 8, 4096)
        integer(self.stride, "stride", 1, 64)
        integer(self.refine_count, "refine_count", 0, 16)
        integer(self.results, "results", 1, 64)
        integer(self.minimum_period, "minimum_period", 2, 2**31-1)
        if self.maximum_period is not None:
            integer(self.maximum_period, "maximum_period",
                    self.minimum_period, 2**31-1)
        integer(self.blend_frames, "blend_frames", 0, 4096)
        if self.blend_frames == 1:
            raise AudioContractError("Crossfade must be zero or at least two frames")
        if not isinstance(self.weights, LoopScoreWeights):
            raise AudioContractError("Expected LoopScoreWeights")

    @property
    def minimum_source_span(self):
        return max(self.context_frames, 2*self.blend_frames, self.minimum_period+self.blend_frames)

    def valid_pair(self, start, stop):
        source_span = stop - start
        period = source_span - self.blend_frames
        return (source_span >= self.minimum_source_span
                and (self.maximum_period is None or period <= self.maximum_period))


@dataclass(frozen=True, slots=True)
class SeamMetrics:
    sample_jump: float
    raw_boundary_step: float
    slope_jump: float
    rms_mismatch: float
    shape_correlation: float
    spectrum_mismatch: float
    head_rms: float
    tail_rms: float
    silence_penalty: float

    def score(self, weights):
        if not isinstance(weights, LoopScoreWeights):
            raise AudioContractError("Expected LoopScoreWeights")
        return float(
            self.sample_jump*weights.sample_jump
            + self.slope_jump*weights.slope_jump
            + self.rms_mismatch*weights.rms_mismatch
            + (1-self.shape_correlation)*weights.shape_mismatch
            + self.spectrum_mismatch*weights.spectrum_mismatch
            + self.silence_penalty*weights.silence_penalty
        )


@dataclass(frozen=True, slots=True)
class LoopCandidate:
    start_frame: int
    stop_frame: int
    period_frames: int
    offset_distance: int
    score: float
    metrics: SeamMetrics

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoopSearchResult:
    sample_rate: int
    channels: int
    source_frames: int
    plan: LoopSearchPlan
    coarse_pairs: int
    refined_pairs: int
    candidates: tuple[LoopCandidate, ...]
    protocol: str = "sonumio-loop-seam-search-v1"

    @property
    def best(self):
        return self.candidates[0]

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source_frames": self.source_frames,
            "plan": asdict(self.plan),
            "coarse_pairs": self.coarse_pairs,
            "refined_pairs": self.refined_pairs,
            "candidates": [x.as_dict() for x in self.candidates],
            "perceptual_seamlessness_established": False,
            "model_inference": False,
        }


@dataclass(slots=True)
class _Endpoint:
    values: np.ndarray
    centered: np.ndarray
    energy: float
    rms: float
    distribution: np.ndarray


def _spectral_distribution(values):
    """Small normalized spectral shape vector; never a timbre classifier."""
    n = values.shape[1]
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(values*window, axis=1))**2
    # Preserve channel identity rather than summing phase-cancelling waveforms.
    edges = np.unique(np.linspace(0, spectrum.shape[1], 9, dtype=np.int64))
    bands = np.stack([spectrum[:, a:b].sum(axis=1)
                      for a, b in zip(edges[:-1], edges[1:])], axis=1)
    total = float(bands.sum())
    return bands/total if total > _EPS else np.zeros_like(bands)


def _endpoint(values):
    values = np.array(values, dtype=np.float64, copy=True)
    centered = values-values.mean(axis=1, keepdims=True)
    return _Endpoint(
        values, centered, float(np.sum(centered**2)),
        float(np.sqrt(np.mean(values**2))), _spectral_distribution(centered),
    )


def _metrics(head, tail, blend):
    scale = max(head.rms, tail.rms, .0001)
    incoming = tail.values[:, -1]-tail.values[:, -2]
    outgoing = head.values[:, 1]-head.values[:, 0]
    wrap = head.values[:, 0]-tail.values[:, -1]
    raw = float(np.sqrt(np.mean(wrap**2))) / scale
    # A legitimate periodic waveform need not have equal neighboring
    # samples. Penalize the deviation from the local predicted step,
    # rather than forcing a flat join at every zero crossing.
    jump = float(np.sqrt(np.mean((wrap-.5*(incoming+outgoing))**2))) / scale
    slope = float(np.sqrt(np.mean((incoming-outgoing)**2))) / scale
    if blend:
        jump *= .25
        slope *= .25
    rms = abs(head.rms-tail.rms)/scale
    if min(head.energy, tail.energy) > _EPS:
        corr = float(np.clip(np.sum(head.centered*tail.centered) /
                            math.sqrt(head.energy*tail.energy), -1, 1))
    elif max(head.energy, tail.energy) <= _EPS:
        corr = 1. if np.array_equal(head.values, tail.values) else 0.
    else:
        corr = 0.
    shape = float(.5*np.abs(head.distribution-tail.distribution).sum())
    silence = max(0., 1-max(head.rms, tail.rms)/.001)
    return SeamMetrics(jump, raw, slope, rms, corr, shape, head.rms, tail.rms, silence)


def measure_loop_seam(audio, start_frame, stop_frame, *, context_frames=256,
                      blend_frames=0, weights=LoopScoreWeights()):
    """Inspect one candidate. Report is normalized to local RMS, not dB."""
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    plan = LoopSearchPlan(start_frame, stop_frame, start_radius=0, stop_radius=0,
                          context_frames=context_frames, blend_frames=blend_frames,
                          weights=weights)
    if stop_frame > audio.frames or not plan.valid_pair(start_frame, stop_frame):
        raise AudioContractError("Loop span does not support requested context and crossfade")
    head = _endpoint(audio.samples[:, start_frame:start_frame+context_frames])
    tail = _endpoint(audio.samples[:, stop_frame-context_frames:stop_frame])
    metrics = _metrics(head, tail, blend_frames)
    return LoopCandidate(start_frame, stop_frame, stop_frame-start_frame-blend_frames,
                         0, metrics.score(weights), metrics)


def _grid(center, radius, low, high, stride):
    a, b = max(low, center-radius), min(high, center+radius)
    if a > b:
        return ()
    points = set(range(a, b+1, stride))
    points.update((a, b))
    if a <= center <= b:
        points.add(center)
    return tuple(sorted(points))


class _Searcher:
    def __init__(self, audio, plan):
        self.audio = audio
        self.plan = plan
        self.heads = {}
        self.tails = {}
        self.candidates = {}
        self.pair_work = 0

    def _get(self, position, head):
        cache = self.heads if head else self.tails
        if position not in cache:
            # Include two float64 matrices plus a conservative FFT/band cost.
            cost = (len(self.heads)+len(self.tails)+1)*self.audio.channels*self.plan.context_frames*24
            if cost > MAX_WINDOWS_BYTES:
                raise AudioLimitError("Loop endpoint cache exceeds bounded memory")
            n = self.plan.context_frames
            start, stop = (position, position+n) if head else (position-n, position)
            cache[position] = _endpoint(self.audio.samples[:, start:stop])
        return cache[position]

    def evaluate(self, start, stop):
        key = (start, stop)
        if key in self.candidates or not self.plan.valid_pair(start, stop):
            return False
        self.pair_work += self.audio.channels*self.plan.context_frames
        if self.pair_work > MAX_PAIR_WORK:
            raise AudioLimitError("Loop search correlation work exceeds budget")
        head, tail = self._get(start, True), self._get(stop, False)
        metrics = _metrics(head, tail, self.plan.blend_frames)
        self.candidates[key] = LoopCandidate(
            start, stop, stop-start-self.plan.blend_frames,
            abs(start-self.plan.start_frame)+abs(stop-self.plan.stop_frame),
            metrics.score(self.plan.weights), metrics,
        )
        return True

    def ordered(self):
        return sorted(self.candidates.values(), key=lambda c: (
            c.score, c.offset_distance, c.start_frame, c.stop_frame,
        ))


def search_loop(audio: AudioBuffer, plan: LoopSearchPlan) -> LoopSearchResult:
    if not isinstance(audio, AudioBuffer) or not isinstance(plan, LoopSearchPlan):
        raise AudioContractError("Expected AudioBuffer and LoopSearchPlan")
    if plan.stop_frame > audio.frames or plan.start_frame >= audio.frames:
        raise AudioContractError("Nominal endpoints outside source")
    n = plan.context_frames
    starts = _grid(plan.start_frame, plan.start_radius, 0, audio.frames-n, plan.stride)
    stops = _grid(plan.stop_frame, plan.stop_radius, n, audio.frames, plan.stride)
    if len(starts)*len(stops) > MAX_GRID_PAIRS:
        raise AudioLimitError("Loop coarse grid exceeds pair budget")
    searcher = _Searcher(audio, plan)
    for start in starts:
        for stop in stops:
            searcher.evaluate(start, stop)
    coarse = len(searcher.candidates)
    if not coarse:
        raise AudioContractError("No admissible loop candidate in declared search bounds")
    # Refine the best coarse candidates only. Deduplicate overlaps and obey
    # the same global work and endpoint-cache caps.
    radius = plan.stride-1
    if radius and plan.refine_count:
        seeds = searcher.ordered()[:plan.refine_count]
        pairs = set()
        for seed in seeds:
            lo_s = max(0, plan.start_frame-plan.start_radius, seed.start_frame-radius)
            hi_s = min(audio.frames-n, plan.start_frame+plan.start_radius, seed.start_frame+radius)
            lo_e = max(n, plan.stop_frame-plan.stop_radius, seed.stop_frame-radius)
            hi_e = min(audio.frames, plan.stop_frame+plan.stop_radius, seed.stop_frame+radius)
            estimate = (hi_s-lo_s+1)*(hi_e-lo_e+1)
            if estimate > MAX_REFINED_PAIRS:
                raise AudioLimitError("Loop refinement neighborhood exceeds pair budget")
            for a in range(lo_s, hi_s+1):
                for b in range(lo_e, hi_e+1):
                    if (a, b) not in searcher.candidates and plan.valid_pair(a, b):
                        pairs.add((a, b))
                        if len(pairs) > MAX_REFINED_PAIRS:
                            raise AudioLimitError("Loop refinement exceeds pair budget")
        for start, stop in sorted(pairs):
            searcher.evaluate(start, stop)
    return LoopSearchResult(
        audio.sample_rate, audio.channels, audio.frames, plan, coarse,
        len(searcher.candidates)-coarse, tuple(searcher.ordered()[:plan.results]),
    )
