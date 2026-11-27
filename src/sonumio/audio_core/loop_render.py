"""Exact circular PCM loop construction and repetition.

Crossfade geometry explicitly consumes an overlap from the source span.
The output is rotated to begin with the head continuation and ends with
the blended tail/head overlap. At the wrap, the final blend sample and
the first output sample follow adjacent source-head frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .contracts import AudioBuffer, AudioSpec, finite_float
from .errors import AudioContractError, AudioLimitError, AudioProcessingError
from .loop_analysis import integer, LoopCandidate

LAWS = ("linear", "cosine", "equal_power")
MAX_LOOP_WORK_BYTES = 128*1024*1024


@dataclass(frozen=True, slots=True)
class LoopRenderPlan:
    start_frame: int
    stop_frame: int
    blend_frames: int = 0
    law: str = "cosine"
    repeats: int = 1
    peak_ceiling: float | None = None

    def __post_init__(self):
        integer(self.start_frame, "start_frame", 0, 2**31-1)
        integer(self.stop_frame, "stop_frame", self.start_frame+2, 2**31-1)
        integer(self.blend_frames, "blend_frames", 0, (self.stop_frame-self.start_frame)//2)
        if self.blend_frames == 1:
            raise AudioContractError("Loop blend must be zero or at least two frames")
        if self.law not in LAWS:
            raise AudioContractError("Unknown loop crossfade law")
        integer(self.repeats, "repeats", 1, 1024)
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling, "peak ceiling", minimum=.001, maximum=1.)

    @property
    def period_frames(self):
        return self.stop_frame-self.start_frame-self.blend_frames

    @classmethod
    def from_candidate(cls, candidate, *, law="cosine", repeats=1, peak_ceiling=None):
        if not isinstance(candidate, LoopCandidate):
            raise AudioContractError("Expected LoopCandidate")
        overlap = candidate.stop_frame-candidate.start_frame-candidate.period_frames
        return cls(candidate.start_frame, candidate.stop_frame, overlap, law, repeats, peak_ceiling)


@dataclass(frozen=True, slots=True)
class CircularDiagnostics:
    boundary_step_max: float
    boundary_step_rms: float
    boundary_slope_change_max: float
    max_internal_step: float
    rms_internal_step: float
    boundary_to_rms_ratio: float | None
    sample_peak: float


@dataclass(frozen=True, slots=True)
class LoopRenderReport:
    plan: LoopRenderPlan
    sample_rate: int
    channels: int
    period_frames: int
    output_frames: int
    rotation_source_frames: int
    pre_limit_peak: float
    attenuation: float
    diagnostics: CircularDiagnostics
    protocol: str = "sonumio-circular-pcm-loop-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "period_seconds": self.period_frames/self.sample_rate,
            "crossfade_consumes_source_overlap": True,
            "perceptual_seamlessness_established": False,
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class LoopRenderResult:
    audio: AudioBuffer
    report: LoopRenderReport


def circular_diagnostics(samples):
    """Measure actual wrap and interior steps, with finite silence semantics."""
    if (not isinstance(samples, np.ndarray) or samples.ndim != 2
            or samples.shape[1] < 1 or samples.shape[0] < 1
            or samples.dtype not in (np.float32, np.float64)
            or not np.isfinite(samples).all()):
        raise AudioContractError("Expected finite channel-first cycle PCM")
    x = samples.astype(np.float64)
    wrap = x[:, 0]-x[:, -1]
    if x.shape[1] > 1:
        steps = np.diff(x, axis=1)
        internal_max = float(np.max(np.abs(steps)))
        internal_rms = float(np.sqrt(np.mean(steps**2)))
        slope = max(float(np.max(np.abs(wrap-steps[:, -1]))),
                    float(np.max(np.abs(steps[:, 0]-wrap))))
    else:
        internal_max = internal_rms = slope = 0.
    wrap_rms = float(np.sqrt(np.mean(wrap**2)))
    return CircularDiagnostics(
        float(np.max(np.abs(wrap))), wrap_rms, slope, internal_max,
        internal_rms, wrap_rms/internal_rms if internal_rms > 1e-15 else None,
        float(np.max(np.abs(x))),
    )


def _ramps(frames, law):
    t = np.linspace(0., 1., frames)
    if law == "linear":
        a, b = 1-t, t
    elif law == "cosine":
        b = .5-.5*np.cos(np.pi*t)
        a = 1-b
    else:
        a, b = np.cos(np.pi*t/2), np.sin(np.pi*t/2)
    # Explicit endpoint assignments eliminate residual trig roundoff.
    a[0], a[-1], b[0], b[-1] = 1., 0., 0., 1.
    return a, b


def render_loop(audio, plan):
    if not isinstance(audio, AudioBuffer) or not isinstance(plan, LoopRenderPlan):
        raise AudioContractError("Expected AudioBuffer and LoopRenderPlan")
    if plan.stop_frame > audio.frames:
        raise AudioContractError("Loop stop exceeds source")
    period = plan.period_frames
    total = period*plan.repeats
    AudioSpec(audio.sample_rate, audio.spec.layout, total)
    # Include cycle and diagnostic scratch plus repeated PCM output/copy.
    estimate = audio.channels*(period*40 + total*12)
    if estimate > MAX_LOOP_WORK_BYTES:
        raise AudioLimitError("Loop render exceeds working-memory budget")
    source = audio.samples[:, plan.start_frame:plan.stop_frame]
    blend = plan.blend_frames
    if not blend:
        cycle = source.astype(np.float64)
    else:
        tail_gain, head_gain = _ramps(blend, plan.law)
        overlap = (source[:, -blend:].astype(np.float64)*tail_gain
                   + source[:, :blend].astype(np.float64)*head_gain)
        cycle = np.concatenate((source[:, blend:-blend], overlap), axis=1)
    if cycle.shape != (audio.channels, period) or not np.isfinite(cycle).all():
        raise AudioProcessingError("Loop synthesis geometry or numerics failed")
    peak = float(np.max(np.abs(cycle)))
    attenuation = 1.
    if plan.peak_ceiling is not None and peak > plan.peak_ceiling:
        attenuation = plan.peak_ceiling/peak
        cycle *= attenuation
    if np.max(np.abs(cycle)) > np.finfo(np.float32).max:
        raise AudioProcessingError("Loop PCM exceeds float32 representation")
    cycle = cycle.astype(np.float32)
    diagnostics = circular_diagnostics(cycle)
    output = np.tile(cycle, (1, plan.repeats))
    result = audio.copy_with(output, source_note="original circular PCM loop")
    return LoopRenderResult(result, LoopRenderReport(
        plan, audio.sample_rate, audio.channels, period, total, blend,
        peak, attenuation, diagnostics,
    ))
