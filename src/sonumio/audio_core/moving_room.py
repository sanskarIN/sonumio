"""Bounded snapshot interpolation for geometric moving-source room rendering.

Every keyframe has a physically static shoebox impulse response. Offline
renders are sample-clock crossfaded between neighboring snapshots with
piecewise linear partition-of-unity weights. This is an approximation for
slow scene motion, NOT a physical wave-equation solver, Doppler renderer
or sample-accurate moving-path retarded-time acoustics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES
from .errors import AudioContractError, AudioLimitError
from .matrix_convolution import convolve_matrix
from .partitioned_convolution import convolve_partitioned
from .room_response import Point3D, ShoeboxRoom, RoomConfig, build_room_response

MAX_SCENE_KEYS = 16
MAX_SCENE_PRODUCTS = 300_000_000
MAX_SCENE_WORK_BYTES = 192 * 1024 * 1024


def _frame(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AudioContractError(name + " must be a nonnegative integer frame")
    return value


@dataclass(frozen=True, slots=True)
class SceneKeyframe:
    """Source and listener positions at an exact audio frame."""

    frame: int
    sources: tuple[Point3D, ...]
    receivers: tuple[Point3D, ...]

    def __post_init__(self) -> None:
        _frame(self.frame, "scene keyframe")
        if (type(self.sources) is not tuple or not 1 <= len(self.sources) <= 8
                or type(self.receivers) is not tuple
                or not 1 <= len(self.receivers) <= 8
                or any(not isinstance(p, Point3D) for p in self.sources)
                or any(not isinstance(p, Point3D) for p in self.receivers)):
            raise AudioContractError("Scene keyframe needs bounded 3D points")


@dataclass(frozen=True, slots=True)
class MotionRoomPlan:
    room: ShoeboxRoom
    config: RoomConfig
    keys: tuple[SceneKeyframe, ...]
    engine: str = "partitioned"

    def __post_init__(self):
        if not isinstance(self.room, ShoeboxRoom) or not isinstance(self.config, RoomConfig):
            raise AudioContractError("Expected room geometry and response configuration")
        if type(self.keys) is not tuple or not 1 <= len(self.keys) <= MAX_SCENE_KEYS:
            raise AudioContractError("Motion plan requires 1..16 scene keyframes")
        if any(not isinstance(key, SceneKeyframe) for key in self.keys):
            raise AudioContractError("Motion plan expects SceneKeyframe objects")
        positions = [key.frame for key in self.keys]
        if positions[0] != 0 or positions != sorted(set(positions)):
            raise AudioContractError("Scene keyframes must start at zero and increase")
        inputs = len(self.keys[0].sources)
        outputs = len(self.keys[0].receivers)
        for key in self.keys:
            if len(key.sources) != inputs or len(key.receivers) != outputs:
                raise AudioContractError("Motion channel counts must remain constant")
            for p in (*key.sources, *key.receivers):
                self.room.validate_point(p)
        if type(self.engine) is not str or self.engine not in ("matrix", "partitioned"):
            raise AudioContractError("Unknown motion scene convolution engine")

    @property
    def input_channels(self) -> int:
        return len(self.keys[0].sources)

    @property
    def output_channels(self) -> int:
        return len(self.keys[0].receivers)


@dataclass(frozen=True, slots=True)
class SnapshotStats:
    frame: int
    path_count: int
    direct_paths: int
    impulse_taps: int
    input_channels: int
    output_channels: int


@dataclass(frozen=True, slots=True)
class MotionRoomReport:
    sample_rate: int
    source_frames: int
    output_frames: int
    input_channels: int
    output_channels: int
    max_impulse_taps: int
    engine: str
    snapshots: tuple[SnapshotStats, ...]
    input_peak: float
    output_peak: float
    include_tail: bool
    protocol: str = "sonumio-moving-static-room-v1"

    def as_dict(self):
        output = asdict(self)
        output["model_inference"] = False
        output["doppler_simulated"] = False
        output["moving_wavefront_physics_claimed"] = False
        output["sample_clock_changed"] = False
        return output


@dataclass(frozen=True, slots=True)
class MotionRoomResult:
    audio: AudioBuffer
    report: MotionRoomReport


def snapshot_weights(
    positions: tuple[int, ...], index: int, frames: int,
) -> np.ndarray:
    """A deterministic nonnegative triangular basis on the output clock.

    Each frame has weights that sum to one across all scene snapshots,
    including the final reverb tail after the last geometry key.
    """
    if (type(positions) is not tuple or not 1 <= len(positions) <= MAX_SCENE_KEYS
            or any(type(p) is not int or p < 0 for p in positions)
            or positions[0] != 0
            or list(positions) != sorted(set(positions))
            or type(index) is not int or not 0 <= index < len(positions)
            or type(frames) is not int or frames < 1
            or positions[-1] >= frames):
        raise AudioContractError("Invalid scene weight geometry")
    if frames * 8 > MAX_SCENE_WORK_BYTES:
        raise AudioLimitError("Scene weight allocation exceeds budget")
    if len(positions) == 1:
        return np.ones(frames, dtype=np.float64)
    sample = np.arange(frames, dtype=np.float64)
    weights = np.zeros(frames, dtype=np.float64)
    if index == 0:
        weights[:positions[0] + 1] = 1
    if index > 0:
        first = positions[index - 1]
        current = positions[index]
        region = (sample >= first) & (sample <= current)
        weights[region] = (sample[region] - first) / (current - first)
    if index + 1 < len(positions):
        current = positions[index]
        next_position = positions[index + 1]
        region = (sample >= current) & (sample <= next_position)
        weights[region] = (next_position - sample[region]) / (
            next_position - current
        )
    else:
        weights[positions[-1]:] = 1
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
        raise AudioContractError("Motion crossfade produced invalid weights")
    return weights


def _preflight(audio: AudioBuffer, plan: MotionRoomPlan, impulses, include_tail):
    if type(include_tail) is not bool:
        raise AudioContractError("include_tail must be boolean")
    if audio.sample_rate != plan.config.sample_rate:
        raise AudioContractError("Motion room and audio sample clocks differ")
    if audio.channels != plan.input_channels:
        raise AudioContractError("Each audio channel requires one moving source")
    if plan.keys[-1].frame >= audio.frames:
        raise AudioContractError("Scene keys must lie within source duration")
    max_taps = max(response.impulse.geometry.taps for response in impulses)
    frames = audio.frames + (max_taps - 1 if include_tail else 0)
    layout = AudioLayout.for_channels(plan.output_channels)
    AudioSpec(audio.sample_rate, layout, frames)
    # Peak working memory: accumulating float64 output, individual bounded
    # render, weights, float32 result, current filters and transient source.
    scratch = (
        8 * frames * plan.output_channels * 2
        + 8 * frames + 4 * frames * plan.output_channels
        + 4 * audio.frames * audio.channels
    )
    if scratch > MAX_SCENE_WORK_BYTES or frames * plan.output_channels * 8 > MAX_BYTES:
        raise AudioLimitError("Moving room scene exceeds output work memory")
    products = 0
    for response in impulses:
        g = response.impulse.geometry
        g.validate_work(audio.frames)
        products += (
            (audio.frames + g.block - 1) // g.block
            * g.inputs * g.outputs * g.bins
        )
    if products > MAX_SCENE_PRODUCTS:
        raise AudioLimitError("Moving room scene exceeds cumulative spectral work")
    return frames, layout


def render_motion_room(
    audio: AudioBuffer,
    plan: MotionRoomPlan,
    *,
    include_tail: bool = True,
) -> MotionRoomResult:
    """Render keyframed positions with phase-coherent linear amplitude weights.

    All snapshots use the same original source and sample clock. This
    crossfades *rendered responses*, not source positions or Doppler speed.
    In particular the direct-path delay switches continuously in amplitude
    but is not a physically moving emission model.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(plan, MotionRoomPlan):
        raise AudioContractError("Expected AudioBuffer and MotionRoomPlan")
    if type(include_tail) is not bool:
        raise AudioContractError("include_tail must be boolean")
    if audio.sample_rate != plan.config.sample_rate:
        raise AudioContractError("Motion room and audio sample clocks differ")
    if audio.channels != plan.input_channels:
        raise AudioContractError("Motion source topology differs from audio")
    if plan.keys[-1].frame >= audio.frames:
        raise AudioContractError("Scene keyframes extend beyond sound duration")
    responses = [
        build_room_response(plan.room, key.sources, key.receivers, plan.config)
        for key in plan.keys
    ]
    frames, layout = _preflight(audio, plan, responses, include_tail)
    renderer = convolve_matrix if plan.engine == "matrix" else convolve_partitioned
    output = np.zeros((plan.output_channels, frames), dtype=np.float64)
    stats = []
    positions = tuple(key.frame for key in plan.keys)
    for i, (key, response) in enumerate(zip(plan.keys, responses)):
        rendered = renderer(
            audio, response.impulse,
            include_tail=include_tail, output_layout=layout,
        )
        weights = snapshot_weights(positions, i, frames)
        count = rendered.frames
        output[:, :count] += rendered.samples.astype(np.float64) * weights[None, :count]
        stats.append(SnapshotStats(
            key.frame, len(response.paths), len(response.direct_paths),
            response.impulse.geometry.taps, plan.input_channels, plan.output_channels,
        ))
    if not np.isfinite(output).all():
        raise AudioContractError("Moving room generated nonfinite output")
    if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Moving room output exceeds finite float32 range")
    result = audio.copy_with(output.astype(np.float32),
                             source_note="interpolated static room snapshots"
                             ) if plan.output_channels == audio.channels and (
                                 not include_tail or frames == audio.frames
                             ) else None
    if result is None:
        from .contracts import make_buffer
        result = make_buffer(output.astype(np.float32), audio.sample_rate,
                             layout=layout, source_note="interpolated room snapshots")
    report = MotionRoomReport(
        audio.sample_rate, audio.frames, result.frames, audio.channels,
        result.channels, max(s.impulse_taps for s in stats), plan.engine,
        tuple(stats), float(np.max(np.abs(audio.samples))),
        float(np.max(np.abs(result.samples))), include_tail,
    )
    return MotionRoomResult(result, report)
