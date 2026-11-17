"""Immutable PCM editing primitives with explicit frame-level boundaries.

Trim, concatenate, crossfade, pad, reverse, loop and silence segmentation
operate on sample buffers only. No file system writes and no implicit
resampling or speculative voice diarization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from .contracts import (
    AudioBuffer, AudioSpec, make_buffer, MAX_BYTES, finite_float
)
from .errors import AudioContractError, AudioLimitError
from .numerics import db_to_gain, equal_power_ramps


def trim(audio: AudioBuffer, start: int = 0, stop: int | None = None
         ) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    end = audio.frames if stop is None else stop
    return audio.sliced(start, end)


def reverse(audio: AudioBuffer) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    return audio.copy_with(audio.samples[:, ::-1],
                           source_note="reverse, nondestructive")


def pad(audio: AudioBuffer, *, before: int = 0,
        after: int = 0, mode: str = "zeros") -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if (type(before) is not int or type(after) is not int
            or before < 0 or after < 0):
        raise AudioContractError("Pad sizes must be nonnegative integer frames")
    if mode not in ("zeros", "edge", "reflect"):
        raise AudioContractError("Unknown PCM pad mode")
    frames = audio.frames + before + after
    AudioSpec(audio.sample_rate, audio.spec.layout, frames)
    if mode == "reflect" and audio.frames == 1 and (before or after):
        raise AudioContractError("Reflect padding needs at least two samples")
    mode_numpy = "constant" if mode == "zeros" else mode
    data = np.pad(audio.samples, ((0, 0), (before, after)), mode=mode_numpy)
    return audio.copy_with(data, source_note="explicit PCM pad")


def concatenate(buffers: Sequence[AudioBuffer], *,
                max_seconds: float = 120) -> AudioBuffer:
    if not isinstance(buffers, (list, tuple)) or not 1 <= len(buffers) <= 256:
        raise AudioContractError("Expected bounded sequence of AudioBuffers")
    if any(not isinstance(x, AudioBuffer) for x in buffers):
        raise AudioContractError("All inputs must be AudioBuffers")
    head = buffers[0]
    for item in buffers:
        if not head.spec.same_clock(item.spec) or head.channels != item.channels:
            raise AudioContractError("Cannot concatenate unequal channel/sample clocks")
    count = sum(x.frames for x in buffers)
    cap = finite_float(max_seconds, "concatenation max_seconds", minimum=.01,
                       maximum=120)
    if count > min(cap, 120) * head.sample_rate:
        raise AudioLimitError("Concatenated duration exceeds explicit bound")
    data = np.concatenate([x.samples for x in buffers], axis=1)
    return head.copy_with(data, source_note="nondestructive PCM concat")


def crossfade(left: AudioBuffer, right: AudioBuffer, *, frames: int,
              law: str = "equal_power") -> AudioBuffer:
    if not isinstance(left, AudioBuffer) or not isinstance(right, AudioBuffer):
        raise AudioContractError("Crossfade requires two AudioBuffers")
    if not left.spec.same_clock(right.spec) or left.channels != right.channels:
        raise AudioContractError("Crossfade source clocks and channels must agree")
    if type(frames) is not int or not 1 <= frames <= min(left.frames, right.frames):
        raise AudioContractError("Crossfade length exceeds source segment")
    if law not in ("linear", "equal_power", "cosine"):
        raise AudioContractError("Unknown crossfade interpolation law")
    if law == "linear":
        tail, head = (
            np.linspace(1, 0, frames, dtype=np.float64),
            np.linspace(0, 1, frames, dtype=np.float64)
        )
    elif law == "equal_power":
        tail, head = equal_power_ramps(frames) if frames > 1 else (
            np.array([0.5]), np.array([0.5])
        )
        if frames > 1:
            tail, head = tail, head
    else:
        t = np.linspace(0, 1, frames, dtype=np.float64)
        tail, head = np.cos(math.pi / 2 * t) ** 2, np.sin(math.pi / 2 * t) ** 2
    middle = (left.samples[:, -frames:].astype(np.float64) * tail[None, :]
              + right.samples[:, :frames].astype(np.float64) * head[None, :])
    data = np.concatenate([
        left.samples[:, :-frames], middle.astype(np.float32),
        right.samples[:, frames:]
    ], axis=1)
    return left.copy_with(data, source_note="crossfade " + law)


def loop(audio: AudioBuffer, repeats: int, *, blend_frames: int = 0,
         max_seconds: float = 120) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if type(repeats) is not int or not 1 <= repeats <= 128:
        raise AudioContractError("Repeats must be 1..128")
    if type(blend_frames) is not int or not 0 <= blend_frames < audio.frames:
        raise AudioContractError("Invalid loop crossfade span")
    predicted = audio.frames * repeats - blend_frames * (repeats - 1)
    cap = finite_float(max_seconds, "loop max_seconds", minimum=.01, maximum=120)
    if predicted > audio.sample_rate * cap:
        raise AudioLimitError("Loop exceeds bounded output duration")
    if repeats == 1:
        return audio.copy_with(audio.samples, source_note="single loop")
    result = audio
    for _ in range(repeats - 1):
        if blend_frames:
            result = crossfade(result, audio, frames=blend_frames, law="linear")
        else:
            result = concatenate((result, audio), max_seconds=max_seconds)
    return result


def split_at(audio: AudioBuffer, boundaries: Sequence[int]
             ) -> tuple[AudioBuffer, ...]:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if not isinstance(boundaries, (list, tuple)) or len(boundaries) > 1000:
        raise AudioContractError("Invalid split boundaries")
    if (any(type(k) is not int for k in boundaries)
            or list(boundaries) != sorted(set(boundaries))
            or any(not 0 < k < audio.frames for k in boundaries)):
        raise AudioContractError("Split frame positions must be sorted and unique")
    positions = (0, *boundaries, audio.frames)
    return tuple(audio.sliced(a, b)
                 for a, b in zip(positions[:-1], positions[1:]))


def chunks(audio: AudioBuffer, frames: int, *, drop_last: bool = False
           ) -> tuple[AudioBuffer, ...]:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if type(frames) is not int or frames <= 0 or frames > audio.frames:
        raise AudioContractError("Invalid chunk length")
    if type(drop_last) is not bool:
        raise AudioContractError("drop_last must be boolean")
    count = audio.frames // frames
    pieces = [
        audio.sliced(k * frames, (k + 1) * frames) for k in range(count)
    ]
    if not drop_last and count * frames < audio.frames:
        pieces.append(audio.sliced(count * frames, audio.frames))
    return tuple(pieces)


@dataclass(frozen=True, slots=True)
class SilenceRegion:
    start_frame: int
    stop_frame: int
    maximum_amplitude: float

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame


def detect_silence(audio: AudioBuffer, *, threshold_dbfs: float = -55,
                   window_frames: int = 1024,
                   minimum_frames: int = 2048) -> tuple[SilenceRegion, ...]:
    """Block-based silence flags, conservative all-channel peak detector."""
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    db = finite_float(threshold_dbfs, "silence threshold", minimum=-120, maximum=0)
    if (type(window_frames) is not int or not 16 <= window_frames <= 65536
            or type(minimum_frames) is not int or minimum_frames < 1):
        raise AudioContractError("Invalid silence window/region length")
    cutoff = db_to_gain(db)
    amplitude = np.max(np.abs(audio.samples.astype(np.float64)), axis=0)
    silent = []
    for start in range(0, audio.frames, window_frames):
        stop = min(start + window_frames, audio.frames)
        silent.append((start, stop, float(amplitude[start:stop].max()) < cutoff))
    regions = []
    first = None
    for start, stop, is_silent in silent:
        if is_silent and first is None:
            first = start
        if not is_silent and first is not None:
            if start - first >= minimum_frames:
                regions.append(SilenceRegion(first, start,
                                              float(amplitude[first:start].max())))
            first = None
    if first is not None and audio.frames - first >= minimum_frames:
        regions.append(SilenceRegion(first, audio.frames,
                                      float(amplitude[first:].max())))
    return tuple(regions)


def trim_edges_by_silence(audio: AudioBuffer, *,
                          threshold_dbfs: float = -55,
                          window_frames: int = 1024,
                          keep_frames: int = 0) -> AudioBuffer:
    if type(keep_frames) is not int or keep_frames < 0:
        raise AudioContractError("Invalid silence guard")
    regions = detect_silence(audio, threshold_dbfs=threshold_dbfs,
                              window_frames=window_frames,
                              minimum_frames=1)
    left = regions[0].stop_frame if regions and regions[0].start_frame == 0 else 0
    right = (regions[-1].start_frame
             if regions and regions[-1].stop_frame == audio.frames
             else audio.frames)
    if left >= right:
        raise AudioContractError("Entire audio is silent at chosen threshold")
    return audio.sliced(max(0, left - keep_frames),
                        min(audio.frames, right + keep_frames))
