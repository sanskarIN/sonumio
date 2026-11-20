"""Nondestructive, sample-aligned local sound timelines and track routing.

A timeline never calls a neural model or a file path. Each clip refers to an
already validated, immutable AudioBuffer. Positions and fades use integer
source-clock frames; the renderer accumulates in float64 with a bounded
working memory budget and explicit output limiting policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import math
import re
import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES, check_rate, make_buffer,
)
from .errors import AudioContractError, AudioLimitError
from .numerics import db_to_gain, balance_stereo, stable_peak
from .windows import envelope

_TRACK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class Clip:
    """Bounded placement of a source excerpt onto a shared timeline."""

    clip_id: str
    source: AudioBuffer
    start_frame: int
    source_start: int = 0
    source_stop: int | None = None
    gain_db: float = 0
    pan: float = 0
    fade_in: int = 0
    fade_out: int = 0
    fade_curve: str = "cosine"
    mute: bool = False

    def __post_init__(self):
        if not isinstance(self.clip_id, str) or not _TRACK_ID.fullmatch(self.clip_id):
            raise AudioContractError("Invalid clip identifier")
        if not isinstance(self.source, AudioBuffer):
            raise AudioContractError("Clip source must be an AudioBuffer")
        stop = self.source.frames if self.source_stop is None else self.source_stop
        if (type(self.start_frame) is not int or self.start_frame < 0
                or type(self.source_start) is not int or self.source_start < 0
                or type(stop) is not int or
                not self.source_start < stop <= self.source.frames):
            raise AudioContractError("Invalid source region or placement")
        object.__setattr__(self, "source_stop", stop)
        if (type(self.fade_in) is not int or type(self.fade_out) is not int
                or self.fade_in < 0 or self.fade_out < 0
                or self.fade_in + self.fade_out > stop - self.source_start):
            raise AudioContractError("Clip fades extend beyond selected source")
        if self.fade_curve not in ("linear", "cosine", "smoothstep", "equal_power"):
            raise AudioContractError("Unknown clip fade curve")
        from .contracts import finite_float
        finite_float(self.gain_db, "clip gain", minimum=-120, maximum=48)
        finite_float(self.pan, "clip pan", minimum=-1, maximum=1)
        if type(self.mute) is not bool:
            raise AudioContractError("mute flag must be boolean")

    @property
    def frames(self) -> int:
        return self.source_stop - self.source_start

    @property
    def stop_frame(self) -> int:
        return self.start_frame + self.frames

    def shifted(self, frames: int) -> "Clip":
        if type(frames) is not int or self.start_frame + frames < 0:
            raise AudioContractError("Cannot shift a clip before timeline origin")
        from dataclasses import replace
        return replace(self, start_frame=self.start_frame + frames)

    def subclip(self, start: int, stop: int) -> "Clip":
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.frames):
            raise AudioContractError("Subclip source bounds invalid")
        from dataclasses import replace
        fade_in = min(self.fade_in, stop - start)
        fade_out = min(self.fade_out, stop - start - fade_in)
        return replace(self, source_start=self.source_start + start,
                       source_stop=self.source_start + stop,
                       start_frame=self.start_frame + start,
                       fade_in=fade_in, fade_out=fade_out)


@dataclass(frozen=True, slots=True)
class Track:
    track_id: str
    clips: tuple[Clip, ...]
    gain_db: float = 0
    mute: bool = False
    solo: bool = False

    def __post_init__(self):
        if not isinstance(self.track_id, str) or not _TRACK_ID.fullmatch(self.track_id):
            raise AudioContractError("Invalid track id")
        if not isinstance(self.clips, tuple) or len(self.clips) > 256:
            raise AudioLimitError("Track accepts at most 256 clips")
        if any(not isinstance(clip, Clip) for clip in self.clips):
            raise AudioContractError("Invalid track clip")
        if len({clip.clip_id for clip in self.clips}) != len(self.clips):
            raise AudioContractError("Duplicate clip IDs in one track")
        from .contracts import finite_float
        finite_float(self.gain_db, "track gain", minimum=-120, maximum=48)
        if type(self.mute) is not bool or type(self.solo) is not bool:
            raise AudioContractError("Track mute/solo flags must be boolean")

    def with_clip(self, clip: Clip) -> "Track":
        return Track(self.track_id, self.clips + (clip,), self.gain_db,
                     self.mute, self.solo)

    def without_clip(self, clip_id: str) -> "Track":
        return Track(self.track_id, tuple(
            c for c in self.clips if c.clip_id != clip_id
        ), self.gain_db, self.mute, self.solo)


@dataclass(frozen=True, slots=True)
class Timeline:
    """Fixed sample clock and output channel layout with nondestructive tracks."""

    sample_rate: int
    output_layout: AudioLayout
    frames: int
    tracks: tuple[Track, ...] = ()

    def __post_init__(self):
        check_rate(self.sample_rate)
        if not isinstance(self.output_layout, AudioLayout):
            raise AudioContractError("Expected explicit output channel topology")
        AudioSpec(self.sample_rate, self.output_layout, self.frames)
        if not isinstance(self.tracks, tuple) or len(self.tracks) > 64:
            raise AudioLimitError("Timeline supports at most 64 tracks")
        if len({x.track_id for x in self.tracks}) != len(self.tracks):
            raise AudioContractError("Duplicate timeline track identifiers")
        if sum(len(x.clips) for x in self.tracks) > 512:
            raise AudioLimitError("Timeline supports at most 512 total clips")
        for track in self.tracks:
            if not isinstance(track, Track):
                raise AudioContractError("Invalid timeline track")
            for clip in track.clips:
                if clip.source.sample_rate != self.sample_rate:
                    raise AudioContractError("Timeline and source sample clocks differ")
                if clip.stop_frame > self.frames:
                    raise AudioContractError("Clip extends beyond timeline")
                if clip.source.channels not in (1, self.output_layout.channels):
                    raise AudioContractError("Unsupported clip/output topology")
                if clip.pan and self.output_layout.channels != 2:
                    raise AudioContractError("Panning only supports stereo output")

    def with_track(self, track: Track) -> "Timeline":
        return Timeline(self.sample_rate, self.output_layout, self.frames,
                        self.tracks + (track,))

    def without_track(self, track_id: str) -> "Timeline":
        return Timeline(self.sample_rate, self.output_layout, self.frames,
                        tuple(t for t in self.tracks if t.track_id != track_id))

    def reorder_tracks(self, order: Sequence[str]) -> "Timeline":
        existing = {t.track_id: t for t in self.tracks}
        if set(order) != set(existing) or len(order) != len(self.tracks):
            raise AudioContractError("New order must contain each track exactly once")
        return Timeline(self.sample_rate, self.output_layout, self.frames,
                        tuple(existing[x] for x in order))

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate


def _route_clip(source: np.ndarray, output_channels: int, pan: float
                ) -> np.ndarray:
    """Explicit mono-to-stereo constant-power panning and stereo balancing."""
    source_channels = source.shape[0]
    if source_channels == output_channels:
        if output_channels == 2 and pan:
            return balance_stereo(source.astype(np.float32), pan).astype(np.float64)
        return source
    if source_channels != 1 or output_channels != 2:
        raise AudioContractError("Clip routing requires equal channels or mono to stereo")
    angle = (float(pan) + 1.0) * math.pi / 4
    return np.vstack((source[0] * math.cos(angle),
                      source[0] * math.sin(angle)))


def render_timeline(timeline: Timeline, *,
                    normalization: str = "none",
                    master_gain_db: float = 0,
                    output_note: str = "local multi-track mix") -> AudioBuffer:
    """Render deterministic offline mix, with optional peak normalization.

    Master normalization='none' preserves overrange float32 signals for
    subsequent explicit limiter/PCM handling. No implicit overwrite occurs.
    """
    if not isinstance(timeline, Timeline):
        raise AudioContractError("Expected Timeline")
    if normalization not in ("none", "peak", "soft"):
        raise AudioContractError("Unknown timeline master normalization")
    from .contracts import finite_float
    master_db = finite_float(master_gain_db, "master gain", minimum=-120, maximum=48)
    memory = timeline.output_layout.channels * timeline.frames * 12
    if memory > MAX_BYTES:
        raise AudioLimitError("Timeline render working memory exceeds cap")
    mix = np.zeros((timeline.output_layout.channels, timeline.frames), dtype=np.float64)
    has_solo = any(t.solo and not t.mute for t in timeline.tracks)
    for track in timeline.tracks:
        if track.mute or (has_solo and not track.solo):
            continue
        track_gain = db_to_gain(track.gain_db)
        for clip in track.clips:
            if clip.mute:
                continue
            piece = clip.source.samples[:, clip.source_start:clip.source_stop]
            converted = _route_clip(
                piece.astype(np.float64), timeline.output_layout.channels, clip.pan
            )
            gain = db_to_gain(clip.gain_db) * track_gain
            curve = envelope(clip.frames, fade_in=clip.fade_in,
                             fade_out=clip.fade_out, curve=clip.fade_curve)
            start, stop = clip.start_frame, clip.stop_frame
            mix[:, start:stop] += converted * (gain * curve)[None, :]
    mix *= db_to_gain(master_db)
    if normalization == "peak":
        peak = float(np.max(np.abs(mix)))
        if peak > 1:
            mix /= peak
    elif normalization == "soft":
        mix = np.tanh(mix)
    if not np.isfinite(mix).all():
        raise AudioContractError("Timeline mix produced nonfinite samples")
    return make_buffer(mix.astype(np.float32), timeline.sample_rate,
                       layout=timeline.output_layout, source_note=output_note)


def timeline_report(timeline: Timeline, *,
                    rendered: AudioBuffer | None = None) -> dict:
    if not isinstance(timeline, Timeline):
        raise AudioContractError("Expected Timeline")
    if rendered is not None and (
        not isinstance(rendered, AudioBuffer)
        or rendered.sample_rate != timeline.sample_rate
        or rendered.frames != timeline.frames
        or rendered.channels != timeline.output_layout.channels
    ):
        raise AudioContractError("Rendered buffer does not match timeline geometry")
    return {
        "sample_rate": timeline.sample_rate,
        "frames": timeline.frames,
        "seconds": timeline.seconds,
        "output_channels": timeline.output_layout.channels,
        "track_count": len(timeline.tracks),
        "clip_count": sum(len(t.clips) for t in timeline.tracks),
        "solo_count": sum(t.solo for t in timeline.tracks),
        "muted_tracks": sum(t.mute for t in timeline.tracks),
        "render_peak": stable_peak(rendered.samples) if rendered is not None else None,
        "not_neural_inference": True,
    }
