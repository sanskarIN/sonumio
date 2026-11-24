"""Bounded, packet-invariant time-localized WAV signal diagnostics.

Windows are aligned to absolute PCM frames, independent of incoming packet
boundaries. This module never retains the audio itself. The caller must
authenticate its source independently: a completed timeline is *not* a
cryptographic proof, perceptual score or source-rights verification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .wave_stream import MAX_STREAM_SECONDS

MAX_TIMELINE_WINDOWS = 4096


@dataclass(frozen=True, slots=True)
class QualityWindow:
    """Inclusive start/exclusive stop on the original sample timeline."""

    start_frame: int
    stop_frame: int
    sample_rate: int
    channels: int
    peak_amplitude: float
    rms_amplitude: float
    dc_offset_absolute: float
    clipping_fraction: float
    active_fraction: float

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame

    def as_dict(self) -> dict:
        data = asdict(self)
        data["frames"] = self.frames
        data["start_seconds"] = self.start_frame / self.sample_rate
        data["stop_seconds"] = self.stop_frame / self.sample_rate
        return data


@dataclass(frozen=True, slots=True)
class QualityTimeline:
    sample_rate: int
    channels: int
    frames: int
    window_frames: int
    windows: tuple[QualityWindow, ...]
    protocol: str = "sonumio-stream-window-signal-health-v1"

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "window_frames": self.window_frames,
            "window_count": len(self.windows),
            "windows": [record.as_dict() for record in self.windows],
            "perceptual_audio_quality_measured": False,
            "text_prompt_adherence_measured": False,
        }


class StreamingQualityTimeline:
    """Measure adjacent fixed-duration PCM windows using bounded memory.

    The sample rate defines the default one-second window. Fragmentation
    between valid sequential FramePackets does not change window boundaries.
    Explicit windows up to 60 seconds are permitted; at most 4096 windows
    are retained, including the final partial window.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        window_frames: int | None = None,
        expected_frames: int | None = None,
        activity_floor_dbfs: float = -70.0,
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid timeline channel count")
        window = sample_rate if window_frames is None else window_frames
        if type(window) is not int or not 1 <= window <= 60 * sample_rate:
            raise AudioContractError("Timeline window must be 1 frame to 60 seconds")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= MAX_STREAM_SECONDS * sample_rate
        ):
            raise AudioLimitError("Invalid timeline expected frame count")
        if (expected_frames is not None
                and (expected_frames + window - 1) // window > MAX_TIMELINE_WINDOWS):
            raise AudioLimitError("Timeline would exceed bounded window capacity")
        db = finite_float(
            activity_floor_dbfs, "timeline activity floor",
            minimum=-160.0, maximum=0.0,
        )
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_frames = window
        self.expected_frames = expected_frames
        self.activity_floor_dbfs = db
        self._threshold = 10.0 ** (db / 20)
        self.frames = 0
        self._start = 0
        self._window_count = 0
        self._sum = np.zeros(channels, dtype=np.float64)
        self._square = np.zeros(channels, dtype=np.float64)
        self._peak = 0.0
        self._active = 0
        self._clipped = 0
        self._windows: list[QualityWindow] = []
        self._finished = False

    def _seal(self) -> None:
        count = self._window_count
        if count == 0:
            return
        if len(self._windows) >= MAX_TIMELINE_WINDOWS:
            raise AudioLimitError("Too many quality timeline windows")
        record = QualityWindow(
            start_frame=self._start,
            stop_frame=self._start + count,
            sample_rate=self.sample_rate,
            channels=self.channels,
            peak_amplitude=self._peak,
            rms_amplitude=float(np.sqrt(np.sum(self._square) / (count * self.channels))),
            dc_offset_absolute=float(np.max(np.abs(self._sum / count))),
            clipping_fraction=self._clipped / (count * self.channels),
            active_fraction=self._active / count,
        )
        self._windows.append(record)
        self._start += count
        self._window_count = 0
        self._sum.fill(0.0)
        self._square.fill(0.0)
        self._peak = 0.0
        self._active = 0
        self._clipped = 0

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Quality timeline is already finalized")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected FramePacket for quality timeline")
        if (packet.sample_rate != self.sample_rate
                or packet.channels != self.channels
                or packet.start_frame != self.frames):
            raise AudioContractError("Timeline packets must be clock-aligned and consecutive")
        total = self.frames + packet.frames
        if total > MAX_STREAM_SECONDS * self.sample_rate or (
            self.expected_frames is not None and total > self.expected_frames
        ):
            raise AudioLimitError("Quality timeline exceeded the declared length")
        if (total + self.window_frames - 1) // self.window_frames > MAX_TIMELINE_WINDOWS:
            raise AudioLimitError("Quality timeline window capacity exceeded")
        offset = 0
        while offset < packet.frames:
            take = min(
                self.window_frames - self._window_count,
                packet.frames - offset,
            )
            segment = packet.samples[:, offset:offset + take].astype(np.float64)
            absolute = np.abs(segment)
            self._sum += np.sum(segment, axis=1, dtype=np.float64)
            self._square += np.sum(segment * segment, axis=1, dtype=np.float64)
            self._peak = max(self._peak, float(np.max(absolute)))
            self._clipped += int(np.count_nonzero(absolute >= 1.0))
            self._active += int(np.count_nonzero(
                np.max(absolute, axis=0) > self._threshold
            ))
            self._window_count += take
            offset += take
            if self._window_count == self.window_frames:
                self._seal()
        self.frames = total

    def finalize(self) -> QualityTimeline:
        if self._finished:
            raise AudioContractError("Quality timeline is already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete quality timeline frame stream")
        self._seal()
        self._finished = True
        if (not self._windows or self._windows[0].start_frame != 0
                or self._windows[-1].stop_frame != self.frames):
            raise AudioIntegrityError("Quality timeline interval coverage mismatch")
        return QualityTimeline(
            self.sample_rate, self.channels, self.frames,
            self.window_frames, tuple(self._windows),
        )
