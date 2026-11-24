"""Packet-invariant adjacent-frame step metrics for canonical PCM streams.

This is a decoded-sample continuity diagnostic. Legitimate percussive attacks
may produce large steps: neither high step values nor a threshold breach prove
an audible click. A source digest is verified by the file reader, not by this
standalone accumulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .wave_stream import MAX_STREAM_SECONDS


@dataclass(frozen=True, slots=True)
class FrameTransitionReport:
    sample_rate: int
    channels: int
    frames: int
    transitions_per_channel: int
    max_adjacent_step: float
    channel_max_steps: tuple[float, ...]
    max_step_at_frame: int | None
    max_step_channel: int | None
    max_allowed_step: float | None
    over_limit_samples: int | None
    protocol: str = "sonumio-pcm-adjacent-step-v1"

    @property
    def passed(self) -> bool:
        return self.max_allowed_step is None or self.over_limit_samples == 0

    def as_dict(self) -> dict:
        samples = self.channels * self.transitions_per_channel
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "transitions_per_channel": self.transitions_per_channel,
            "max_adjacent_step": self.max_adjacent_step,
            "channel_max_steps": list(self.channel_max_steps),
            "max_step_at_frame": self.max_step_at_frame,
            "max_step_channel": self.max_step_channel,
            "max_allowed_step": self.max_allowed_step,
            "over_limit_samples": self.over_limit_samples,
            "over_limit_fraction": (
                None if self.over_limit_samples is None
                else 0.0 if samples == 0
                else self.over_limit_samples / samples
            ),
            "passed": self.passed,
            "audible_click_proven": False,
            "perceptual_audio_quality_measured": False,
        }


class StreamingFrameTransitionAccumulator:
    """Track all adjacent sample steps, including across packet boundaries.

    A single last-frame vector is retained, plus a fixed per-channel maximum.
    A partial stream must not be treated as a verified or complete file.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        expected_frames: int | None = None,
        max_allowed_step: float | None = None,
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid frame-transition channel count")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= MAX_STREAM_SECONDS * sample_rate
        ):
            raise AudioLimitError("Invalid frame-transition expected frames")
        if max_allowed_step is not None:
            max_allowed_step = finite_float(
                max_allowed_step, "max_adjacent_step",
                minimum=0.0, maximum=2.0,
            )
        self.sample_rate = sample_rate
        self.channels = channels
        self.expected_frames = expected_frames
        self.max_allowed_step = max_allowed_step
        self.frames = 0
        self._last: np.ndarray | None = None
        self._maximum = np.zeros(channels, dtype=np.float64)
        self._highest_step = 0.0
        self._max_step_at_frame: int | None = None
        self._max_step_channel: int | None = None
        self._over = 0
        self._finished = False

    def _take(self, steps: np.ndarray, first_stop_frame: int) -> None:
        if steps.size == 0:
            return
        # Time is prioritized over channel for deterministic earliest-event
        # ties. For equal steps at one frame, select the lowest channel index.
        frame_peaks = np.max(steps, axis=0)
        offset = int(np.argmax(frame_peaks))
        peak = float(frame_peaks[offset])
        if self._max_step_at_frame is None or peak > self._highest_step:
            self._highest_step = peak
            self._max_step_at_frame = first_stop_frame + offset
            self._max_step_channel = int(np.argmax(steps[:, offset]))
        self._maximum = np.maximum(
            self._maximum, np.max(steps, axis=1),
        )
        if self.max_allowed_step is not None:
            self._over += int(np.count_nonzero(steps > self.max_allowed_step))

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Cannot update finalized frame transitions")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected FramePacket for transitions")
        if (packet.sample_rate != self.sample_rate
                or packet.channels != self.channels
                or packet.start_frame != self.frames):
            raise AudioContractError("Frame transitions require consecutive clock-matched packets")
        total = self.frames + packet.frames
        if total > self.sample_rate * MAX_STREAM_SECONDS or (
            self.expected_frames is not None and total > self.expected_frames
        ):
            raise AudioLimitError("Frame transitions exceed declared duration")
        data = packet.samples.astype(np.float64)
        if self._last is not None:
            self._take(np.abs(data[:, :1] - self._last[:, None]), packet.start_frame)
        if packet.frames >= 2:
            self._take(np.abs(np.diff(data, axis=1)), packet.start_frame + 1)
        self._last = data[:, -1].copy()
        self.frames = total

    def finalize(self) -> FrameTransitionReport:
        if self._finished:
            raise AudioContractError("Frame transition accumulator already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete frame transition stream")
        self._finished = True
        peaks = tuple(float(x) for x in self._maximum)
        return FrameTransitionReport(
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames=self.frames,
            transitions_per_channel=self.frames - 1,
            max_adjacent_step=max(peaks),
            channel_max_steps=peaks,
            max_step_at_frame=self._max_step_at_frame,
            max_step_channel=self._max_step_channel,
            max_allowed_step=self.max_allowed_step,
            over_limit_samples=(
                None if self.max_allowed_step is None else self._over
            ),
        )
