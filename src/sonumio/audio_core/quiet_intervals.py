"""Packet-invariant quiet intervals on the original decoded PCM clock.

Quiet means every channel is at or below the selected activity floor.
Intentional pauses can satisfy this definition; it is not a dropout detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .contracts import MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .wave_stream import MAX_STREAM_SECONDS


@dataclass(frozen=True, slots=True)
class QuietIntervalReport:
    sample_rate: int
    channels: int
    frames: int
    activity_floor_dbfs: float
    quiet_frames: int
    quiet_runs: int
    leading_quiet_frames: int
    trailing_quiet_frames: int
    internal_quiet_runs: int
    longest_internal_quiet_frames: int
    longest_internal_start: int | None
    max_internal_quiet_frames: int | None
    protocol: str = "sonumio-quiet-intervals-v1"

    @property
    def passed(self) -> bool:
        return (self.max_internal_quiet_frames is None
                or self.longest_internal_quiet_frames <= self.max_internal_quiet_frames)

    def as_dict(self) -> dict:
        report = asdict(self)
        report["quiet_fraction"] = self.quiet_frames / self.frames
        report["longest_internal_stop"] = (
            None if self.longest_internal_start is None
            else self.longest_internal_start + self.longest_internal_quiet_frames
        )
        report["longest_internal_seconds"] = (
            self.longest_internal_quiet_frames / self.sample_rate
        )
        report["passed"] = self.passed
        report["audible_dropout_proven"] = False
        return report


def validate_internal_quiet_budget(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise AudioContractError("max_internal_quiet_frames must be a nonnegative integer")


class StreamingQuietIntervalAccumulator:
    """Retain one open run and compact summaries, not an interval timeline.

    Leading and trailing quiet are excluded from the internal-run budget.
    Equal-length internal runs select the earliest start. All coordinates
    are zero-based, with exclusive stop frames. Empty streams are rejected.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        expected_frames: int | None = None,
        activity_floor_dbfs: float = -70,
        max_internal_quiet_frames: int | None = None,
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid quiet-interval channel count")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= sample_rate * MAX_STREAM_SECONDS
        ):
            raise AudioLimitError("Invalid quiet-interval expected frames")
        validate_internal_quiet_budget(max_internal_quiet_frames)
        self.activity_floor_dbfs = finite_float(
            activity_floor_dbfs, "activity_floor_dbfs", minimum=-160, maximum=0,
        )
        self._floor = 10 ** (self.activity_floor_dbfs / 20.0)
        self.sample_rate = sample_rate
        self.channels = channels
        self.expected_frames = expected_frames
        self.max_internal_quiet_frames = max_internal_quiet_frames
        self.frames = 0
        self._quiet = 0
        self._runs = 0
        self._leading = 0
        self._internal_runs = 0
        self._longest = 0
        self._longest_start: int | None = None
        self._pending: int | None = None
        self._finished = False

    def _record_closed(self, starts: np.ndarray, stops: np.ndarray) -> None:
        if not starts.size:
            return
        self._runs += int(starts.size)
        if starts[0] == 0:
            self._leading = int(stops[0])
            starts, stops = starts[1:], stops[1:]
        if not starts.size:
            return
        self._internal_runs += int(starts.size)
        lengths = stops - starts
        index = int(np.argmax(lengths))
        length = int(lengths[index])
        if length > self._longest:
            self._longest = length
            self._longest_start = int(starts[index])

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Cannot update finalized quiet intervals")
        if not isinstance(packet, FramePacket) or (
            packet.sample_rate != self.sample_rate
            or packet.channels != self.channels
            or packet.start_frame != self.frames
        ):
            raise AudioContractError("Quiet intervals require consecutive clock-matched packets")
        total = packet.stop_frame
        if total > self.sample_rate * MAX_STREAM_SECONDS or (
            self.expected_frames is not None and total > self.expected_frames
        ):
            raise AudioLimitError("Quiet intervals exceed declared duration")
        # Match the existing signal-health activity definition exactly.
        quiet = np.max(np.abs(packet.samples.astype(np.float64)), axis=0) <= self._floor
        self._quiet += int(np.count_nonzero(quiet))
        # False sentinels bracket runs in this packet. Reconnect the prior
        # open run before summarizing completed intervals.
        edges = np.diff(quiet.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(edges == 1).astype(np.int64) + self.frames
        stops = np.flatnonzero(edges == -1).astype(np.int64) + self.frames
        if self._pending is not None:
            if quiet[0]:
                starts[0] = self._pending
            else:
                self._record_closed(
                    np.array([self._pending], dtype=np.int64),
                    np.array([self.frames], dtype=np.int64),
                )
        self._pending = None
        if quiet[-1]:
            self._pending = int(starts[-1])
            starts, stops = starts[:-1], stops[:-1]
        self._record_closed(starts, stops)
        self.frames = total

    def finalize(self) -> QuietIntervalReport:
        if self._finished:
            raise AudioContractError("Quiet intervals already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete quiet-interval stream")
        self._finished = True
        trailing = 0
        if self._pending is not None:
            trailing = self.frames - self._pending
            self._runs += 1
            if self._pending == 0:
                self._leading = self.frames
        return QuietIntervalReport(
            sample_rate=self.sample_rate, channels=self.channels,
            frames=self.frames, activity_floor_dbfs=self.activity_floor_dbfs,
            quiet_frames=self._quiet, quiet_runs=self._runs,
            leading_quiet_frames=self._leading, trailing_quiet_frames=trailing,
            internal_quiet_runs=self._internal_runs,
            longest_internal_quiet_frames=self._longest,
            longest_internal_start=self._longest_start,
            max_internal_quiet_frames=self.max_internal_quiet_frames,
        )
