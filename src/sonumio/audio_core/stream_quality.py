"""Constant-memory signal-health inspection for verified canonical PCM WAVs.

A packet accumulator is independent of storage and neural models. A file
inspection is *not* returned until WavPacketReader authenticates the entire
source, including RIFF padding and an optional caller-supplied SHA256. The
measurements are sample-level diagnostics, not audible quality or prompt
alignment scores. Numeric summation order can differ slightly from the
in-memory implementation when packet partitioning changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .contracts import MAX_CHANNELS, check_rate
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .quality import QualityLimits, QualityReport, _quality_report_from_metrics
from .quality_timeline import QualityTimeline, StreamingQualityTimeline
from .streaming import FramePacket
from .wave_reader import WavPacketInfo, WavPacketReader
from .wave_stream import MAX_STREAM_PACKET_FRAMES, MAX_STREAM_SECONDS


class StreamingQualityAccumulator:
    """Aggregate consecutive immutable packets without retaining their audio.

    Expected frame counts are enforced when supplied. A caller must not
    interpret a partial accumulator as a completed inspection: finalize()
    succeeds only after the expected number of frames has been received.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        expected_frames: int | None = None,
        limits: QualityLimits = QualityLimits(),
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid quality-inspection channel count")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= sample_rate * MAX_STREAM_SECONDS
        ):
            raise AudioLimitError("Invalid bounded quality-inspection frame count")
        if not isinstance(limits, QualityLimits):
            raise AudioContractError("Expected QualityLimits")
        self.sample_rate = sample_rate
        self.channels = channels
        self.expected_frames = expected_frames
        self.limits = limits
        self.frames = 0
        self._channel_sums = np.zeros(channels, dtype=np.float64)
        self._channel_squared_sums = np.zeros(channels, dtype=np.float64)
        self._peak = 0.0
        self._clipped = 0
        self._active = 0
        self._initial_boundary = None
        self._final_boundary = 0.0
        self._finished = False

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Cannot update a finalized quality report")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected an immutable FramePacket")
        if (packet.sample_rate != self.sample_rate
                or packet.channels != self.channels
                or packet.start_frame != self.frames):
            raise AudioContractError("Non-consecutive or mismatched quality packet")
        total = self.frames + packet.frames
        if (total > self.sample_rate * MAX_STREAM_SECONDS
                or self.expected_frames is not None
                and total > self.expected_frames):
            raise AudioLimitError("Quality packets exceed declared duration")
        # Float64 accumulation avoids large float32 reductions across long WAVs.
        data = packet.samples.astype(np.float64)
        absolute = np.abs(data)
        self._channel_sums += np.sum(data, axis=1, dtype=np.float64)
        self._channel_squared_sums += np.sum(data * data, axis=1, dtype=np.float64)
        self._peak = max(self._peak, float(np.max(absolute)))
        self._clipped += int(np.count_nonzero(absolute >= 1.0))
        floor = 10 ** (self.limits.activity_floor_dbfs / 20.0)
        self._active += int(np.count_nonzero(np.max(absolute, axis=0) > floor))
        if self._initial_boundary is None:
            self._initial_boundary = float(np.max(absolute[:, 0]))
        self._final_boundary = float(np.max(absolute[:, -1]))
        self.frames = total

    def finalize(self) -> QualityReport:
        if self._finished:
            raise AudioContractError("Quality accumulator is already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete quality-inspection frame stream")
        self._finished = True
        frames = self.frames
        rms_by_channel = np.sqrt(self._channel_squared_sums / frames)
        rms = float(np.sqrt(np.mean(self._channel_squared_sums / frames)))
        dc = float(np.max(np.abs(self._channel_sums / frames)))
        clipping = self._clipped / (self.channels * frames)
        active = self._active / frames
        jump = 0.0 if frames < 2 else max(self._initial_boundary, self._final_boundary)
        highest = float(np.max(rms_by_channel))
        imbalance = (
            0.0 if highest <= 1e-12 else
            20.0 * math.log10(highest / max(float(np.min(rms_by_channel)), 1e-12))
        )
        return _quality_report_from_metrics(
            sample_rate=self.sample_rate, frames=frames, channels=self.channels,
            peak=self._peak, rms=rms, dc=dc, clipping=clipping,
            active=active, jump=jump, imbalance=imbalance, limits=self.limits,
        )


@dataclass(frozen=True, slots=True)
class VerifiedWavQuality:
    info: WavPacketInfo
    source_sha256: str
    signal_health: QualityReport
    timeline: QualityTimeline | None = None

    def as_dict(self) -> dict:
        result = {
            "sample_rate": self.info.sample_rate,
            "channels": self.info.channels,
            "frames": self.info.frames,
            "bits_per_sample": self.info.bits_per_sample,
            "file_bytes": self.info.file_bytes,
            "data_bytes": self.info.data_bytes,
            "source_sha256": self.source_sha256,
            "source_verified": True,
            "signal_health": self.signal_health.as_dict(),
            "model_quality_measured": False,
            "recording_rights_verified": False,
        }
        if self.timeline is not None:
            result["timeline"] = self.timeline.as_dict()
        return result


def inspect_wav_quality(
    path: str | Path, *, packet_frames: int = 8192,
    expected_sha256: str | None = None,
    limits: QualityLimits = QualityLimits(),
    include_timeline: bool = False,
) -> VerifiedWavQuality:
    """Inspect canonical local integer PCM with complete-file verification.

    The file reader enforces 8 kHz..192 kHz, 1..8 channels, one hour, a
    bounded packet size, no symbolic links and a strict RIFF layout. It
    never materializes the complete audio. Failed/partial reads do not
    produce a VerifiedWavQuality or public JSON result.
    """
    if type(packet_frames) is not int or not 1 <= packet_frames <= MAX_STREAM_PACKET_FRAMES:
        raise AudioContractError("Invalid quality-inspection packet frame size")
    if not isinstance(limits, QualityLimits):
        raise AudioContractError("Expected QualityLimits")
    if type(include_timeline) is not bool:
        raise AudioContractError("include_timeline must be boolean")
    with WavPacketReader(
        path, packet_frames=packet_frames, expected_sha256=expected_sha256,
    ) as reader:
        accumulator = StreamingQualityAccumulator(
            reader.info.sample_rate, reader.info.channels,
            expected_frames=reader.info.frames, limits=limits,
        )
        timeline = (
            StreamingQualityTimeline(
                reader.info.sample_rate, reader.info.channels,
                expected_frames=reader.info.frames,
                activity_floor_dbfs=limits.activity_floor_dbfs,
            )
            if include_timeline else None
        )
        for packet in reader:
            accumulator.update(packet)
            if timeline is not None:
                timeline.update(packet)
        if reader.state != "verified" or reader.verified_sha256 is None:
            raise AudioIntegrityError("Incomplete verified WAV inspection")
        health = accumulator.finalize()
        timed = timeline.finalize() if timeline is not None else None
        return VerifiedWavQuality(reader.info, reader.verified_sha256, health, timed)
