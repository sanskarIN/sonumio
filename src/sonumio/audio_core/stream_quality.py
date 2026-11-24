"""Constant-memory signal-health inspection for verified canonical PCM WAVs.

A packet accumulator is independent of storage and neural models. A file
inspection is *not* returned until WavPacketReader authenticates the entire
source, including RIFF padding and an optional caller-supplied SHA256. The
measurements are sample-level diagnostics, not audible quality or prompt
alignment scores. Numeric summation order can differ slightly from the
in-memory implementation when packet partitioning changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from .contracts import MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .quality import QualityLimits, QualityReport, _quality_report_from_metrics
from .quality_timeline import QualityTimeline, StreamingQualityTimeline
from .frame_transitions import (
    FrameTransitionReport, StreamingFrameTransitionAccumulator,
)
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
        self._activity_floor = 10 ** (limits.activity_floor_dbfs / 20.0)
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
        self._active += int(np.count_nonzero(
            np.max(absolute, axis=0) > self._activity_floor
        ))
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
class PcmRailReport:
    """Exact decoded integer endpoint counts, not inferred distortion."""

    positive_samples: int
    negative_samples: int
    total_samples: int
    max_rail_fraction: float | None = None

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (
            self.positive_samples, self.negative_samples, self.total_samples
        )):
            raise AudioContractError("Invalid PCM rail count")
        if (self.total_samples == 0
                or self.positive_samples + self.negative_samples > self.total_samples):
            raise AudioContractError("PCM rail counts exceed decoded samples")
        if self.max_rail_fraction is not None:
            finite_float(
                self.max_rail_fraction, "max_pcm_rail_fraction",
                minimum=0.0, maximum=1.0,
            )

    @property
    def rail_fraction(self) -> float:
        return (self.positive_samples + self.negative_samples) / self.total_samples

    @property
    def passed(self) -> bool:
        return (self.max_rail_fraction is None
                or self.rail_fraction <= self.max_rail_fraction)

    def as_dict(self) -> dict:
        return {
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "total_samples": self.total_samples,
            "rail_fraction": self.rail_fraction,
            "max_rail_fraction": self.max_rail_fraction,
            "passed": self.passed,
            "possible_full_scale_saturation": self.rail_fraction > 0,
            "audible_clipping_proven": False,
        }


@dataclass(frozen=True, slots=True)
class VerifiedWavQuality:
    info: WavPacketInfo
    source_sha256: str
    signal_health: QualityReport
    timeline: QualityTimeline | None = None
    pcm_rails: PcmRailReport | None = None
    limits: QualityLimits = QualityLimits()
    packet_frames: int = 8192
    expected_sha256_required: bool = False
    max_pcm_rail_fraction: float | None = None
    frame_continuity: FrameTransitionReport | None = None

    @property
    def inspection_policy_passed(self) -> bool:
        """Combined signal-health and optional exact PCM rail policy outcome."""
        return (
            self.signal_health.passed
            and (self.pcm_rails is None or self.pcm_rails.passed)
            and (self.frame_continuity is None or self.frame_continuity.passed)
        )

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
            "inspection_policy_passed": self.inspection_policy_passed,
            "inspection_protocol": "sonumio-verified-wav-inspection-v1",
            "inspection_controls": {
                "packet_frames": self.packet_frames,
                "expected_sha256_required": self.expected_sha256_required,
                "timeline_enabled": self.timeline is not None,
                "continuity_enabled": self.frame_continuity is not None,
                "max_adjacent_step": (
                    self.frame_continuity.max_allowed_step
                    if self.frame_continuity is not None else None
                ),
                "quality_limits": asdict(self.limits),
                "max_pcm_rail_fraction": self.max_pcm_rail_fraction,
            },
            "signal_health": self.signal_health.as_dict(),
            "model_quality_measured": False,
            "recording_rights_verified": False,
        }
        if self.timeline is not None:
            result["timeline"] = self.timeline.as_dict()
        if self.pcm_rails is not None:
            result["pcm_rails"] = self.pcm_rails.as_dict()
        if self.frame_continuity is not None:
            result["frame_continuity"] = self.frame_continuity.as_dict()
        return result


def inspect_wav_quality(
    path: str | Path, *, packet_frames: int = 8192,
    expected_sha256: str | None = None,
    limits: QualityLimits = QualityLimits(),
    include_timeline: bool = False,
    max_pcm_rail_fraction: float | None = None,
    include_continuity: bool = False,
    max_adjacent_step: float | None = None,
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
    if type(include_continuity) is not bool:
        raise AudioContractError("include_continuity must be boolean")
    if max_adjacent_step is not None:
        max_adjacent_step = finite_float(
            max_adjacent_step, "max_adjacent_step",
            minimum=0.0, maximum=2.0,
        )
    if max_pcm_rail_fraction is not None:
        finite_float(
            max_pcm_rail_fraction, "max_pcm_rail_fraction",
            minimum=0.0, maximum=1.0,
        )
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
        transition_meter = (
            StreamingFrameTransitionAccumulator(
                reader.info.sample_rate, reader.info.channels,
                expected_frames=reader.info.frames,
                max_allowed_step=max_adjacent_step,
            )
            if include_continuity or max_adjacent_step is not None else None
        )
        for packet in reader:
            accumulator.update(packet)
            if timeline is not None:
                timeline.update(packet)
            if transition_meter is not None:
                transition_meter.update(packet)
        if reader.state != "verified" or reader.verified_sha256 is None:
            raise AudioIntegrityError("Incomplete verified WAV inspection")
        health = accumulator.finalize()
        timed = timeline.finalize() if timeline is not None else None
        continuity = (
            transition_meter.finalize() if transition_meter is not None else None
        )
        rail_counts = reader.verified_pcm_rail_counts
        if rail_counts is None:
            raise AudioIntegrityError("Unverified WAV PCM endpoints")
        rails = PcmRailReport(
            positive_samples=rail_counts[0],
            negative_samples=rail_counts[1],
            total_samples=reader.info.frames * reader.info.channels,
            max_rail_fraction=max_pcm_rail_fraction,
        )
        return VerifiedWavQuality(
            info=reader.info,
            source_sha256=reader.verified_sha256,
            signal_health=health,
            timeline=timed,
            pcm_rails=rails,
            limits=limits,
            packet_frames=packet_frames,
            expected_sha256_required=expected_sha256 is not None,
            max_pcm_rail_fraction=max_pcm_rail_fraction,
            frame_continuity=continuity,
        )
