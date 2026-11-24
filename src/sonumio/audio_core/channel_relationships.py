"""Bounded pairwise channel relationships on a decoded PCM clock.

Correlation is centered Pearson correlation, not a phase measurement.
Fold-down retention describes a hypothetical equal-weight average of two
channels; it is not a prescribed surround downmix or an audible-quality score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import MAX_CHANNELS, check_rate
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .wave_stream import MAX_STREAM_SECONDS


@dataclass(frozen=True, slots=True)
class ChannelPairReport:
    first_channel: int
    second_channel: int
    centered_correlation: float | None
    equal_mix_rms: float
    reference_rms: float
    equal_mix_retention: float | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChannelRelationshipReport:
    sample_rate: int
    channels: int
    frames: int
    channel_rms: tuple[float, ...]
    channel_dc: tuple[float, ...]
    pairs: tuple[ChannelPairReport, ...]
    protocol: str = "sonumio-channel-relationships-v1"

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "channel_rms": list(self.channel_rms),
            "channel_dc": list(self.channel_dc),
            "pairs": [p.as_dict() for p in self.pairs],
            "perceptual_audio_quality_measured": False,
            "physical_channel_layout_inferred": False,
            "phase_inversion_proven": False,
        }


class StreamingChannelRelationshipAccumulator:
    """Accumulate centered covariance and direct equal-mix energy.

    Uses parallel centered-moment combination to avoid subtracting large
    squared DC terms from raw moments. Direct pair mix energies preserve
    small residuals when opposite-polarity samples nearly cancel.
    Persistent memory is O(channels squared); scratch scales with a packet.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        expected_frames: int | None = None,
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid channel-relationship channel count")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= sample_rate * MAX_STREAM_SECONDS
        ):
            raise AudioLimitError("Invalid channel-relationship expected frames")
        self.sample_rate = sample_rate
        self.channels = channels
        self.expected_frames = expected_frames
        self.frames = 0
        self._means = np.zeros(channels, dtype=np.float64)
        self._centered = np.zeros((channels, channels), dtype=np.float64)
        self._energy = np.zeros(channels, dtype=np.float64)
        self._pair_indices = tuple(
            (i, j) for i in range(channels) for j in range(i + 1, channels)
        )
        self._mixed_energy = np.zeros(len(self._pair_indices), dtype=np.float64)
        self._finished = False

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Cannot update finalized channel relationships")
        if not isinstance(packet, FramePacket) or (
            packet.sample_rate != self.sample_rate
            or packet.channels != self.channels
            or packet.start_frame != self.frames
        ):
            raise AudioContractError("Channel relationships require consecutive clock-matched packets")
        total = packet.stop_frame
        if total > self.sample_rate * MAX_STREAM_SECONDS or (
            self.expected_frames is not None and total > self.expected_frames
        ):
            raise AudioLimitError("Channel relationships exceed declared duration")
        data = packet.samples.astype(np.float64)
        n = packet.frames
        batch_means = np.mean(data, axis=1, dtype=np.float64)
        centered = data - batch_means[:, None]
        batch_centered = centered @ centered.T
        delta = batch_means - self._means
        # Chan-style combination of independent centered second moments.
        self._centered += batch_centered + np.outer(delta, delta) * (
            self.frames * n / total
        )
        self._means += delta * (n / total)
        self._energy += np.einsum("ij,ij->i", data, data)
        for k, (i, j) in enumerate(self._pair_indices):
            mix = (data[i] + data[j]) * 0.5
            self._mixed_energy[k] += float(np.dot(mix, mix))
        self.frames = total

    def finalize(self) -> ChannelRelationshipReport:
        if self._finished:
            raise AudioContractError("Channel relationships already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete channel-relationship stream")
        self._finished = True
        pairs = []
        for k, (i, j) in enumerate(self._pair_indices):
            vi = float(self._centered[i, i])
            vj = float(self._centered[j, j])
            correlation = None
            if vi > 0.0 and vj > 0.0:
                correlation = float(self._centered[i, j]) / (
                    math.sqrt(vi) * math.sqrt(vj)
                )
                correlation = min(1.0, max(-1.0, correlation))
            reference_energy = float((self._energy[i] + self._energy[j]) * 0.5)
            mixed_energy = float(self._mixed_energy[k])
            reference_rms = math.sqrt(reference_energy / self.frames)
            equal_mix_rms = math.sqrt(mixed_energy / self.frames)
            retention = (
                min(1.0, math.sqrt(mixed_energy / reference_energy))
                if reference_energy > 0.0 else None
            )
            pairs.append(ChannelPairReport(
                i, j, correlation, equal_mix_rms, reference_rms, retention,
            ))
        return ChannelRelationshipReport(
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames=self.frames,
            channel_rms=tuple(float(x) for x in np.sqrt(self._energy / self.frames)),
            channel_dc=tuple(float(x) for x in self._means),
            pairs=tuple(pairs),
        )
