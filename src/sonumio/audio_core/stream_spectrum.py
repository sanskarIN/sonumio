"""Verified, bounded-memory, blockwise PCM spectral energy diagnostics.

This is a rectangular-window periodogram aggregate, not a perceptual
quality score, source classifier, or neural audio model. Each decoded sample
belongs to exactly one FFT block; the final incomplete block is zero padded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .contracts import MAX_CHANNELS, check_rate
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .wave_reader import WavPacketInfo, WavPacketReader
from .wave_stream import MAX_STREAM_SECONDS


MIN_FFT_FRAMES = 64
MAX_FFT_FRAMES = 8192
# Limit scratch arrays independently of incoming packet and FFT geometry.
FFT_BATCH_FRAMES = 65536


@dataclass(frozen=True, slots=True)
class SpectrumConfig:
    fft_frames: int = 1024

    def __post_init__(self) -> None:
        n = self.fft_frames
        if (type(n) is not int or not MIN_FFT_FRAMES <= n <= MAX_FFT_FRAMES
                or n & (n - 1)):
            raise AudioContractError("fft_frames must be a power of two from 64 to 8192")


@dataclass(frozen=True, slots=True)
class ChannelSpectrum:
    channel: int
    mean_square: float
    dominant_bin: int | None
    dominant_frequency_hz: float | None
    centroid_hz: float | None
    dc_bin_fraction: float | None
    nyquist_bin_fraction: float | None
    bin_mean_square: tuple[float, ...]

    def as_dict(self, *, include_bins: bool = False) -> dict:
        result = {
            "channel": self.channel,
            "mean_square": self.mean_square,
            "dominant_bin": self.dominant_bin,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "centroid_hz": self.centroid_hz,
            "dc_bin_fraction": self.dc_bin_fraction,
            "nyquist_bin_fraction": self.nyquist_bin_fraction,
        }
        if include_bins:
            result["bin_mean_square"] = list(self.bin_mean_square)
        return result


@dataclass(frozen=True, slots=True)
class SpectrumReport:
    sample_rate: int
    channels: int
    frames: int
    fft_frames: int
    complete_blocks: int
    padded_tail_frames: int
    channel_spectra: tuple[ChannelSpectrum, ...]
    protocol: str = "sonumio-block-spectrum-v1"

    @property
    def bin_width_hz(self) -> float:
        return self.sample_rate / self.fft_frames

    @property
    def blocks(self) -> int:
        return self.complete_blocks + int(self.padded_tail_frames > 0)

    def as_dict(self, *, include_bins: bool = False) -> dict:
        if type(include_bins) is not bool:
            raise AudioContractError("include_bins must be boolean")
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "fft_frames": self.fft_frames,
            "bin_width_hz": self.bin_width_hz,
            "bin_count": self.fft_frames // 2 + 1,
            "blocks": self.blocks,
            "complete_blocks": self.complete_blocks,
            "padded_tail_frames": self.padded_tail_frames,
            "window": "rectangular",
            "overlap_frames": 0,
            "normalization": "one-sided-bin-energy-per-original-frame",
            "channel_spectra": [
                channel.as_dict(include_bins=include_bins)
                for channel in self.channel_spectra
            ],
            "perceptual_audio_quality_measured": False,
            "source_category_inferred": False,
        }


class StreamingSpectrumAccumulator:
    """One-sided aggregate energy with Parseval-consistent normalization.

    For a block of length N, each rFFT bin contributes |X[k]|²/N,
    doubled except at DC and Nyquist. Summing across blocks then dividing
    by the original source frame count gives mean-square contributions.
    Padding the final block never changes the denominator. The sum over
    bins equals time-domain mean square, up to floating-point error.
    """

    def __init__(
        self, sample_rate: int, channels: int, *,
        config: SpectrumConfig = SpectrumConfig(),
        expected_frames: int | None = None,
    ) -> None:
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid spectrum channel count")
        if not isinstance(config, SpectrumConfig):
            raise AudioContractError("Expected SpectrumConfig")
        if expected_frames is not None and (
            type(expected_frames) is not int
            or not 1 <= expected_frames <= sample_rate * MAX_STREAM_SECONDS
        ):
            raise AudioLimitError("Invalid spectrum expected frames")
        self.sample_rate = sample_rate
        self.channels = channels
        self.config = config
        self.expected_frames = expected_frames
        self.frames = 0
        self._complete = 0
        n = config.fft_frames
        self._pending = np.empty((channels, n), dtype=np.float64)
        self._pending_frames = 0
        self._energy = np.zeros((channels, n // 2 + 1), dtype=np.float64)
        self._weights = np.full(n // 2 + 1, 2.0 / n, dtype=np.float64)
        self._weights[0] = self._weights[-1] = 1.0 / n
        self._finished = False

    def _take_blocks(self, blocks: np.ndarray) -> None:
        # Input axes: channels, blocks, samples. All FFTs use a fixed
        # original-source block grid, independent of packet boundaries.
        transformed = np.fft.rfft(blocks, axis=-1)
        squared = transformed.real ** 2 + transformed.imag ** 2
        self._energy += np.sum(squared, axis=1, dtype=np.float64) * self._weights

    def update(self, packet: FramePacket) -> None:
        if self._finished:
            raise AudioContractError("Cannot update finalized spectrum")
        if not isinstance(packet, FramePacket) or (
            packet.sample_rate != self.sample_rate
            or packet.channels != self.channels
            or packet.start_frame != self.frames
        ):
            raise AudioContractError("Spectrum requires consecutive clock-matched packets")
        total = packet.stop_frame
        if total > self.sample_rate * MAX_STREAM_SECONDS or (
            self.expected_frames is not None and total > self.expected_frames
        ):
            raise AudioLimitError("Spectrum exceeds declared duration")
        n = self.config.fft_frames
        offset = 0
        data = packet.samples
        if self._pending_frames:
            count = min(n - self._pending_frames, packet.frames)
            end = self._pending_frames + count
            self._pending[:, self._pending_frames:end] = data[:, :count]
            self._pending_frames = end
            offset = count
            if end == n:
                self._take_blocks(self._pending[:, None, :])
                self._complete += 1
                self._pending_frames = 0
        # Batch full windows to avoid one Python FFT call per short block.
        # The fixed batch-frame cap bounds FFT scratch even on 1M-frame
        # FramePackets provided by non-file callers.
        batch_blocks = FFT_BATCH_FRAMES // n
        while packet.frames - offset >= n:
            count = min((packet.frames - offset) // n, batch_blocks)
            stop = offset + count * n
            block = data[:, offset:stop].astype(np.float64).reshape(
                self.channels, count, n,
            )
            self._take_blocks(block)
            self._complete += count
            offset = stop
        if offset < packet.frames:
            remaining = packet.frames - offset
            self._pending[:, :remaining] = data[:, offset:]
            self._pending_frames = remaining
        self.frames = total

    def finalize(self) -> SpectrumReport:
        if self._finished:
            raise AudioContractError("Spectrum accumulator already finalized")
        if self.frames == 0 or (
            self.expected_frames is not None and self.frames != self.expected_frames
        ):
            raise AudioIntegrityError("Incomplete spectrum stream")
        self._finished = True
        n = self.config.fft_frames
        tail = self._pending_frames
        if tail:
            self._pending[:, tail:] = 0.0
            self._take_blocks(self._pending[:, None, :])
        bin_power = self._energy / self.frames
        frequencies = np.arange(n // 2 + 1, dtype=np.float64) * (
            self.sample_rate / n
        )
        channels = []
        for i, power in enumerate(bin_power):
            energy = float(np.sum(power, dtype=np.float64))
            dominant = int(np.argmax(power)) if energy > 0 else None
            channels.append(ChannelSpectrum(
                channel=i, mean_square=energy,
                dominant_bin=dominant,
                dominant_frequency_hz=(
                    float(frequencies[dominant]) if dominant is not None else None
                ),
                centroid_hz=(
                    float(np.dot(power, frequencies) / energy) if energy > 0 else None
                ),
                dc_bin_fraction=float(power[0] / energy) if energy > 0 else None,
                nyquist_bin_fraction=float(power[-1] / energy) if energy > 0 else None,
                bin_mean_square=tuple(float(x) for x in power),
            ))
        return SpectrumReport(
            sample_rate=self.sample_rate, channels=self.channels,
            frames=self.frames, fft_frames=n, complete_blocks=self._complete,
            padded_tail_frames=tail, channel_spectra=tuple(channels),
        )


@dataclass(frozen=True, slots=True)
class VerifiedWavSpectrum:
    info: WavPacketInfo
    source_sha256: str
    expected_sha256_required: bool
    packet_frames: int
    spectrum: SpectrumReport

    def as_dict(self, *, include_bins: bool = False) -> dict:
        return {
            "source_verified": True,
            "source_sha256": self.source_sha256,
            "expected_sha256_required": self.expected_sha256_required,
            "packet_frames": self.packet_frames,
            "wav_geometry": asdict(self.info),
            "spectrum": self.spectrum.as_dict(include_bins=include_bins),
            "recording_rights_verified": False,
        }


def inspect_wav_spectrum(
    path: str | Path, *,
    config: SpectrumConfig = SpectrumConfig(),
    packet_frames: int = 8192,
    expected_sha256: str | None = None,
) -> VerifiedWavSpectrum:
    """Return a result only after the full canonical PCM file is verified."""
    if not isinstance(config, SpectrumConfig):
        raise AudioContractError("Expected SpectrumConfig")
    with WavPacketReader(
        path, packet_frames=packet_frames, expected_sha256=expected_sha256,
    ) as reader:
        meter = StreamingSpectrumAccumulator(
            reader.info.sample_rate, reader.info.channels,
            expected_frames=reader.info.frames, config=config,
        )
        for packet in reader:
            meter.update(packet)
        if reader.state != "verified" or reader.verified_sha256 is None:
            raise AudioIntegrityError("Incomplete verified WAV spectrum")
        return VerifiedWavSpectrum(
            reader.info, reader.verified_sha256,
            expected_sha256 is not None, packet_frames, meter.finalize(),
        )
