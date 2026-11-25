"""Verified, bounded local WAV sample-clock conversion.

The reader hashes every original source byte, while the writer retains an
unpublished scratch file. Only a completely verified source and exact output
frame count permit atomic publication. This is not a realtime resampler.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path

import numpy as np

from .dither import QuantizationSettings
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .hashing import strict_sha256
from .resample import ResampleSettings
from .stream_resample import StreamingSincResampler, MAX_RESAMPLE_INPUT_FRAMES
from .streaming import FramePacket
from .wave_reader import WavPacketReader
from .wave_stream import WavPacketSpec, WavPacketWriter


@dataclass(frozen=True, slots=True)
class WavResampleResult:
    source: Path
    destination: Path
    source_sha256: str
    output_sha256: str
    source_rate: int
    target_rate: int
    source_frames: int
    output_frames: int
    channels: int
    source_bits: int
    output_bits: int
    output_bytes: int
    peak_before_clipping: float
    clipped_samples: int
    dither_seed: int | None
    settings: ResampleSettings
    protocol: str = "sonumio-stream-sinc-v1"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["source"] = str(self.source)
        result["destination"] = str(self.destination)
        result["duration_difference_seconds"] = (
            self.output_frames / self.target_rate - self.source_frames / self.source_rate
        )
        result["neural_training_data_modified"] = False
        result["perceptual_audio_quality_measured"] = False
        return result


def resample_wav_file(
    source: str | Path, destination: str | Path, *,
    settings: ResampleSettings, packet_frames: int = 4096,
    output_bits: int | None = None, dither_seed: int | None = None,
    clip_policy: str = "reject", expected_source_sha256: str | None = None,
    expected_output_sha256: str | None = None, overwrite: bool = False,
    create_parent: bool = False,
) -> WavResampleResult:
    """Convert canonical integer PCM using one verified streaming read.

    Sinc interpolation may overshoot normalized PCM. Reject by default;
    hard clipping must be requested explicitly. The digest gate precedes
    final publication. Same-path and existing hard-link aliases are denied.
    """
    if not isinstance(settings, ResampleSettings) or settings.method != "sinc":
        raise AudioContractError("WAV resampling requires sinc settings")
    if type(packet_frames) is not int or not 1 <= packet_frames <= MAX_RESAMPLE_INPUT_FRAMES:
        raise AudioContractError("Invalid WAV resampling packet size")
    if clip_policy not in ("reject", "hard"):
        raise AudioContractError("Unknown WAV resampling clip policy")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("WAV resampling path options must be boolean")
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    src, dst = Path(source), Path(destination)
    if src.resolve(strict=False) == dst.resolve(strict=False):
        raise AudioContractError("Input and output WAV paths must differ")
    if src.is_file() and dst.is_file() and os.path.samefile(src, dst):
        raise AudioContractError("Input and output WAV files must differ")
    peak = 0.0
    clipped = 0
    with WavPacketReader(src, packet_frames=packet_frames,
                         expected_sha256=expected_source_sha256) as reader:
        info = reader.info
        converter = StreamingSincResampler(
            info.sample_rate, info.channels, info.frames, settings,
        )
        quantization = QuantizationSettings(
            bits=info.bits_per_sample if output_bits is None else output_bits,
            dither=dither_seed is not None, seed=dither_seed,
        )
        spec = WavPacketSpec(settings.target_rate, info.channels,
                             converter.expected_output_frames, quantization)
        with WavPacketWriter(dst, spec, overwrite=overwrite,
                             create_parent=create_parent) as writer:
            def write(packets: tuple[FramePacket, ...]) -> None:
                nonlocal peak, clipped
                for packet in packets:
                    maximum = float(np.max(np.abs(packet.samples)))
                    peak = max(peak, maximum)
                    if maximum > 1:
                        if clip_policy == "reject":
                            raise AudioLimitError("Resampling exceeds normalized PCM range")
                        clipped += int(np.count_nonzero(np.abs(packet.samples) > 1))
                        packet = FramePacket(packet.sample_rate, packet.start_frame,
                                             np.clip(packet.samples, -1, 1))
                    writer.push(packet)

            for packet in reader:
                write(converter.push(packet))
            source_hash = reader.verified_sha256
            if source_hash is None:
                raise AudioIntegrityError("WAV resampling source verification incomplete")
            write(converter.finish())
            if writer.frames_written != converter.expected_output_frames:
                raise AudioIntegrityError("WAV resampling produced incorrect output frame count")
            result = writer.finish(expected_sha256=expected_output_sha256)
    return WavResampleResult(
        src, result.path, source_hash, result.wav_sha256, info.sample_rate,
        result.sample_rate, info.frames, result.frames, result.channels,
        info.bits_per_sample, result.bits_per_sample, result.bytes_written,
        peak, clipped, dither_seed, settings,
    )
