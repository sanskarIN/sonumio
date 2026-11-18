"""Two-pass, verified, bounded-memory local WAV postprocessing.

A first pass measures PCM peak/RMS and verifies the complete source digest.
A second pass verifies the same digest while applying a fixed deterministic
gain and optional peak ceiling. A WavPacketWriter scratch output is not
published until the second source read completes successfully.

This module is strictly numerical postprocessing of *existing* approved
local audio. It neither generates neural audio nor validates copyright or
speaker consent. There is no network I/O, model downloading or API exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np

from .contracts import finite_float
from .dither import QuantizationSettings
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .hashing import strict_sha256
from .stream_dither import STREAM_TPDF_VERSION
from .streaming import FramePacket
from .wave_reader import WavPacketReader
from .wave_stream import (
    MAX_STREAM_PACKET_FRAMES, WavPacketSpec, WavPacketWriter,
)

MAX_PEAK_BOOST_DB = 12.0
ClipPolicy = Literal["reject", "hard"]


@dataclass(frozen=True, slots=True)
class WavProcessingConfig:
    """Explicit processing parameters; no automatic loudness decisions."""

    gain_db: float = 0.0
    peak_ceiling_dbfs: float | None = None
    allow_peak_boost: bool = False
    clip_policy: ClipPolicy = "reject"
    packet_frames: int = 8192
    output_bits: int | None = None
    dither_seed: int | None = None

    def __post_init__(self) -> None:
        finite_float(self.gain_db, "WAV gain dB", minimum=-60, maximum=24)
        if self.peak_ceiling_dbfs is not None:
            finite_float(self.peak_ceiling_dbfs, "WAV peak ceiling dBFS",
                         minimum=-60, maximum=0)
        if type(self.allow_peak_boost) is not bool:
            raise AudioContractError("allow_peak_boost must be boolean")
        if self.clip_policy not in ("reject", "hard"):
            raise AudioContractError("clip_policy must be reject or hard")
        if (type(self.packet_frames) is not int or
                not 1 <= self.packet_frames <= MAX_STREAM_PACKET_FRAMES):
            raise AudioContractError("Invalid WAV processor packet frame count")
        if self.output_bits is not None and (
                type(self.output_bits) is not int or
                self.output_bits not in (8, 16, 24, 32)):
            raise AudioContractError("Output PCM bits must be 8, 16, 24 or 32")
        if self.dither_seed is not None and (
                type(self.dither_seed) is not int or
                not 0 <= self.dither_seed < 2**32):
            raise AudioContractError("Streaming WAV dither requires 32-bit seed")


@dataclass(frozen=True, slots=True)
class WavProcessingResult:
    source: Path
    destination: Path
    source_sha256: str
    output_sha256: str
    frames: int
    sample_rate: int
    channels: int
    source_bits: int
    output_bits: int
    source_peak: float
    source_rms: float
    output_peak_before_quantization: float
    output_rms_before_quantization: float
    effective_gain_db: float
    clipped_samples: int
    output_bytes: int
    original_model_inference: bool = False
    dither_applied: bool = False
    dither_seed: int | None = None
    dither_protocol: str | None = None


def _peak_and_square(samples: np.ndarray) -> tuple[float, float]:
    """Compute deterministic float64 magnitude and sum of squared samples."""
    data = samples.astype(np.float64, copy=False)
    return float(np.max(np.abs(data))), float(np.sum(np.square(data), dtype=np.float64))


def _effective_gain(source_peak: float, cfg: WavProcessingConfig) -> float:
    gain = 10.0 ** (float(cfg.gain_db) / 20.0)
    if cfg.peak_ceiling_dbfs is None or source_peak == 0:
        return gain
    ceiling = 10.0 ** (float(cfg.peak_ceiling_dbfs) / 20.0)
    pre_peak = source_peak * gain
    correction = ceiling / pre_peak
    if correction < 1.0:
        gain *= correction
    elif cfg.allow_peak_boost:
        gain *= min(correction, 10.0 ** (MAX_PEAK_BOOST_DB / 20.0))
    return gain


def process_wav_file(
    source: str | Path, destination: str | Path, *,
    config: WavProcessingConfig | None = None,
    expected_source_sha256: str | None = None,
    overwrite: bool = False,
    create_parent: bool = False,
) -> WavProcessingResult:
    """Verify source twice, then atomically publish bounded PCM output.

    This function never publishes a partial output. A failed hash, malformed
    source, unsafe gain, unsupported target filesystem, destination collision
    or user exception aborts and removes the temporary output file.
    """
    if config is None:
        config = WavProcessingConfig()
    if not isinstance(config, WavProcessingConfig):
        raise AudioContractError("Expected WavProcessingConfig")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("WAV processing file options must be boolean")
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    src, dst = Path(source), Path(destination)
    if src.resolve(strict=False) == dst.resolve(strict=False):
        raise AudioContractError("Input and output WAV paths must differ")
    # Hard-link aliases can name the same inode despite distinct paths.
    if src.is_file() and dst.is_file() and os.path.samefile(src, dst):
        raise AudioContractError("Input and output WAV files must differ")

    # First pass: no destination created. Stats use bounded per-packet arrays,
    # never a materialized full AudioBuffer.
    with WavPacketReader(
        src, packet_frames=config.packet_frames,
        expected_sha256=expected_source_sha256,
    ) as reader:
        source_info = reader.info
        source_peak, squared_sum = 0.0, 0.0
        for packet in reader:
            peak, sq = _peak_and_square(packet.samples)
            source_peak = max(source_peak, peak)
            squared_sum += sq
        source_sha256 = reader.verified_sha256
    if source_sha256 is None:
        raise AudioIntegrityError("First-pass WAV verification incomplete")
    count = source_info.frames * source_info.channels
    source_rms = math.sqrt(squared_sum / count)
    effective_gain = _effective_gain(source_peak, config)

    bits = (source_info.bits_per_sample if config.output_bits is None
            else config.output_bits)
    spec = WavPacketSpec(
        source_info.sample_rate, source_info.channels, source_info.frames,
        QuantizationSettings(bits=bits, dither=config.dither_seed is not None,
                             seed=config.dither_seed),
    )
    second_peak, second_squared_sum, clipped_samples = 0.0, 0.0, 0
    with WavPacketReader(
        src, packet_frames=config.packet_frames,
        expected_sha256=source_sha256,
    ) as reader:
        with WavPacketWriter(
            dst, spec, overwrite=overwrite, create_parent=create_parent,
        ) as writer:
            for packet in reader:
                working = packet.samples.astype(np.float64) * effective_gain
                exceeding = np.abs(working) > 1.0
                over_count = int(np.count_nonzero(exceeding))
                if over_count and config.clip_policy == "reject":
                    raise AudioLimitError(
                        "Gain would clip WAV samples; choose peak ceiling "
                        "or explicitly request hard clipping"
                    )
                if config.clip_policy == "hard":
                    working = np.clip(working, -1.0, 1.0)
                    clipped_samples += over_count
                processed = working.astype(np.float32)
                peak, sq = _peak_and_square(processed)
                second_peak = max(second_peak, peak)
                second_squared_sum += sq
                writer.push(FramePacket(
                    source_info.sample_rate, packet.start_frame, processed,
                ))
            if reader.verified_sha256 != source_sha256:
                raise AudioIntegrityError(
                    "Second-pass WAV source verification incomplete"
                )
            result = writer.finish()
    return WavProcessingResult(
        source=src, destination=result.path,
        source_sha256=source_sha256, output_sha256=result.wav_sha256,
        frames=result.frames, sample_rate=result.sample_rate,
        channels=result.channels, source_bits=source_info.bits_per_sample,
        output_bits=result.bits_per_sample, source_peak=source_peak,
        source_rms=source_rms,
        output_peak_before_quantization=second_peak,
        output_rms_before_quantization=math.sqrt(second_squared_sum / count),
        effective_gain_db=20 * math.log10(effective_gain),
        clipped_samples=clipped_samples,
        output_bytes=result.bytes_written,
        dither_applied=config.dither_seed is not None,
        dither_seed=config.dither_seed,
        dither_protocol=(STREAM_TPDF_VERSION
                         if config.dither_seed is not None else None),
    )
