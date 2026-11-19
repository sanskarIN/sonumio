"""Two-pass, verified, bounded-memory local WAV postprocessing.

A first pass measures PCM peak/RMS and verifies the complete source digest.
A second pass verifies the same digest while optionally trimming quiet
edges and applying deterministic gain, per-channel DC removal and a sample
peak ceiling. WavPacketWriter does not publish scratch output until the
second source read completes successfully.

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
    remove_dc: bool = False
    trim_silence_dbfs: float | None = None
    fade_in_frames: int = 0
    fade_out_frames: int = 0
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
        if type(self.remove_dc) is not bool:
            raise AudioContractError("remove_dc must be boolean")
        if self.trim_silence_dbfs is not None:
            finite_float(self.trim_silence_dbfs, "WAV trim threshold dBFS",
                         minimum=-120, maximum=-1)
        # Exact integer frame counts avoid ms/seconds rounding ambiguity.
        # Each fade spans at most ten seconds at the maximum supported rate.
        for name, value in (("fade_in_frames", self.fade_in_frames),
                            ("fade_out_frames", self.fade_out_frames)):
            if type(value) is not int or not 0 <= value <= 1_920_000:
                raise AudioContractError(name + " must be an integer in [0, 1920000]")
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
    dc_removed: bool = False
    removed_dc_offsets: tuple[float, ...] = ()
    source_frames: int = 0
    retained_start_frame: int = 0
    retained_end_frame: int = 0
    trim_silence_dbfs: float | None = None
    fade_in_frames: int = 0
    fade_out_frames: int = 0


def _edge_fade_envelope(
    start: int, stop: int, total: int, fade_in: int, fade_out: int,
) -> np.ndarray:
    """Return packet-local, position-exact linear edge fade amplitudes.

    First/last fade frames have exactly zero/one endpoints when N >= 2.
    A one-frame fade mutes only that endpoint. If fades overlap, their
    attenuations multiply. This vector depends on absolute output frames,
    never on how the reader partitions audio into packets.
    """
    positions = np.arange(start, stop, dtype=np.float64)
    envelope = np.ones(stop - start, dtype=np.float64)
    if fade_in:
        if fade_in == 1:
            envelope[positions == 0] = 0.0
        else:
            envelope *= np.minimum(positions / (fade_in - 1), 1.0)
    if fade_out:
        remaining = total - 1 - positions
        if fade_out == 1:
            envelope[remaining == 0] = 0.0
        else:
            envelope *= np.minimum(remaining / (fade_out - 1), 1.0)
    return envelope


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
    expected_output_sha256: str | None = None,
    overwrite: bool = False,
    create_parent: bool = False,
) -> WavProcessingResult:
    """Verify source twice, then atomically publish bounded PCM output.

    This function never publishes a partial output. A failed hash, malformed
    source, unsafe gain, mismatched output digest, unsupported target
    filesystem, destination collision or user exception aborts and removes
    the temporary output file.
    """
    if config is None:
        config = WavProcessingConfig()
    if not isinstance(config, WavProcessingConfig):
        raise AudioContractError("Expected WavProcessingConfig")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("WAV processing file options must be boolean")
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
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
        # Trim decisions are made on original decoded samples, prior to
        # centering, gain, dither or PCM conversion. All channels must agree
        # on a single retained frame interval to preserve their alignment.
        trim_threshold = (None if config.trim_silence_dbfs is None else
                          10.0 ** (config.trim_silence_dbfs / 20.0))
        retained_start = None
        retained_end = None
        if config.remove_dc:
            # Per-channel bounded statistics: O(channels), not O(frames).
            # Every decoded PCM float32 sample is an exact multiple of
            # 2**-31 (all supported PCM bit depths <= 32). A fixed-point
            # int64 sum is *exact* and packet-partition invariant. Even at
            # 192 kHz * 3600 s, per-channel |sum| < 2**61, no overflow.
            channel_units = np.zeros(source_info.channels, dtype=np.int64)
            channel_min = np.full(source_info.channels, np.inf)
            channel_max = np.full(source_info.channels, -np.inf)
        for packet in reader:
            peak, sq = _peak_and_square(packet.samples)
            source_peak = max(source_peak, peak)
            squared_sum += sq
            if trim_threshold is not None:
                active = np.flatnonzero(np.any(
                    np.abs(packet.samples) > trim_threshold, axis=0,
                ))
                if active.size:
                    if retained_start is None:
                        retained_start = packet.start_frame + int(active[0])
                    retained_end = packet.start_frame + int(active[-1]) + 1
            if config.remove_dc:
                fixed = np.ldexp(packet.samples.astype(np.float64), 31)
                channel_units += np.sum(
                    fixed.astype(np.int64), axis=1, dtype=np.int64,
                )
                channel_min = np.minimum(channel_min, np.min(packet.samples, axis=1))
                channel_max = np.maximum(channel_max, np.max(packet.samples, axis=1))
        source_sha256 = reader.verified_sha256
    if source_sha256 is None:
        raise AudioIntegrityError("First-pass WAV verification incomplete")
    if trim_threshold is None:
        retained_start, retained_end = 0, source_info.frames
    elif retained_start is None or retained_end is None:
        # Refuse an empty result; no parent or temporary output has been
        # created at this point.
        raise AudioLimitError("No WAV frames exceed the silence trim threshold")
    retained_frames = retained_end - retained_start
    if (config.fade_in_frames > retained_frames or
            config.fade_out_frames > retained_frames):
        # Fail before creating even an output directory or scratch file.
        raise AudioLimitError("WAV fade exceeds retained output frame count")
    count = source_info.frames * source_info.channels
    source_rms = math.sqrt(squared_sum / count)
    if config.remove_dc:
        # Convert to float64 once after exact, overflow-safe accumulation.
        # This avoids NumPy floating reduction/order drift at chunk boundaries.
        dc_offsets = channel_units.astype(np.float64) / (
            source_info.frames * (1 << 31)
        )
        # Centering can increase the peak above its raw PCM magnitude.
        # A gain/ceiling decision must use the post-centering extrema.
        centered_peak = float(np.max(np.maximum(
            np.abs(channel_min - dc_offsets), np.abs(channel_max - dc_offsets),
        )))
    else:
        dc_offsets = None
        centered_peak = source_peak
    effective_gain = _effective_gain(centered_peak, config)

    bits = (source_info.bits_per_sample if config.output_bits is None
            else config.output_bits)
    spec = WavPacketSpec(
        source_info.sample_rate, source_info.channels, retained_frames,
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
                # Still consume *every* source packet to verify the full
                # original SHA, including discarded head and tail frames.
                start = max(packet.start_frame, retained_start)
                stop = min(packet.stop_frame, retained_end)
                if start >= stop:
                    continue
                selected = packet.samples[
                    :, start - packet.start_frame:stop - packet.start_frame,
                ]
                working = selected.astype(np.float64)
                if dc_offsets is not None:
                    working -= dc_offsets[:, None]
                if config.fade_in_frames or config.fade_out_frames:
                    output_start = start - retained_start
                    envelope = _edge_fade_envelope(
                        output_start, output_start + selected.shape[1],
                        retained_frames, config.fade_in_frames,
                        config.fade_out_frames,
                    )
                    working *= envelope[None, :]
                working *= effective_gain
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
                    source_info.sample_rate, start - retained_start, processed,
                ))
            if reader.verified_sha256 != source_sha256:
                raise AudioIntegrityError(
                    "Second-pass WAV source verification incomplete"
                )
            result = writer.finish(expected_sha256=expected_output_sha256)
    return WavProcessingResult(
        source=src, destination=result.path,
        source_sha256=source_sha256, output_sha256=result.wav_sha256,
        frames=result.frames, sample_rate=result.sample_rate,
        channels=result.channels, source_bits=source_info.bits_per_sample,
        output_bits=result.bits_per_sample, source_peak=source_peak,
        source_rms=source_rms,
        output_peak_before_quantization=second_peak,
        output_rms_before_quantization=math.sqrt(
            second_squared_sum / (retained_frames * source_info.channels)
        ),
        effective_gain_db=20 * math.log10(effective_gain),
        clipped_samples=clipped_samples,
        output_bytes=result.bytes_written,
        dither_applied=config.dither_seed is not None,
        dither_seed=config.dither_seed,
        dither_protocol=(STREAM_TPDF_VERSION
                         if config.dither_seed is not None else None),
        dc_removed=config.remove_dc,
        removed_dc_offsets=(tuple(float(offset) for offset in dc_offsets)
                            if dc_offsets is not None else ()),
        source_frames=source_info.frames,
        retained_start_frame=retained_start,
        retained_end_frame=retained_end,
        trim_silence_dbfs=config.trim_silence_dbfs,
        fade_in_frames=config.fade_in_frames,
        fade_out_frames=config.fade_out_frames,
    )
