"""Verified, bounded local WAV duration and pitch transformation.

The whole source snapshot is verified before synthesis or output publication.
This uses the strict canonical integer-PCM transport, not a streaming
time-scale algorithm. No external files, network or model inference.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import os
import numpy as np

from .contracts import AudioSpec, AudioLayout, make_buffer
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from .hashing import strict_sha256
from .wave_reader import WavPacketReader
from .waveio import encode_wav_bytes
from .dither import QuantizationSettings
from .time_scale import TimeScaleSettings, stretch_audio, target_frames_for_ratio
from .pitch_shift import PitchShiftSettings, pitch_shift

# Peak total memory includes the immutable reader snapshot, DSP work arrays
# and full integer PCM encoding. A stricter input/output cap than the general
# AudioBuffer contract keeps this offline wrapper useful on ordinary machines.
MAX_TRANSFORM_PCM_BYTES = 24 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class WavTransformConfig:
    ratio: float | None = None
    target_frames: int | None = None
    semitones: float | None = None
    time_scale: TimeScaleSettings = TimeScaleSettings()
    lobes: int = 24
    output_bits: int | None = None
    dither_seed: int | None = None
    packet_frames: int = 8192

    def __post_init__(self):
        if sum(v is not None for v in (self.ratio, self.target_frames, self.semitones)) != 1:
            raise AudioContractError("Specify exactly one duration ratio, frame count or pitch shift")
        if not isinstance(self.time_scale, TimeScaleSettings):
            raise AudioContractError("Expected TimeScaleSettings")
        if self.ratio is not None:
            target_frames_for_ratio(2, self.ratio)
        if self.target_frames is not None and (
            type(self.target_frames) is not int or not 1 <= self.target_frames <= 2**31-1
        ):
            raise AudioContractError("Invalid target frame count")
        if type(self.lobes) is not int or not 4 <= self.lobes <= 128:
            raise AudioContractError("Invalid pitch resampling lobe count")
        if self.semitones is not None:
            self.pitch_settings()
        if self.output_bits is not None and (
            type(self.output_bits) is not int or self.output_bits not in (8, 16, 24, 32)
        ):
            raise AudioFormatError("Invalid integer WAV output depth")
        if self.dither_seed is not None:
            QuantizationSettings(dither=True, seed=self.dither_seed)
        if type(self.packet_frames) is not int or not 1 <= self.packet_frames <= 65536:
            raise AudioContractError("Invalid transform reader packet length")

    def pitch_settings(self):
        from dataclasses import replace
        return PitchShiftSettings(
            self.semitones,
            time_scale=replace(self.time_scale, peak_ceiling=None),
            lobes=self.lobes,
            peak_ceiling=self.time_scale.peak_ceiling,
        )

    def output_frames(self, source_frames):
        if self.semitones is not None:
            return source_frames
        if self.ratio is not None:
            return target_frames_for_ratio(source_frames, self.ratio)
        return self.target_frames


@dataclass(frozen=True, slots=True)
class WavTransformReport:
    operation: str
    source_sha256: str
    output_sha256: str
    source_frames: int
    output_frames: int
    sample_rate: int
    channels: int
    source_bits: int
    output_bits: int
    output_bytes: int
    dither_seed: int | None
    transform: dict
    protocol: str = "sonumio-verified-offline-wav-transform-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "source_verified_before_publication": True,
            "model_inference": False,
            "perceptual_quality_established": False,
            "streaming_transform": False,
        }


def _distinct_paths(source, target):
    if source.resolve() == target.resolve():
        raise AudioFormatError("Source and destination must differ")
    if target.exists():
        try:
            if os.path.samefile(source, target):
                raise AudioFormatError("Source and destination share a file identity")
        except FileNotFoundError:
            pass


def _preflight(info, config):
    layout = AudioLayout.for_channels(info.channels)
    source_spec = AudioSpec(info.sample_rate, layout, info.frames)
    target_frames = config.output_frames(info.frames)
    target_spec = AudioSpec(info.sample_rate, layout, target_frames)
    if target_frames * 2 < info.frames or target_frames > info.frames * 2:
        raise AudioContractError("Target duration must lie between half and twice the source")
    if max(source_spec.byte_count, target_spec.byte_count) > MAX_TRANSFORM_PCM_BYTES:
        raise AudioLimitError("Offline WAV transform exceeds PCM byte budget")
    return layout


def transform_wav_file(source, destination, *, config,
                       expected_source_sha256=None, expected_output_sha256=None,
                       overwrite=False, create_parent=False):
    """Read one verified snapshot, transform, then publish atomically.

    Output clipping is rejected rather than hidden by the integer encoder.
    A caller may request an explicit peak ceiling for attenuation.
    Source and destination aliases (including existing hard links) fail.
    Like other CLI utilities this assumes a trusted local parent directory,
    not hostile concurrent filesystem writers.
    """
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(config, WavTransformConfig):
        raise AudioContractError("Expected WavTransformConfig")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Publication options must be booleans")
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != ".wav":
        raise AudioFormatError("Transform destination must end in .wav")
    _distinct_paths(source, destination)
    with WavPacketReader(source, packet_frames=config.packet_frames,
                         expected_sha256=expected_source_sha256) as reader:
        info = reader.info
        layout = _preflight(info, config)
        samples = np.empty((info.channels, info.frames), dtype=np.float32)
        position = 0
        for packet in reader:
            if packet.start_frame != position:
                raise AudioIntegrityError("Unexpected verified reader packet offset")
            samples[:, position:packet.stop_frame] = packet.samples
            position = packet.stop_frame
        digest = reader.verified_sha256
        if digest is None or position != info.frames:
            raise AudioIntegrityError("Incomplete verified WAV snapshot")
    audio = make_buffer(samples, info.sample_rate, layout=layout,
                        source_note="verified canonical WAV snapshot")
    del samples
    if config.semitones is not None:
        result = pitch_shift(audio, config.pitch_settings())
        operation = "pitch"
    else:
        result = stretch_audio(audio, config.output_frames(info.frames),
                               settings=config.time_scale)
        operation = "duration"
    if np.max(np.abs(result.audio.samples)) > 1:
        raise AudioContractError("Transformed PCM exceeds full scale; specify a peak ceiling")
    bits = info.bits_per_sample if config.output_bits is None else config.output_bits
    quantization = QuantizationSettings(
        bits=bits, dither=config.dither_seed is not None, seed=config.dither_seed,
    )
    wav = encode_wav_bytes(result.audio, settings=quantization)
    _distinct_paths(source, destination)
    published = publish_generated_wav(
        destination, wav, overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return WavTransformReport(
        operation, digest, published.wav_sha256, info.frames, result.audio.frames,
        info.sample_rate, info.channels, info.bits_per_sample, bits,
        published.bytes_written, config.dither_seed, result.report.as_dict(),
    )
