"""Verified streaming mastering inspection and atomic offline WAV export.

The output is measured again after integer encoding/decoding. A sample
ceiling or a floating PCM estimate alone cannot establish the encoded
file's finite-sinc peak estimate. The optional final encoding guard
reserves a conservative quantization-error margin and verifies it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import os
import numpy as np

from .contracts import AudioBuffer
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioProcessingError
from .hashing import strict_sha256
from .dither import QuantizationSettings
from .wave_reader import WavPacketReader
from .wave_snapshot import read_verified_pcm
from .waveio import encode_wav_bytes, decode_wav_bytes
from .interpolated_peak import PeakEstimateSettings, interpolation_kernels
from .gated_energy import GatedEnergySettings
from .mastering import (
    MasteringSettings, MasteringReport, SignalMeasurement, MasteringMeasurement,
    master_audio, measure_mastering,
)

ENCODED_PEAK_POLICIES = ("reject", "attenuate")


@dataclass(frozen=True, slots=True)
class MasteringWavOptions:
    mastering: MasteringSettings = MasteringSettings()
    output_bits: int | None = None
    dither_seed: int | None = None
    encoded_peak_policy: str = "attenuate"

    def __post_init__(self):
        if not isinstance(self.mastering, MasteringSettings):
            raise AudioContractError("Expected MasteringSettings")
        if self.output_bits is not None:
            QuantizationSettings(bits=self.output_bits)
        if self.dither_seed is not None:
            QuantizationSettings(dither=True, seed=self.dither_seed)
        if self.encoded_peak_policy not in ENCODED_PEAK_POLICIES:
            raise AudioContractError("Unknown encoded peak policy")


@dataclass(frozen=True, slots=True)
class VerifiedMasteringInspection:
    source_sha256: str
    source_bits: int
    source_bytes: int
    measurement: SignalMeasurement
    protocol: str = "sonumio-verified-mastering-inspection-v1"

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "source_bits": self.source_bits,
            "source_bytes": self.source_bytes,
            "source_verified": True,
            "measurement": self.measurement.as_dict(),
            "streaming_measurement": True,
        }


@dataclass(frozen=True, slots=True)
class EncodedMastering:
    wav: bytes
    measurement: SignalMeasurement
    attenuation_db: float
    quantization_peak_reserve: float
    attempts: int


@dataclass(frozen=True, slots=True)
class VerifiedMasteringExport:
    source: dict
    output_sha256: str
    output_bytes: int
    output_bits: int
    dither_seed: int | None
    mastering: MasteringReport
    encoded_measurement: SignalMeasurement
    encoding_attenuation_db: float
    quantization_peak_reserve: float
    encoding_attempts: int
    protocol: str = "sonumio-verified-mastering-export-v1"

    def as_dict(self):
        target = self.mastering.settings.target_rms_dbfs
        encoded_level = self.encoded_measurement.energy.gated_rms_dbfs
        return {
            **self.source,
            "protocol": self.protocol,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "output_bits": self.output_bits,
            "dither_seed": self.dither_seed,
            "mastering": self.mastering.as_dict(),
            "encoded_measurement": self.encoded_measurement.as_dict(),
            "encoded_target_error_db": (
                encoded_level-target if encoded_level is not None and target is not None else None
            ),
            "encoding_attenuation_db": self.encoding_attenuation_db,
            "quantization_peak_reserve": self.quantization_peak_reserve,
            "encoding_attempts": self.encoding_attempts,
            "encoded_peak_reverified": True,
            "certified_true_peak": False,
            "lufs_measured": False,
            "model_inference": False,
        }


def inspect_mastering_wav(source, *, peak=PeakEstimateSettings(),
                          energy=GatedEnergySettings(), packet_frames=8192,
                          expected_sha256=None):
    """Measure one canonical descriptor; no full PCM snapshot is needed."""
    if type(packet_frames) is not int or not 1 <= packet_frames <= 65536:
        raise AudioContractError("Invalid mastering inspection packet length")
    with WavPacketReader(Path(source), packet_frames=packet_frames,
                         expected_sha256=expected_sha256) as reader:
        info = reader.info
        accumulator = MasteringMeasurement(
            info.sample_rate, info.channels, peak=peak, energy=energy,
            expected_frames=info.frames,
        )
        for packet in reader:
            accumulator.update(packet)
        digest = reader.verified_sha256
        if digest is None:
            raise AudioIntegrityError("Incomplete verified mastering inspection")
        measurement = accumulator.finish()
    return VerifiedMasteringInspection(
        digest, info.bits_per_sample, info.file_bytes, measurement,
    )


def quantization_peak_reserve(quantization, peak):
    """Conservative propagated PCM error bound for this finite-sinc kernel.

    Without dither the positive integer rail can introduce a full LSB;
    with TPDF, dither plus rounding can introduce 1.5 LSB. A small extra
    float32 reserve covers conversion and gain-rounding error. The
    maximum absolute phase-kernel sum bounds interpolation amplification
    of the per-sample error. Verification remains authoritative.
    """
    if not isinstance(quantization, QuantizationSettings) or not isinstance(peak, PeakEstimateSettings):
        raise AudioContractError("Invalid quantization peak-reserve configuration")
    if quantization.headroom_db != 0:
        raise AudioContractError("Encoding guard requires unity quantization headroom")
    lsb = 1/(1 << (quantization.bits-1))
    bound = (1.5 if quantization.dither else 1.)*lsb+2*np.finfo(np.float32).eps
    multiplier = max(1., *(float(np.abs(k).sum()) for k in interpolation_kernels(peak)))
    return bound*multiplier


def _encode_and_measure(audio, quantization, settings):
    wav = encode_wav_bytes(audio, settings=quantization)
    decoded = decode_wav_bytes(wav, source_note="verified encoded mastering PCM")
    measurement = measure_mastering(
        decoded, peak=settings.peak, energy=settings.energy,
        packet_frames=settings.packet_frames,
    )
    if decoded.spec != audio.spec:
        # WAV does not preserve custom labels. Geometry, not labels,
        # determines compatibility of the decoded canonical channel layout.
        if (decoded.sample_rate != audio.sample_rate or decoded.frames != audio.frames
                or decoded.channels != audio.channels):
            raise AudioProcessingError("Encoded mastering WAV geometry changed")
    return wav, measurement


def encode_mastered_audio(audio, *, quantization, settings, policy="attenuate"):
    """Encode and verify the estimated peak of the actual integer PCM.

    A failed first estimate never publishes. If attenuation is selected,
    a second attempt uses a conservative kernel-amplified reserve, then
    measures the actual decoded output again. An impossible low-bit-depth
    ceiling fails closed, as does an unsuccessful second attempt.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, MasteringSettings):
        raise AudioContractError("Expected AudioBuffer and MasteringSettings")
    if not isinstance(quantization, QuantizationSettings):
        raise AudioContractError("Expected QuantizationSettings")
    if policy not in ENCODED_PEAK_POLICIES:
        raise AudioContractError("Unknown encoded peak policy")
    if quantization.headroom_db != 0:
        raise AudioContractError("Encoding guard requires unity quantization headroom")
    if np.max(np.abs(audio.samples)) > 1:
        raise AudioContractError("Mastered PCM exceeds normalized sample range")
    wav, measured = _encode_and_measure(audio, quantization, settings)
    if measured.peak.estimated_peak <= settings.ceiling:
        return EncodedMastering(wav, measured, 0., 0., 1)
    if policy == "reject":
        raise AudioProcessingError("Encoded WAV exceeds finite-sinc estimate ceiling")
    reserve = quantization_peak_reserve(quantization, settings.peak)
    available = settings.ceiling-reserve
    if available <= 0:
        raise AudioProcessingError("Quantization reserve exceeds requested ceiling")
    original = measure_mastering(
        audio, peak=settings.peak, energy=settings.energy,
        packet_frames=settings.packet_frames,
    )
    if original.peak.estimated_peak <= 0:
        # At very low ceilings, explicit dither on digital silence may
        # itself exceed the estimated peak budget. Never erase dither
        # silently or claim the requested policy was satisfied.
        raise AudioProcessingError("Encoded silence exceeds requested ceiling")
    gain = min(1., available/original.peak.estimated_peak)
    if gain >= 1:
        raise AudioProcessingError("Encoded peak guard cannot derive attenuation")
    scaled = audio.copy_with((audio.samples.astype(np.float64)*gain).astype(np.float32),
                             source_note="quantization-aware peak guard")
    wav, measured = _encode_and_measure(scaled, quantization, settings)
    if measured.peak.estimated_peak > settings.ceiling:
        raise AudioProcessingError("Encoded WAV still exceeds estimate after attenuation")
    return EncodedMastering(wav, measured, 20*math.log10(gain), reserve, 2)


def _distinct(source, destination):
    if source.resolve() == destination.resolve():
        raise AudioFormatError("Mastering source and destination must differ")
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                raise AudioFormatError("Mastering source and destination share a file identity")
        except FileNotFoundError:
            pass


def master_wav_file(source, destination, *, options=MasteringWavOptions(),
                    expected_sha256=None, expected_output_sha256=None,
                    overwrite=False, create_parent=False):
    """Master one verified offline snapshot and publish encoded bytes atomically."""
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(options, MasteringWavOptions):
        raise AudioContractError("Expected MasteringWavOptions")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Mastering publication flags must be Boolean")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != ".wav":
        raise AudioFormatError("Mastering destination must end in .wav")
    _distinct(source, destination)
    snapshot = read_verified_pcm(
        source, expected_sha256=expected_sha256,
        packet_frames=options.mastering.packet_frames,
    )
    result = master_audio(snapshot.audio, options.mastering)
    bits = snapshot.info.bits_per_sample if options.output_bits is None else options.output_bits
    quantization = QuantizationSettings(
        bits=bits, dither=options.dither_seed is not None, seed=options.dither_seed,
    )
    encoded = encode_mastered_audio(
        result.audio, quantization=quantization,
        settings=options.mastering, policy=options.encoded_peak_policy,
    )
    _distinct(source, destination)
    publication = publish_generated_wav(
        destination, encoded.wav, overwrite=overwrite,
        create_parent=create_parent, expected_sha256=expected_output_sha256,
    )
    return VerifiedMasteringExport(
        snapshot.as_dict(), publication.wav_sha256, publication.bytes_written,
        bits, options.dither_seed, result.report, encoded.measurement,
        encoded.attenuation_db, encoded.quantization_peak_reserve, encoded.attempts,
    )
