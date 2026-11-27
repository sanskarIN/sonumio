"""Fixed-clock, duration-preserving pitch transposition via DSP.

Stretch at the source clock, then use ratio-aware sinc conversion from
a virtual clock. The integer virtual rate is reported rather than claiming
an exact real-valued pitch ratio. No learned voice or instrument model is
involved, and formants are not explicitly preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math
import numpy as np

from .contracts import AudioBuffer, finite_float, check_rate
from .errors import AudioContractError, AudioProcessingError
from .resample import ResampleSettings, sinc_resample
from .time_scale import TimeScaleSettings, stretch_audio, target_frames_for_ratio


@dataclass(frozen=True, slots=True)
class PitchShiftSettings:
    semitones: float
    time_scale: TimeScaleSettings = TimeScaleSettings(method="phase_vocoder")
    lobes: int = 24
    peak_ceiling: float | None = None

    def __post_init__(self):
        finite_float(self.semitones, "semitones", minimum=-12, maximum=12)
        if not isinstance(self.time_scale, TimeScaleSettings):
            raise AudioContractError("Expected time scale settings")
        if self.time_scale.peak_ceiling is not None:
            raise AudioContractError("Apply peak ceiling after pitch conversion, not before")
        if type(self.lobes) is not int or not 4 <= self.lobes <= 128:
            raise AudioContractError("Pitch resampling lobes must be 4..128")
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling, "peak ceiling", minimum=.001, maximum=1.)


@dataclass(frozen=True, slots=True)
class PitchShiftReport:
    requested_semitones: float
    effective_semitones: float
    source_sample_rate: int
    virtual_sample_rate: int
    source_frames: int
    intermediate_frames: int
    converted_frames: int
    output_frames: int
    boundary_correction_frames: int
    pre_limit_peak: float
    output_peak: float
    attenuation: float
    time_scale: dict
    protocol: str = "sonumio-offline-pitch-shift-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "model_inference": False,
            "formants_preserved": False,
            "perceptual_quality_established": False,
        }


@dataclass(frozen=True, slots=True)
class PitchShiftResult:
    audio: AudioBuffer
    report: PitchShiftReport


def pitch_shift(audio: AudioBuffer, settings: PitchShiftSettings) -> PitchShiftResult:
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, PitchShiftSettings):
        raise AudioContractError("Expected AudioBuffer and PitchShiftSettings")
    requested = float(settings.semitones)
    virtual = round(audio.sample_rate * 2 ** (requested / 12))
    # Do not bypass clock restrictions by relabelling an illegal buffer.
    check_rate(virtual)
    effective = 12 * math.log2(virtual / audio.sample_rate)
    target = target_frames_for_ratio(audio.frames, virtual / audio.sample_rate)
    stretched = stretch_audio(audio, target, settings=settings.time_scale)
    if virtual == audio.sample_rate:
        converted = stretched.audio.samples
    else:
        cfg = ResampleSettings(audio.sample_rate, lobes=settings.lobes)
        converted = sinc_resample(stretched.audio.samples, virtual,
                                  audio.sample_rate, settings=cfg)
    actual_frames = converted.shape[1]
    correction = audio.frames - actual_frames
    # Nearest integer intermediate frame followed by nearest integer
    # conversion can disagree by one frame at bounded ratios.
    if abs(correction) > 1:
        raise AudioProcessingError("Unexpected pitch conversion frame discrepancy")
    if correction < 0:
        converted = converted[:, :audio.frames]
    elif correction > 0:
        converted = np.pad(converted, ((0, 0), (0, correction)), mode="edge")
    pre = float(np.max(np.abs(converted)))
    scale = 1.
    if settings.peak_ceiling is not None and pre > settings.peak_ceiling:
        scale = settings.peak_ceiling / pre
        converted = (converted.astype(np.float64) * scale).astype(np.float32)
    result = audio.copy_with(converted, source_note="offline pitch shift")
    return PitchShiftResult(result, PitchShiftReport(
        requested, effective, audio.sample_rate, virtual, audio.frames, target,
        actual_frames, result.frames, correction, pre,
        float(np.max(np.abs(result.samples))), scale, stretched.report.as_dict(),
    ))
