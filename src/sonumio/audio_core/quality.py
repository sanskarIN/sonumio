"""Signal-level health checks for bounded first-party audio pipelines.

This is NOT a speech, song or generative quality benchmark: it can detect
nonfinite samples, excessive clipping, channel imbalance, long silence,
abrupt boundary clicks and DC offset. It cannot tell whether the output
actually follows a human text prompt or sounds natural.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError
from .numerics import stable_peak, stable_rms, dbfs


@dataclass(frozen=True, slots=True)
class QualityLimits:
    max_peak: float = 1.0
    max_clipping_fraction: float = 0
    max_dc_absolute: float = 0.05
    max_boundary_jump: float = 0.25
    max_channel_imbalance_db: float = 24
    min_active_fraction: float = 0.001
    activity_floor_dbfs: float = -70

    def __post_init__(self):
        for name, value, low, high in (
            ("max_peak", self.max_peak, .01, 10),
            ("max_clipping_fraction", self.max_clipping_fraction, 0, 1),
            ("max_dc_absolute", self.max_dc_absolute, 0, 1),
            ("max_boundary_jump", self.max_boundary_jump, 0, 10),
            ("max_channel_imbalance_db", self.max_channel_imbalance_db, 0, 120),
            ("min_active_fraction", self.min_active_fraction, 0, 1),
            ("activity_floor_dbfs", self.activity_floor_dbfs, -160, 0),
        ):
            finite_float(value, name, minimum=low, maximum=high)


@dataclass(frozen=True, slots=True)
class QualityFinding:
    code: str
    severity: str
    measured: float
    limit: float
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualityReport:
    sample_rate: int
    frames: int
    channels: int
    peak_amplitude: float
    rms_amplitude: float
    dc_offset_absolute: float
    clipping_fraction: float
    active_fraction: float
    boundary_jump: float
    channel_imbalance_db: float
    findings: tuple[QualityFinding, ...]
    protocol: str = "sonumio-basic-signal-health-v1"

    @property
    def passed(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        value = asdict(self)
        value["passed"] = self.passed
        value["perceptual_audio_quality_measured"] = False
        value["text_prompt_adherence_measured"] = False
        value["listening_test_performed"] = False
        return value


def measure_quality(audio: AudioBuffer, *,
                    limits: QualityLimits = QualityLimits()) -> QualityReport:
    if not isinstance(audio, AudioBuffer) or not isinstance(limits, QualityLimits):
        raise AudioContractError("Expected AudioBuffer and QualityLimits")
    data = audio.samples.astype(np.float64)
    absolute = np.abs(data)
    peak, rms = stable_peak(audio.samples), stable_rms(audio.samples)
    dc = float(np.max(np.abs(np.mean(data, axis=1))))
    clipping = float(np.mean(absolute >= 1))
    active = float(np.mean(np.max(absolute, axis=0)
                           > 10 ** (limits.activity_floor_dbfs / 20)))
    if audio.frames < 2:
        jump = 0.0
    else:
        # Only boundary discontinuities, not all in-clip transients.
        jump = float(max(np.max(np.abs(data[:, :1])),
                         np.max(np.abs(data[:, -1:]))))
    rms_by_channel = np.sqrt(np.mean(data * data, axis=1))
    if np.max(rms_by_channel) <= 1e-12:
        imbalance = 0.0
    else:
        imbalance = 20 * math.log10(
            float(np.max(rms_by_channel))
            / max(float(np.min(rms_by_channel)), 1e-12)
        )
    checks = (
        ("peak", peak, limits.max_peak,
         "Measured sample amplitude exceeds selected peak budget"),
        ("clip_fraction", clipping, limits.max_clipping_fraction,
         "At least one sample clipped or explicit clip budget was exceeded"),
        ("dc_offset", dc, limits.max_dc_absolute,
         "Mean PCM bias exceeds explicit DC-offset limit"),
        ("boundary_jump", jump, limits.max_boundary_jump,
         "PCM boundary does not fade toward zero"),
        ("channel_imbalance", imbalance, limits.max_channel_imbalance_db,
         "Per-channel RMS imbalance exceeds selected budget"),
    )
    findings = [
        QualityFinding(code, "warning", value, cap, note)
        for code, value, cap, note in checks if value > cap
    ]
    if active < limits.min_active_fraction:
        findings.append(QualityFinding(
            "activity", "warning", active, limits.min_active_fraction,
            "Almost all frames are below the declared activity floor",
        ))
    return QualityReport(
        audio.sample_rate, audio.frames, audio.channels,
        peak, rms, dc, clipping, active, jump, imbalance,
        tuple(findings),
    )


def waveform_integrity_report(audio: AudioBuffer, *,
                              limits: QualityLimits = QualityLimits()
                              ) -> dict:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected audio buffer")
    from .hashing import digest_audio
    result = measure_quality(audio, limits=limits).as_dict()
    result["pcm_sha256"] = digest_audio(audio)
    result["numerically_finite"] = True
    result["model_weights_loaded"] = False
    result["does_not_verify_rights_or_consent"] = True
    return result


def compare_quality_reports(before: QualityReport, after: QualityReport) -> dict:
    if not isinstance(before, QualityReport) or not isinstance(after, QualityReport):
        raise AudioContractError("Expected two signal health reports")
    if (before.sample_rate != after.sample_rate or
            before.channels != after.channels or before.frames != after.frames):
        raise AudioContractError("Cannot compare different signal geometries")
    previous = {x.code for x in before.findings}
    current = {x.code for x in after.findings}
    return {
        "source_frames": before.frames,
        "new_failures": sorted(current - previous),
        "resolved_failures": sorted(previous - current),
        "persisting_failures": sorted(current & previous),
        "peak_delta": after.peak_amplitude - before.peak_amplitude,
        "dc_delta": after.dc_offset_absolute - before.dc_offset_absolute,
        "activity_delta": after.active_fraction - before.active_fraction,
        "human_perceptual_improvement_inferred": False,
    }
