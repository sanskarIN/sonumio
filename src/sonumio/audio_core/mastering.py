"""Explicit gated-RMS gain staging and estimated intersample peak guard.

The local offline pipeline never advertises broadcast conformance or
perceptual mastering quality. It preserves channel topology, makes
boost caps and dynamic limiting opt-in, and reports the actual residual
between requested and achieved gated RMS level.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError, AudioProcessingError
from .streaming import FramePacket
from .interpolated_peak import (
    PeakEstimateSettings, InterpolatedPeakAccumulator, InterpolatedPeakReport,
)
from .gated_energy import (
    GatedEnergySettings, GatedEnergyAccumulator, GatedEnergyReport,
)
from .lookahead_limiter import LookaheadSettings, LimiterReport, limit_audio

MAX_MASTERING_PCM_BYTES = 24*1024*1024
MAX_PACKET_FRAMES = 65_536
MODES = ("static", "lookahead")


@dataclass(frozen=True, slots=True)
class SignalMeasurement:
    peak: InterpolatedPeakReport
    energy: GatedEnergyReport

    def as_dict(self):
        return {
            "peak": self.peak.as_dict(),
            "energy": self.energy.as_dict(),
            "protocol": "sonumio-mastering-signal-measurement-v1",
        }


class MasteringMeasurement:
    """One source pass feeding both independent measurement accumulators."""

    def __init__(self, sample_rate, channels, *, peak=PeakEstimateSettings(),
                 energy=GatedEnergySettings(), expected_frames=None):
        self.peak = InterpolatedPeakAccumulator(
            sample_rate, channels, settings=peak, expected_frames=expected_frames,
        )
        self.energy = GatedEnergyAccumulator(
            sample_rate, channels, settings=energy, expected_frames=expected_frames,
        )
        self._state = "open"

    @property
    def state(self):
        return self._state

    def update(self, packet):
        if self._state != "open":
            raise AudioContractError("Cannot update closed mastering measurement")
        try:
            self.peak.update(packet)
            self.energy.update(packet)
        except Exception:
            # If the second accumulator rejects a packet after the first
            # accepted it, the joint measurement cannot be safely resumed.
            self._state = "failed"
            raise

    def finish(self):
        if self._state != "open":
            raise AudioContractError("Cannot finalize closed mastering measurement")
        try:
            result = SignalMeasurement(self.peak.finish(), self.energy.finish())
        except Exception:
            self._state = "failed"
            raise
        self._state = "finished"
        return result


def measure_mastering(audio, *, peak=PeakEstimateSettings(),
                      energy=GatedEnergySettings(), packet_frames=8192):
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if type(packet_frames) is not int or not 1 <= packet_frames <= MAX_PACKET_FRAMES:
        raise AudioContractError("Invalid mastering measurement packet length")
    acc = MasteringMeasurement(
        audio.sample_rate, audio.channels, peak=peak, energy=energy,
        expected_frames=audio.frames,
    )
    for start in range(0, audio.frames, packet_frames):
        acc.update(FramePacket(audio.sample_rate, start,
                              audio.samples[:, start:start+packet_frames]))
    return acc.finish()


@dataclass(frozen=True, slots=True)
class MasteringSettings:
    target_rms_dbfs: float | None = -20.
    max_boost_db: float = 0.
    estimated_peak_ceiling_dbfs: float = -1.
    mode: str = "static"
    peak: PeakEstimateSettings = PeakEstimateSettings()
    energy: GatedEnergySettings = GatedEnergySettings()
    lookahead_ms: int = 5
    release_ms: int = 80
    packet_frames: int = 8192

    def __post_init__(self):
        if self.target_rms_dbfs is not None:
            finite_float(self.target_rms_dbfs, "target gated RMS", minimum=-60, maximum=-3)
        finite_float(self.max_boost_db, "maximum boost", minimum=0, maximum=30)
        finite_float(self.estimated_peak_ceiling_dbfs, "estimated peak ceiling",
                     minimum=-40, maximum=0)
        if self.mode not in MODES:
            raise AudioContractError("Unknown mastering policy")
        if not isinstance(self.peak, PeakEstimateSettings) or not isinstance(self.energy, GatedEnergySettings):
            raise AudioContractError("Invalid mastering measurement configuration")
        LookaheadSettings(self.estimated_peak_ceiling_dbfs, self.lookahead_ms, self.release_ms)
        if type(self.packet_frames) is not int or not 1 <= self.packet_frames <= MAX_PACKET_FRAMES:
            raise AudioContractError("Invalid mastering packet length")

    @property
    def ceiling(self):
        return 10**(self.estimated_peak_ceiling_dbfs/20)


@dataclass(frozen=True, slots=True)
class MasteringReport:
    settings: MasteringSettings
    source: SignalMeasurement
    output: SignalMeasurement
    requested_gain_db: float
    initial_gain_db: float
    static_peak_guard_db: float
    final_peak_guard_db: float
    limiter: LimiterReport | None
    gated_target_error_db: float | None
    gated_signal_present: bool
    protocol: str = "sonumio-bounded-offline-mastering-v1"

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "settings": asdict(self.settings),
            "source": self.source.as_dict(),
            "output": self.output.as_dict(),
            "requested_gain_db": self.requested_gain_db,
            "initial_gain_db": self.initial_gain_db,
            "static_peak_guard_db": self.static_peak_guard_db,
            "final_peak_guard_db": self.final_peak_guard_db,
            "limiter": self.limiter.as_dict() if self.limiter else None,
            "gated_target_error_db": self.gated_target_error_db,
            "gated_signal_present": self.gated_signal_present,
            "certified_true_peak": False,
            "lufs_measured": False,
            "perceptual_quality_established": False,
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class MasteringResult:
    audio: AudioBuffer
    report: MasteringReport


def _measure(audio, settings):
    return measure_mastering(
        audio, peak=settings.peak, energy=settings.energy,
        packet_frames=settings.packet_frames,
    )


def _scale(audio, gain):
    if not math.isfinite(gain) or gain <= 0:
        raise AudioProcessingError("Invalid mastering gain")
    if gain == 1.:
        return audio.copy_with(audio.samples, source_note="mastering unity gain")
    samples = audio.samples.astype(np.float64)*gain
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > np.finfo(np.float32).max:
        raise AudioProcessingError("Mastering exceeds float32 PCM representation")
    return audio.copy_with(samples.astype(np.float32),
                           source_note="offline mastering gain")


def _guard_gain(peak, ceiling):
    if peak <= ceiling:
        return 1.
    # Small explicit margin covers float32 rounding on gain application.
    return ceiling/peak*(1-4*np.finfo(np.float32).eps)


def master_audio(audio, settings=MasteringSettings()):
    """Apply requested gated RMS gain, bounded by measured peak policy.

    The static mode attenuates the entire clip uniformly; the lookahead
    mode first performs linked sample-peak limiting. Both modes remeasure
    the output and use a final uniform attenuation only when necessary
    to satisfy this specific finite-sinc estimate. It is not proof of a
    continuous-time analog ceiling.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, MasteringSettings):
        raise AudioContractError("Expected AudioBuffer and MasteringSettings")
    if audio.spec.byte_count > MAX_MASTERING_PCM_BYTES:
        raise AudioLimitError("Mastering exceeds offline PCM byte budget")
    source = _measure(audio, settings)
    gated = source.energy.gated_rms_dbfs
    target = settings.target_rms_dbfs
    requested = 0. if gated is None or target is None else target-gated
    initial = min(requested, settings.max_boost_db)
    gain = 10**(initial/20)
    static_guard = 1.
    limiter_report = None
    if settings.mode == "static":
        static_guard = _guard_gain(source.peak.estimated_peak*gain, settings.ceiling)
        work = _scale(audio, gain*static_guard)
    else:
        # A source can have very large finite float32 values. To prevent
        # avoidable overflow before a limiter, reject unsupported gain
        # staging rather than clip or silently change the target.
        work = _scale(audio, gain)
        limited = limit_audio(work, LookaheadSettings(
            settings.estimated_peak_ceiling_dbfs,
            settings.lookahead_ms, settings.release_ms,
        ))
        work = limited.audio
        limiter_report = limited.report
    output = _measure(work, settings)
    final_guard = _guard_gain(output.peak.estimated_peak, settings.ceiling)
    if final_guard < 1.:
        work = _scale(work, final_guard)
        output = _measure(work, settings)
    if output.peak.estimated_peak > settings.ceiling:
        raise AudioProcessingError("Mastering failed its finite-sinc estimate ceiling")
    final_rms = output.energy.gated_rms_dbfs
    residual = final_rms-target if final_rms is not None and target is not None else None
    return MasteringResult(work, MasteringReport(
        settings, source, output, requested, initial,
        20*math.log10(static_guard), 20*math.log10(final_guard),
        limiter_report, residual, gated is not None,
    ))
