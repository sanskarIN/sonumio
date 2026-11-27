"""Streaming unweighted gated energy diagnostics for local audio mastering.

The detector uses mean-square PCM averaged over channels. It intentionally
does not implement K weighting, surround weights or a calibrated LUFS
measurement. Absolute and relative gates operate on full fixed-clock
windows; a clip shorter than one window receives one declared short block.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioLimitError
from .streaming import FramePacket

MAX_WINDOWS = 100_000
MAX_PACKET_FRAMES = 65_536
MAX_DURATION_SECONDS = 3600
MAX_WINDOW_PRODUCTS = 500_000_000


@dataclass(frozen=True, slots=True)
class GatedEnergySettings:
    window_ms: int = 400
    hop_ms: int = 100
    absolute_gate_dbfs: float = -70.
    relative_gate_db: float = -10.

    def __post_init__(self):
        if type(self.window_ms) is not int or not 50 <= self.window_ms <= 2000:
            raise AudioContractError("Energy window must be 50..2000 ms")
        if type(self.hop_ms) is not int or not 10 <= self.hop_ms <= self.window_ms:
            raise AudioContractError("Energy hop must lie inside window")
        finite_float(self.absolute_gate_dbfs, "absolute energy gate", minimum=-120, maximum=-10)
        finite_float(self.relative_gate_db, "relative energy gate", minimum=-40, maximum=0)

    def geometry(self, rate):
        check_rate(rate)
        # Integer nearest rounding, with positive durations guaranteed.
        return ((rate*self.window_ms+500)//1000, (rate*self.hop_ms+500)//1000)


def energy_db(power):
    return 10*math.log10(power) if power > 0 else None


@dataclass(frozen=True, slots=True)
class GatedEnergyReport:
    sample_rate: int
    channels: int
    frames: int
    settings: GatedEnergySettings
    window_frames: int
    hop_frames: int
    full_windows: int
    short_window_used: bool
    omitted_tail_frames: int
    absolute_pass_windows: int
    relative_pass_windows: int
    overall_rms_dbfs: float | None
    absolute_gated_rms_dbfs: float | None
    relative_threshold_dbfs: float | None
    gated_rms_dbfs: float | None
    maximum_window_dbfs: float | None
    active_window_fraction: float
    protocol: str = "sonumio-unweighted-gated-energy-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "lufs_measured": False,
            "k_weighting_applied": False,
            "model_inference": False,
        }


def summarize_energy(powers, *, settings):
    """Apply two-stage gating in linear energy before converting to dB."""
    if (not isinstance(powers, np.ndarray) or powers.ndim != 1 or not len(powers)
            or not np.isfinite(powers).all() or np.any(powers < 0)):
        raise AudioContractError("Expected nonnegative finite window energies")
    if not isinstance(settings, GatedEnergySettings):
        raise AudioContractError("Expected GatedEnergySettings")
    absolute_floor = 10**(settings.absolute_gate_dbfs/10)
    absolute = powers[powers >= absolute_floor]
    if not len(absolute):
        return 0, 0, None, None, None
    average = float(np.mean(absolute))
    relative_floor = average*10**(settings.relative_gate_db/10)
    threshold = max(absolute_floor, relative_floor)
    relative = absolute[absolute >= threshold]
    return (
        len(absolute), len(relative), energy_db(average), energy_db(threshold),
        energy_db(float(np.mean(relative))) if len(relative) else None,
    )


class GatedEnergyAccumulator:
    """Bounded window energies; no raw PCM retained after overlap passes.

    Window sums use the same source-frame partition regardless of incoming
    packet size. Overall RMS is separately accumulated over every frame,
    including a trailing partial window omitted from gated statistics.
    """

    def __init__(self, sample_rate, channels, *, settings=GatedEnergySettings(),
                 expected_frames=None):
        self.sample_rate = check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid energy detector channel count")
        if not isinstance(settings, GatedEnergySettings):
            raise AudioContractError("Expected GatedEnergySettings")
        if expected_frames is not None and (
            type(expected_frames) is not int or not 1 <= expected_frames <= sample_rate*MAX_DURATION_SECONDS
        ):
            raise AudioLimitError("Invalid expected energy frame count")
        self.channels = channels
        self.settings = settings
        self.expected_frames = expected_frames
        self.window, self.hop = settings.geometry(sample_rate)
        self.frames_seen = 0
        self.next_start = 0
        self._base = 0
        self._buffer = np.empty(0, dtype=np.float64)
        self._powers = []
        self._total = 0.
        self._finished = False
        self._check_budget(expected_frames or 1)

    @property
    def retained_frames(self):
        return len(self._buffer)

    def _check_budget(self, frames):
        windows = 0 if frames < self.window else 1+(frames-self.window)//self.hop
        if windows > MAX_WINDOWS or windows*self.window > MAX_WINDOW_PRODUCTS:
            raise AudioLimitError("Gated energy window work exceeds budget")

    def update(self, packet):
        if self._finished:
            raise AudioContractError("Cannot update finalized energy detector")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected immutable FramePacket")
        if (packet.start_frame != self.frames_seen or packet.channels != self.channels
                or packet.sample_rate != self.sample_rate):
            raise AudioContractError("Energy detector packet position or clock changed")
        if packet.frames > MAX_PACKET_FRAMES:
            raise AudioLimitError("Energy detector packet exceeds bounded size")
        if packet.stop_frame > self.sample_rate*MAX_DURATION_SECONDS:
            raise AudioLimitError("Energy detector duration exceeds budget")
        if self.expected_frames is not None and packet.stop_frame > self.expected_frames:
            raise AudioContractError("Energy detector exceeded expected frame count")
        self._check_budget(packet.stop_frame)
        values = packet.samples.astype(np.float64)
        powers = np.mean(values*values, axis=0)
        self._total += float(np.sum(powers))
        self._buffer = np.concatenate((self._buffer, powers))
        self.frames_seen = packet.stop_frame
        while self.next_start+self.window <= self.frames_seen:
            local = self.next_start-self._base
            block = self._buffer[local:local+self.window]
            self._powers.append(float(np.mean(block)))
            self.next_start += self.hop
        if self.next_start > self._base:
            self._buffer = self._buffer[self.next_start-self._base:].copy()
            self._base = self.next_start

    def finish(self):
        if self._finished or not self.frames_seen:
            raise AudioContractError("Cannot finalize empty or finalized energy detector")
        if self.expected_frames is not None and self.frames_seen != self.expected_frames:
            raise AudioContractError("Incomplete energy detector frame count")
        full = len(self._powers)
        short = full == 0
        if short:
            self._powers.append(float(np.mean(self._buffer)))
        powers = np.array(self._powers, dtype=np.float64)
        absolute, relative, abs_db, threshold, gated = summarize_energy(
            powers, settings=self.settings,
        )
        omitted = 0 if short else self.frames_seen-((full-1)*self.hop+self.window)
        self._finished = True
        self._buffer = np.empty(0, dtype=np.float64)
        return GatedEnergyReport(
            self.sample_rate, self.channels, self.frames_seen, self.settings,
            self.window, self.hop, full, short, omitted, absolute, relative,
            energy_db(self._total/self.frames_seen), abs_db, threshold, gated,
            energy_db(float(powers.max())), absolute/len(powers),
        )


def measure_gated_energy(audio, *, settings=GatedEnergySettings(), packet_frames=8192):
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if type(packet_frames) is not int or not 1 <= packet_frames <= MAX_PACKET_FRAMES:
        raise AudioContractError("Invalid energy detector packet length")
    accumulator = GatedEnergyAccumulator(
        audio.sample_rate, audio.channels, settings=settings, expected_frames=audio.frames,
    )
    for start in range(0, audio.frames, packet_frames):
        accumulator.update(FramePacket(audio.sample_rate, start,
                           audio.samples[:, start:start+packet_frames]))
    return accumulator.finish()
