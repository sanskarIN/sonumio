"""Bounded, streaming finite-sinc intersample peak estimation.

This is an explicitly defined numerical reference, NOT a calibrated
ITU-R BS.1770 true-peak meter. It evaluates a finite Kaiser-windowed
sinc reconstruction at integer subdivisions of each source interval.
It measures only within the original sample domain; no oversampled PCM
array, extra time interval, external dependency or model is created.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, MAX_CHANNELS, check_rate, finite_float
from .errors import AudioContractError, AudioLimitError
from .streaming import FramePacket

MAX_PACKET_FRAMES = 65_536
MAX_PRODUCTS = 900_000_000
MAX_DURATION_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class PeakEstimateSettings:
    factor: int = 4
    radius: int = 16
    beta: float = 8.6

    def __post_init__(self):
        if type(self.factor) is not int or self.factor not in (2, 4, 8):
            raise AudioContractError("Interpolation factor must be 2, 4 or 8")
        if type(self.radius) is not int or not 4 <= self.radius <= 64:
            raise AudioContractError("Interpolation radius must be 4..64")
        finite_float(self.beta, "Kaiser beta", minimum=0, maximum=20)

    @property
    def taps(self):
        return 2*self.radius

    def work(self, channels, frames):
        return max(0, frames-1)*channels*(self.factor-1)*self.taps


@dataclass(frozen=True, slots=True)
class PeakLocation:
    channel: int
    frame: int
    phase: int
    amplitude: float
    signed_value: float

    def as_dict(self, factor, sample_rate):
        return {
            **asdict(self),
            "fractional_frame": self.frame+self.phase/factor,
            "seconds": (self.frame+self.phase/factor)/sample_rate,
        }


@dataclass(frozen=True, slots=True)
class InterpolatedPeakReport:
    sample_rate: int
    channels: int
    frames: int
    settings: PeakEstimateSettings
    sample_peak: float
    estimated_peak: float
    channel_peaks: tuple[PeakLocation, ...]
    products: int
    protocol: str = "sonumio-finite-sinc-peak-estimate-v1"

    @property
    def estimated_peak_dbfs(self):
        return 20*math.log10(self.estimated_peak) if self.estimated_peak else None

    @property
    def overshoot_db(self):
        return 20*math.log10(self.estimated_peak/self.sample_peak) if self.sample_peak else None

    @property
    def strongest(self):
        return min(self.channel_peaks, key=lambda p: (-p.amplitude, p.frame, p.phase, p.channel))

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "settings": asdict(self.settings),
            "sample_peak": self.sample_peak,
            "estimated_peak": self.estimated_peak,
            "estimated_peak_dbfs": self.estimated_peak_dbfs,
            "overshoot_db": self.overshoot_db,
            "channel_peaks": [p.as_dict(self.settings.factor, self.sample_rate)
                              for p in self.channel_peaks],
            "strongest": self.strongest.as_dict(self.settings.factor, self.sample_rate),
            "products": self.products,
            "boundary_policy": "zero_extension_inside_original_interval",
            "certified_true_peak": False,
            "model_inference": False,
        }


def interpolation_kernels(settings):
    """Return the noninteger phase kernels in ascending phase order.

    The centered support contains exactly 2*radius integer source taps.
    Each phase is independently normalized for unit DC response away
    from the zero-extended edges. Integer phase is the original sample,
    not a numerically approximated sinc convolution.
    """
    if not isinstance(settings, PeakEstimateSettings):
        raise AudioContractError("Expected PeakEstimateSettings")
    radius = settings.radius
    offsets = np.arange(-radius+1, radius+1, dtype=np.float64)
    kernels = []
    for phase in range(1, settings.factor):
        distance = offsets-phase/settings.factor
        inside = np.abs(distance) < radius
        shape = np.sqrt(np.maximum(0., 1-(distance/radius)**2))
        window = np.where(inside, np.i0(settings.beta*shape)/np.i0(settings.beta), 0.)
        weights = np.sinc(distance)*window
        total = float(weights.sum())
        if not math.isfinite(total) or abs(total) <= 1e-12:
            raise AudioContractError("Degenerate interpolation kernel")
        weights = weights/total
        weights.setflags(write=False)
        kernels.append(weights)
    return tuple(kernels)


class InterpolatedPeakAccumulator:
    """Measure consecutive packet partitions with bounded retained halo.

    A noninteger source interval is evaluated only when every required
    future tap is present, or during finalization at a known EOF. The
    last integer source frame is measured, but no interval after it.
    Results use earliest-frame, earliest-phase deterministic tie order.
    """

    def __init__(self, sample_rate, channels, *, settings=PeakEstimateSettings(),
                 expected_frames=None):
        self.sample_rate = check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid peak estimator channel count")
        if not isinstance(settings, PeakEstimateSettings):
            raise AudioContractError("Expected PeakEstimateSettings")
        if expected_frames is not None and (
            type(expected_frames) is not int or not 1 <= expected_frames <= sample_rate*MAX_DURATION_SECONDS
        ):
            raise AudioLimitError("Invalid expected peak estimator frame count")
        if expected_frames is not None and settings.work(channels, expected_frames) > MAX_PRODUCTS:
            raise AudioLimitError("Peak interpolation work exceeds budget")
        self.channels = channels
        self.settings = settings
        self.expected_frames = expected_frames
        self.kernels = interpolation_kernels(settings)
        self.frames_seen = 0
        self.next_interval = 0
        self._base = 0
        self._buffer = np.empty((channels, 0), dtype=np.float64)
        self._best = [PeakLocation(c, 0, 0, 0., 0.) for c in range(channels)]
        self._sample_peak = 0.
        self._finished = False

    @property
    def retained_frames(self):
        return self._buffer.shape[1]

    def _record(self, channel, values, start, phase):
        if not len(values):
            return
        index = int(np.argmax(np.abs(values)))
        value = float(values[index])
        candidate = PeakLocation(channel, start+index, phase, abs(value), value)
        current = self._best[channel]
        if ((-candidate.amplitude, candidate.frame, candidate.phase)
                < (-current.amplitude, current.frame, current.phase)):
            self._best[channel] = candidate

    def _intervals(self, stop):
        start = self.next_interval
        if stop <= start:
            return
        radius = self.settings.radius
        lo = start-radius+1
        hi = stop+radius
        valid_lo, valid_hi = max(0, lo), min(self.frames_seen, hi)
        if valid_lo < self._base or valid_hi > self._base+self.retained_frames:
            raise AudioContractError("Peak estimator discarded required context")
        values = self._buffer[:, valid_lo-self._base:valid_hi-self._base]
        left, right = valid_lo-lo, hi-valid_hi
        if left or right:
            values = np.pad(values, ((0, 0), (left, right)))
        expected = stop-start
        for channel, row in enumerate(values):
            for phase, kernel in enumerate(self.kernels, 1):
                output = np.correlate(row, kernel, mode="valid")
                if len(output) != expected or not np.isfinite(output).all():
                    raise AudioContractError("Invalid interpolation output")
                self._record(channel, output, start, phase)
        self.next_interval = stop

    def _discard(self):
        keep = max(0, self.next_interval-self.settings.radius+1)
        if keep > self._base:
            self._buffer = self._buffer[:, keep-self._base:].copy()
            self._base = keep

    def update(self, packet):
        if self._finished:
            raise AudioContractError("Cannot update finalized peak estimator")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected immutable FramePacket")
        if (packet.start_frame != self.frames_seen or packet.sample_rate != self.sample_rate
                or packet.channels != self.channels):
            raise AudioContractError("Peak estimator packet clock, topology or position changed")
        if packet.frames > MAX_PACKET_FRAMES:
            raise AudioLimitError("Peak estimator packet exceeds bounded size")
        if packet.stop_frame > self.sample_rate*MAX_DURATION_SECONDS:
            raise AudioLimitError("Peak estimator duration exceeds budget")
        if self.expected_frames is not None and packet.stop_frame > self.expected_frames:
            raise AudioContractError("Peak estimator exceeded expected frame count")
        if self.settings.work(self.channels, packet.stop_frame) > MAX_PRODUCTS:
            raise AudioLimitError("Peak interpolation work exceeds budget")
        values = packet.samples.astype(np.float64)
        for channel, row in enumerate(values):
            self._record(channel, row, packet.start_frame, 0)
            self._sample_peak = max(self._sample_peak, float(np.max(np.abs(row))))
        self._buffer = np.concatenate((self._buffer, values), axis=1)
        self.frames_seen = packet.stop_frame
        self._intervals(max(0, self.frames_seen-self.settings.radius))
        self._discard()

    def finish(self):
        if self._finished or self.frames_seen == 0:
            raise AudioContractError("Cannot finalize empty or finalized peak estimator")
        if self.expected_frames is not None and self.frames_seen != self.expected_frames:
            raise AudioContractError("Incomplete peak estimator frame count")
        self._intervals(self.frames_seen-1)
        self._finished = True
        self._buffer = np.empty((self.channels, 0), dtype=np.float64)
        peaks = tuple(self._best)
        return InterpolatedPeakReport(
            self.sample_rate, self.channels, self.frames_seen, self.settings,
            self._sample_peak, max(p.amplitude for p in peaks), peaks,
            self.settings.work(self.channels, self.frames_seen),
        )


def estimate_interpolated_peak(audio, *, settings=PeakEstimateSettings(),
                               packet_frames=8192):
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if type(packet_frames) is not int or not 1 <= packet_frames <= MAX_PACKET_FRAMES:
        raise AudioContractError("Invalid peak estimator packet length")
    estimator = InterpolatedPeakAccumulator(
        audio.sample_rate, audio.channels, settings=settings, expected_frames=audio.frames,
    )
    for start in range(0, audio.frames, packet_frames):
        estimator.update(FramePacket(audio.sample_rate, start,
            audio.samples[:, start:start+packet_frames]))
    return estimator.finish()
