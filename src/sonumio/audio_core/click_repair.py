"""Sparse waveform impulse detection and bounded local interpolation.

This opt-in CPU restoration reference detects isolated outliers relative to
a rolling median/MAD baseline. It does not classify musical transients
semantically, de-noise stationary backgrounds, or infer missing instruments.
Corrections are local, explicit and reported per channel.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError

MAX_DETECTION_BYTES = 192 * 1024 * 1024
MAX_EVENTS = 20_000
MAX_WINDOW = 31
MAX_RUN = 12


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class ClickRepairOptions:
    window_frames: int = 11
    mad_multiplier: float = 10.
    minimum_deviation: float = .04
    max_run_frames: int = 3
    max_fraction: float = .01
    interpolation: str = "linear"
    link_channels: bool = False

    def __post_init__(self) -> None:
        size = _int(self.window_frames, "window_frames", 5, MAX_WINDOW)
        if size % 2 != 1:
            raise AudioContractError("Click detection window must be odd")
        _int(self.max_run_frames, "max_run_frames", 1, MAX_RUN)
        if self.max_run_frames >= (size - 1) // 2:
            raise AudioContractError("Maximum click run must fit inside median window")
        for name, value, low, high in (
            ("mad_multiplier", self.mad_multiplier, 3., 100.),
            ("minimum_deviation", self.minimum_deviation, 1e-6, 10.),
            ("max_fraction", self.max_fraction, 0., .10),
        ):
            finite_float(value, name, minimum=low, maximum=high)
        if self.interpolation not in ("linear", "cubic"):
            raise AudioContractError("Click interpolation must be linear or cubic")
        if type(self.link_channels) is not bool:
            raise AudioContractError("Click channel linkage must be boolean")


@dataclass(frozen=True, slots=True)
class ClickDetection:
    """Immutable masks and diagnostic residual, same shape as audio PCM."""

    mask: np.ndarray
    residual: np.ndarray
    options: ClickRepairOptions

    def __post_init__(self):
        if not isinstance(self.options, ClickRepairOptions):
            raise AudioContractError("Expected ClickRepairOptions")
        mask = self.mask
        residual = self.residual
        if (not isinstance(mask, np.ndarray) or mask.ndim != 2
                or mask.dtype != np.bool_ or mask.size == 0
                or not isinstance(residual, np.ndarray)
                or residual.shape != mask.shape or residual.dtype != np.float64
                or not np.isfinite(residual).all()
                or np.any(residual < 0)):
            raise AudioContractError("Invalid click detection map")
        if mask.nbytes + residual.nbytes > MAX_DETECTION_BYTES:
            raise AudioLimitError("Click detection metadata exceeds memory budget")
        immutable_mask = np.frombuffer(mask.tobytes(), dtype=np.bool_).reshape(mask.shape)
        immutable_residual = np.frombuffer(
            np.asarray(residual, order="C").tobytes(), dtype=np.float64
        ).reshape(residual.shape)
        object.__setattr__(self, "mask", immutable_mask)
        object.__setattr__(self, "residual", immutable_residual)

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def fraction(self) -> float:
        return self.count / self.mask.size


@dataclass(frozen=True, slots=True)
class RepairEvent:
    channel: int
    start_frame: int
    stop_frame: int
    peak_absolute_residual: float
    interpolation: str

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class ClickRepairReport:
    sample_rate: int
    channels: int
    frames: int
    corrected_frames: int
    events: tuple[RepairEvent, ...]
    interpolation: str
    linked_channels: bool
    input_peak: float
    output_peak: float
    protocol: str = "sonumio-sparse-click-repair-v1"

    def as_dict(self):
        out = asdict(self)
        out["model_inference"] = False
        out["perceptual_quality_established"] = False
        out["automatic_transient_classification"] = False
        return out


@dataclass(frozen=True, slots=True)
class ClickRepairResult:
    audio: AudioBuffer
    detection: ClickDetection
    report: ClickRepairReport


def _run_spans(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return ()
    differences = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[differences + 1]))
    ends = np.concatenate((indices[differences] + 1, [indices[-1] + 1]))
    return tuple((int(a), int(b)) for a, b in zip(starts, ends))


def _validate_mask(mask, channels, frames, options):
    if (not isinstance(mask, np.ndarray)
            or mask.shape != (channels, frames)
            or mask.dtype != np.bool_):
        raise AudioContractError("Repair mask must be boolean [channel,frame]")
    if np.count_nonzero(mask) > options.max_fraction * mask.size + 1e-9:
        raise AudioLimitError("Proposed repairs exceed maximum corrected fraction")
    if np.any(mask[:, 0]) or np.any(mask[:, -1]):
        raise AudioContractError("Cannot interpolate boundary samples without neighbors")
    events = []
    for channel in range(channels):
        for start, stop in _run_spans(mask[channel]):
            if stop - start > options.max_run_frames:
                raise AudioContractError("Masked impulse run exceeds interpolation bound")
            events.append((channel, start, stop))
            if len(events) > MAX_EVENTS:
                raise AudioLimitError("Too many repair events")
    return tuple(events)


def _preflight(audio, options):
    if not isinstance(audio, AudioBuffer) or not isinstance(options, ClickRepairOptions):
        raise AudioContractError("Expected AudioBuffer and ClickRepairOptions")
    channels, frames = audio.channels, audio.frames
    width = options.window_frames
    # Padded PCM, median, MAD scratch, residual, mask and readonly copies.
    estimate = 8 * channels * frames * (3 * width + 14)
    if estimate > MAX_DETECTION_BYTES:
        raise AudioLimitError("Click detection exceeds bounded working memory")
    if frames < width + 2:
        raise AudioContractError("Click detection requires more PCM than median window")


def detect_clicks(
    audio: AudioBuffer,
    options: ClickRepairOptions = ClickRepairOptions(),
) -> ClickDetection:
    """Flag short, sparse channel-wise outliers using median absolute deviation.

    Candidate runs that cannot be interpolated without untouched neighbors,
    or exceed maximum duration, are left untouched. If too many samples
    qualify, reject instead of guessing that dense signal structure is noise.
    """
    _preflight(audio, options)
    data = audio.samples.astype(np.float64)
    radius = options.window_frames // 2
    padded = np.pad(data, ((0, 0), (radius, radius)), mode="edge")
    windows = sliding_window_view(padded, options.window_frames, axis=1)
    center = np.median(windows, axis=-1)
    deviations = np.abs(windows - center[:, :, None])
    local_mad = np.median(deviations, axis=-1)
    residual = np.abs(data - center)
    cutoff = np.maximum(options.minimum_deviation,
                        options.mad_multiplier * 1.4826 * local_mad)
    candidate = residual > cutoff
    candidate[:, 0] = False
    candidate[:, -1] = False
    # The default is independent channels. Channel linkage is opt-in and
    # mirrors suspected events to each channel to preserve common timing.
    if options.link_channels:
        candidate[:] = np.any(candidate, axis=0, keepdims=True)
    approved = np.zeros_like(candidate)
    for channel in range(audio.channels):
        for start, stop in _run_spans(candidate[channel]):
            if stop - start <= options.max_run_frames:
                approved[channel, start:stop] = True
    _validate_mask(approved, audio.channels, audio.frames, options)
    return ClickDetection(approved, residual, options)


def _linear(left, right, length):
    return left + (right - left) * (np.arange(1, length + 1) / (length + 1))


def _cubic(data, left_index, right_index):
    """Bounded monotone Hermite segment between two unmodified anchors."""
    y0, y1 = float(data[left_index]), float(data[right_index])
    length = right_index - left_index
    difference = (y1 - y0) / length
    if left_index:
        slope0 = y0 - float(data[left_index - 1])
    else:
        slope0 = difference
    if right_index + 1 < len(data):
        slope1 = float(data[right_index + 1]) - y1
    else:
        slope1 = difference
    if difference == 0:
        slope0 = slope1 = 0.
    else:
        sign = math.copysign(1., difference)
        bound = 3 * abs(difference)
        slope0 = sign * min(bound, max(0., sign * slope0))
        slope1 = sign * min(bound, max(0., sign * slope1))
    t = np.arange(1, length) / length
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    values = h00 * y0 + h10 * length * slope0 + h01 * y1 + h11 * length * slope1
    # Clamp to anchor range even after floating-point rounding.
    return np.clip(values, min(y0, y1), max(y0, y1))


def repair_clicks(
    audio: AudioBuffer,
    options: ClickRepairOptions = ClickRepairOptions(),
    *,
    explicit_mask: np.ndarray | None = None,
) -> ClickRepairResult:
    """Restore only selected sample spans; never alter unmasked samples.

    An explicit mask is useful after user review. It is validated with the
    same sparse-event and boundary constraints as automatically detected
    candidates. No more than one channel's data is silently inferred.
    """
    _preflight(audio, options)
    if explicit_mask is None:
        detection = detect_clicks(audio, options)
    else:
        source_mask = np.asarray(explicit_mask)
        if source_mask.dtype != np.bool_:
            raise AudioContractError("Explicit repair mask must have boolean dtype")
        # Explicit selections have no algorithmic residual baseline.
        detection = ClickDetection(
            source_mask,
            np.zeros(audio.samples.shape, dtype=np.float64),
            options,
        )
    events = _validate_mask(detection.mask, audio.channels, audio.frames, options)
    output = audio.mutable_copy(dtype="float64")
    reports = []
    for channel, start, stop in events:
        left = start - 1
        right = stop
        if options.interpolation == "linear":
            interpolated = _linear(output[channel, left], output[channel, right],
                                   stop - start)
        else:
            interpolated = _cubic(output[channel], left, right)
        output[channel, start:stop] = interpolated
        peak = float(np.max(detection.residual[channel, start:stop]))
        reports.append(RepairEvent(channel, start, stop, peak, options.interpolation))
    if not np.isfinite(output).all():
        raise AudioContractError("Restoration produced nonfinite audio")
    if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Restoration exceeds float32 range")
    result = audio.copy_with(output.astype(np.float32),
                             source_note="opt-in sparse waveform click restoration")
    return ClickRepairResult(
        result, detection,
        ClickRepairReport(
            audio.sample_rate, audio.channels, audio.frames, detection.count,
            tuple(reports), options.interpolation, options.link_channels,
            float(np.max(np.abs(audio.samples))),
            float(np.max(np.abs(result.samples))),
        ),
    )
