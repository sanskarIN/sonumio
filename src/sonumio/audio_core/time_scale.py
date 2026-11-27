"""Bounded offline pitch-preserving duration transformation.

Two deliberately distinct NumPy references are provided: WSOLA with
multichannel waveform matching, and a phase vocoder with linked channel
phase transport. Neither is a neural model or a guarantee of perceptual
quality. The public transform uses an exact output frame count and never
changes the input sample clock or layout.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, AudioSpec, finite_float
from .errors import AudioContractError, AudioLimitError, AudioProcessingError

METHODS = ("wsola", "phase_vocoder")
MAX_WORKING_BYTES = 192 * 1024 * 1024
MAX_CORRELATION_PRODUCTS = 240_000_000
MAX_FFT_POINTS = 32_000_000
MIN_RATIO = .5
MAX_RATIO = 2.
_EPSILON = 1e-15


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class TimeScaleSettings:
    method: str = "wsola"
    window_frames: int = 1024
    search_frames: int = 128
    fft_hop: int | None = None
    peak_ceiling: float | None = None

    def __post_init__(self):
        if self.method not in METHODS:
            raise AudioContractError("Unknown duration transformation method")
        size = _integer(self.window_frames, "window_frames", 64, 8192)
        if size & (size - 1):
            raise AudioContractError("Time scale window must be a power of two")
        _integer(self.search_frames, "search_frames", 0, size // 2)
        if self.fft_hop is not None:
            _integer(self.fft_hop, "fft_hop", 16, size // 4)
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling, "peak ceiling", minimum=.001, maximum=1)

    @property
    def synthesis_hop(self):
        return self.window_frames // 2 if self.method == "wsola" else (
            self.fft_hop if self.fft_hop is not None else self.window_frames // 4
        )


@dataclass(frozen=True, slots=True)
class TimeScaleReport:
    method: str
    source_frames: int
    output_frames: int
    sample_rate: int
    channels: int
    window_frames: int
    synthesis_hop: int
    analysis_windows: int
    adjusted_windows: int
    max_absolute_offset: int
    source_peak: float
    pre_limit_peak: float
    output_peak: float
    attenuation: float
    identity: bool
    protocol: str = "sonumio-offline-time-scale-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "duration_ratio": self.output_frames / self.source_frames,
            "sample_clock_changed": False,
            "model_inference": False,
            "perceptual_quality_established": False,
        }


@dataclass(frozen=True, slots=True)
class TimeScaleResult:
    audio: AudioBuffer
    report: TimeScaleReport


def target_frames_for_ratio(source_frames: int, ratio: float) -> int:
    """Round to nearest frame, with ties to even, on the declared clock."""
    _integer(source_frames, "source_frames", 1, 2**31 - 1)
    factor = finite_float(ratio, "duration ratio", minimum=MIN_RATIO,
                          maximum=MAX_RATIO)
    return max((source_frames + 1) // 2, round(source_frames * factor))


def _nearest_ratio(numerator, denominator):
    q, r = divmod(numerator, denominator)
    return q + int(2 * r > denominator or (2 * r == denominator and q % 2))


def _geometry(audio, target_frames, settings):
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, TimeScaleSettings):
        raise AudioContractError("Expected AudioBuffer and TimeScaleSettings")
    _integer(target_frames, "target_frames", 1, 2**31 - 1)
    if target_frames * 2 < audio.frames or target_frames > 2 * audio.frames:
        raise AudioContractError("Exact target duration must lie between half and twice the source")
    AudioSpec(audio.sample_rate, audio.spec.layout, target_frames)
    size = settings.window_frames
    hop = settings.synthesis_hop
    count = (target_frames + hop - 1) // hop + 1
    # One padded source, one accumulation buffer, normalization weights and
    # conservative space for FFT/correlation scratch; do not allocate a
    # complete spectrogram or a sliding-window candidate matrix.
    pad = size + settings.search_frames
    source_size = audio.frames + 2 * pad
    output_size = (count - 1) * hop + size
    estimate = (8 * audio.channels * (source_size + output_size)
                + 8 * output_size + 64 * audio.channels * size
                + 8 * audio.channels * target_frames)
    if estimate > MAX_WORKING_BYTES:
        raise AudioLimitError("Time scale exceeds bounded working memory")
    if settings.method == "wsola":
        work = count * audio.channels * hop * (2 * settings.search_frames + 1)
        if work > MAX_CORRELATION_PRODUCTS:
            raise AudioLimitError("WSOLA correlation work exceeds budget; reduce search or duration")
    elif count * audio.channels * size > MAX_FFT_POINTS:
        raise AudioLimitError("Phase vocoder FFT work exceeds budget")
    return size, hop, count, pad, output_size


def _window(size):
    # Periodic Hann, complementary at half-window stride.
    return .5 - .5 * np.cos(2 * np.pi * np.arange(size, dtype=np.float64) / size)


def _source_frame(padded, center, pad, size):
    start = center + pad - size // 2
    frame = padded[:, start:start + size]
    if frame.shape[1] != size:
        raise AudioProcessingError("Unexpected duration transform boundary")
    return frame


def _correlation_scores(reference, candidates):
    """Centered, joint-channel normalized correlation for every lag.

    Per-channel means are removed, but channel pairs are never downmixed;
    opposite-polarity stereo therefore retains useful matching evidence.
    Zero-energy candidates receive no preference over the nominal center.
    """
    width = reference.shape[1]
    centered = reference - reference.mean(axis=1, keepdims=True)
    energy = float(np.sum(centered * centered))
    count = candidates.shape[1] - width + 1
    if energy <= _EPSILON:
        return None
    dot = np.zeros(count, dtype=np.float64)
    norms = np.zeros(count, dtype=np.float64)
    for ref, values in zip(centered, candidates, strict=True):
        dot += np.correlate(values, ref, mode="valid")
        sums = np.concatenate(([0.], np.cumsum(values)))
        squares = np.concatenate(([0.], np.cumsum(values * values)))
        local_sum = sums[width:] - sums[:-width]
        local_square = squares[width:] - squares[:-width]
        norms += np.maximum(0., local_square - local_sum * local_sum / width)
    valid = norms > _EPSILON
    scores = np.full(count, -np.inf)
    scores[valid] = np.clip(dot[valid] / np.sqrt(energy * norms[valid]), -1, 1)
    return scores


def _match_center(padded, previous_tail, nominal, pad, size, search, source_frames):
    # Search only complete unpadded grains; boundary grains remain anchored.
    half = size // 2
    lo = max(half, nominal - search)
    hi = min(source_frames - half, nominal + search)
    if not search or lo > nominal or hi < nominal or lo == hi:
        return nominal
    first = pad + lo - half
    candidates = padded[:, first:first + hi - lo + half]
    scores = _correlation_scores(previous_tail, candidates)
    if scores is None or not np.isfinite(scores).any():
        return nominal
    best = float(np.max(scores))
    # Deterministic tie-breaking prefers nominal position then earliest lag.
    offsets = np.flatnonzero(scores >= best - 1e-12) + lo
    return int(min(offsets, key=lambda pos: (abs(pos - nominal), pos)))


def _wsola(padded, source_frames, target, size, hop, count, pad, output, weights, search):
    window = _window(size)
    previous_tail = None
    adjusted = maximum = 0
    for index in range(count):
        position = index * hop
        nominal = _nearest_ratio(position * source_frames, target)
        center = nominal
        if previous_tail is not None and position < target:
            center = _match_center(
                padded, previous_tail, nominal, pad, size, search, source_frames,
            )
        offset = abs(center - nominal)
        adjusted += bool(offset)
        maximum = max(maximum, offset)
        grain = _source_frame(padded, center, pad, size)
        output[:, position:position + size] += grain * window
        weights[position:position + size] += window
        previous_tail = grain[:, hop:].copy()
    return adjusted, maximum


def _wrap_phase(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _linked_phase_increment(current, previous, expected):
    """Estimate one rotation per bin from all channel cross-products.

    Preserve the current inter-channel complex ratios. A silent/degenerate
    bin advances at its nominal frequency rather than choosing a noisy
    independent phase per channel.
    """
    cross = np.sum(current * previous.conj(), axis=0)
    valid = np.abs(cross) > _EPSILON
    observed = np.angle(cross)
    observed = np.where(valid, observed, expected)
    return expected + _wrap_phase(observed - expected)


def _phase_vocoder(padded, source_frames, target, size, hop, count, pad, output, weights):
    window = _window(size)
    omega = 2 * np.pi * np.arange(size // 2 + 1) / size
    rotation = np.zeros_like(omega)
    previous = None
    last_center = None
    for index in range(count):
        position = index * hop
        center = _nearest_ratio(position * source_frames, target)
        grain = _source_frame(padded, center, pad, size)
        spectrum = np.fft.rfft(grain * window, axis=1)
        if previous is not None:
            delta = center - last_center
            if delta <= 0:
                raise AudioProcessingError("Analysis clock failed to advance")
            advance = _linked_phase_increment(spectrum, previous, omega * delta)
            rotation = _wrap_phase(rotation + advance * (hop / delta - 1))
        transformed = spectrum * np.exp(1j * rotation)
        # Real DC and Nyquist are required by a real inverse transform.
        # Do not rotate those bins into imaginary values and lose energy.
        transformed[:, 0] = spectrum[:, 0]
        transformed[:, -1] = spectrum[:, -1]
        frame = np.fft.irfft(transformed, n=size, axis=1)
        output[:, position:position + size] += frame * window
        weights[position:position + size] += window * window
        previous = spectrum
        last_center = center
    return 0, 0


def _finish(audio, samples, settings, target, count, adjusted, maximum, identity):
    if not np.isfinite(samples).all():
        raise AudioProcessingError("Time scaling produced nonfinite samples")
    pre = float(np.max(np.abs(samples)))
    scale = 1.
    if settings.peak_ceiling is not None and pre > settings.peak_ceiling:
        scale = settings.peak_ceiling / pre
        samples = samples * scale
    if np.max(np.abs(samples)) > np.finfo(np.float32).max:
        raise AudioProcessingError("Time scaling exceeds float32 range")
    result = audio.copy_with(samples.astype(np.float32),
                             source_note="offline duration transform: " + settings.method)
    report = TimeScaleReport(
        settings.method, audio.frames, target, audio.sample_rate, audio.channels,
        settings.window_frames, settings.synthesis_hop, count, adjusted, maximum,
        float(np.max(np.abs(audio.samples))), pre,
        float(np.max(np.abs(result.samples))), scale, identity,
    )
    return TimeScaleResult(result, report)


def stretch_audio(audio: AudioBuffer, target_frames: int, *,
                  settings: TimeScaleSettings = TimeScaleSettings()) -> TimeScaleResult:
    """Change duration, approximately preserving pitch, without resampling.

    The exact identity path does not touch phase or overlap-add. Both
    references use zero-padded centered windows; short or highly transient
    clips may not sound natural. Use the optional peak ceiling to attenuate,
    never to boost. No hard clipping is performed.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, TimeScaleSettings):
        raise AudioContractError("Expected AudioBuffer and TimeScaleSettings")
    _integer(target_frames, "target_frames", 1, 2**31 - 1)
    if target_frames == audio.frames:
        return _finish(audio, audio.samples, settings, target_frames, 0, 0, 0, True)
    size, hop, count, pad, output_size = _geometry(audio, target_frames, settings)
    padded = np.pad(audio.samples.astype(np.float64), ((0, 0), (pad, pad)))
    output = np.zeros((audio.channels, output_size), dtype=np.float64)
    weights = np.zeros(output_size, dtype=np.float64)
    args = (padded, audio.frames, target_frames, size, hop, count, pad, output, weights)
    if settings.method == "wsola":
        adjusted, maximum = _wsola(*args, settings.search_frames)
    else:
        adjusted, maximum = _phase_vocoder(*args)
    start = size // 2
    stop = start + target_frames
    coverage = weights[start:stop]
    if coverage.shape != (target_frames,) or np.any(coverage <= _EPSILON):
        raise AudioProcessingError("Incomplete time scale synthesis coverage")
    signal = output[:, start:stop] / coverage
    return _finish(audio, signal, settings, target_frames, count, adjusted, maximum, False)
