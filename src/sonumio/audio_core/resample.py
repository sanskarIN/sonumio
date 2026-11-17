"""Bounded sample-clock conversion with explicitly labelled fidelity.

Two algorithms are provided:
  - linear: fast but NOT anti-aliased; never silently used for downsampling.
  - sinc: normalized finite-window sinc interpolation with anti-alias cutoff.

This is deterministic CPU NumPy research code for local PCM conversion.
It is NOT a substitute for critically assessed production resamplers,
and it is not wired into existing Sonumio training or inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import (
    AudioBuffer, AudioSpec, check_rate, as_channel_first,
    validate_allocation, MAX_BYTES,
)
from .errors import AudioContractError, AudioLimitError


@dataclass(frozen=True, slots=True)
class ResampleSettings:
    target_rate: int
    method: str = "sinc"
    lobes: int = 24
    window: str = "kaiser"
    beta: float = 8.6
    chunk_frames: int = 8192

    def __post_init__(self):
        check_rate(self.target_rate)
        if self.method not in ("sinc", "linear"):
            raise AudioContractError("Unknown interpolation method")
        if type(self.lobes) is not int or not 4 <= self.lobes <= 128:
            raise AudioContractError("Sinc interpolation requires 4..128 lobes")
        if self.window not in ("kaiser", "hann", "blackman"):
            raise AudioContractError("Invalid resampling kernel window")
        if type(self.beta) not in (float, int) or not math.isfinite(
            self.beta
        ) or not 0 <= self.beta <= 20:
            raise AudioContractError("Invalid Kaiser shape")
        if type(self.chunk_frames) is not int or not 64 <= self.chunk_frames <= 65536:
            raise AudioContractError("Resampling chunk length must be 64..65536")


def output_frames(input_frames: int, source_rate: int, target_rate: int) -> int:
    source_rate, target_rate = check_rate(source_rate), check_rate(target_rate)
    if type(input_frames) is not int or input_frames <= 0:
        raise AudioContractError("Cannot resample empty/invalid source frame count")
    # Rational integer rounding. Ties select even to avoid systematic bias.
    numerator = input_frames * target_rate
    quotient, remainder = divmod(numerator, source_rate)
    twice = remainder * 2
    if twice > source_rate or (twice == source_rate and quotient % 2):
        quotient += 1
    if quotient == 0:
        raise AudioContractError("Target rate creates zero frames")
    return quotient


def linear_resample(source: np.ndarray, in_rate: int, out_rate: int,
                    *, allow_aliasing: bool = False) -> np.ndarray:
    if type(allow_aliasing) is not bool:
        raise AudioContractError("Expected explicit aliasing policy")
    check_rate(in_rate)
    check_rate(out_rate)
    if out_rate < in_rate and not allow_aliasing:
        raise AudioContractError(
            "Linear interpolation downsampling aliases; pass allow_aliasing=True explicitly"
        )
    data = as_channel_first(source, float64=True)
    n = output_frames(data.shape[1], in_rate, out_rate)
    validate_allocation(data.shape[0], n)
    if in_rate == out_rate:
        return data.astype(np.float32)
    positions = np.arange(n, dtype=np.float64) * (in_rate / out_rate)
    positions = np.minimum(positions, data.shape[1] - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, data.shape[1] - 1)
    a = positions - left
    return ((1 - a)[None, :] * data[:, left]
            + a[None, :] * data[:, right]).astype(np.float32)


def _window(distances: np.ndarray, radius: float, method: str,
            beta: float) -> np.ndarray:
    ratio = np.minimum(np.abs(distances) / radius, 1.0)
    if method == "hann":
        values = 0.5 * (1.0 + np.cos(math.pi * ratio))
    elif method == "blackman":
        values = (0.42 + 0.5 * np.cos(math.pi * ratio)
                  + 0.08 * np.cos(2 * math.pi * ratio))
    else:
        values = np.i0(beta * np.sqrt(np.maximum(0, 1 - ratio ** 2))) / np.i0(beta)
    return np.where(np.abs(distances) < radius, values, 0.0)


def sinc_resample(source: np.ndarray, in_rate: int, out_rate: int, *,
                  settings: ResampleSettings | None = None) -> np.ndarray:
    """Bandlimited approximate resampling in bounded frame batches.

    The centered windowed-sinc kernel is normalized per output sample,
    preserving DC including at clip boundaries. For downsampling, cutoff
    is reduced to the target Nyquist; upsampling keeps original Nyquist.
    """
    check_rate(in_rate)
    check_rate(out_rate)
    if settings is None:
        settings = ResampleSettings(out_rate)
    if settings.target_rate != out_rate or settings.method != "sinc":
        raise AudioContractError("Sinc settings must match target sample rate")
    data = as_channel_first(source, float64=True)
    frames = output_frames(data.shape[1], in_rate, out_rate)
    channels = data.shape[0]
    validate_allocation(channels, frames)
    if in_rate == out_rate:
        return data.astype(np.float32)
    if channels * frames * 8 > MAX_BYTES:
        raise AudioLimitError("Resample output exceeds double working budget")
    result = np.empty((channels, frames), dtype=np.float32)
    width = settings.lobes
    taps = np.arange(-width + 1, width + 1, dtype=np.int64)
    cutoff = min(1.0, out_rate / in_rate)
    radius = float(width)
    for start in range(0, frames, settings.chunk_frames):
        stop = min(frames, start + settings.chunk_frames)
        pos = (np.arange(start, stop, dtype=np.float64)
               * (in_rate / out_rate))
        centers = np.floor(pos).astype(np.int64)
        positions = centers[:, None] + taps[None, :]
        difference = pos[:, None] - positions
        kernel = cutoff * np.sinc(cutoff * difference) * _window(
            difference, radius, settings.window, float(settings.beta)
        )
        valid = (positions >= 0) & (positions < data.shape[1])
        kernel = np.where(valid, kernel, 0.0)
        weight = kernel.sum(axis=1)
        if np.any(np.abs(weight) <= 1e-12):
            raise AudioContractError("Sinc interpolation kernel underflow")
        kernel /= weight[:, None]
        safe = np.clip(positions, 0, data.shape[1] - 1)
        for channel in range(channels):
            # Matrix row (out-frame) reductions do not mix channels.
            result[channel, start:stop] = np.sum(
                data[channel, safe] * kernel, axis=1, dtype=np.float64
            ).astype(np.float32)
    if not np.isfinite(result).all():
        raise AudioContractError("Sinc resampler produced nonfinite output")
    return result


def resample_audio(buffer: AudioBuffer, settings: ResampleSettings
                   ) -> AudioBuffer:
    if not isinstance(buffer, AudioBuffer) or not isinstance(
        settings, ResampleSettings
    ):
        raise AudioContractError("Expected AudioBuffer and ResampleSettings")
    if settings.method == "linear":
        if settings.target_rate < buffer.sample_rate:
            raise AudioContractError(
                "Use the anti-aliased sinc method for downsampling"
            )
        data = linear_resample(buffer.samples, buffer.sample_rate,
                               settings.target_rate)
    else:
        data = sinc_resample(buffer.samples, buffer.sample_rate,
                             settings.target_rate, settings=settings)
    spec = AudioSpec(settings.target_rate, buffer.spec.layout, data.shape[1])
    return AudioBuffer(data, spec, source_note=buffer.source_note)


def resample_ratio(buffer: AudioBuffer, numerator: int, denominator: int, *,
                   lobes: int = 24) -> AudioBuffer:
    if (type(numerator) is not int or type(denominator) is not int
            or numerator <= 0 or denominator <= 0 or numerator > 64
            or denominator > 64):
        raise AudioContractError("Expected bounded positive rational rate ratio")
    g = math.gcd(numerator, denominator)
    numerator, denominator = numerator // g, denominator // g
    # Reject non-integral sample clocks instead of inventing a sample rate.
    result, remainder = divmod(buffer.sample_rate * numerator, denominator)
    if remainder:
        raise AudioContractError("Rational resample yields noninteger sample clock")
    return resample_audio(buffer, ResampleSettings(result, lobes=lobes))


def resample_report(original: AudioBuffer, converted: AudioBuffer,
                    settings: ResampleSettings) -> dict:
    if (not isinstance(original, AudioBuffer)
            or not isinstance(converted, AudioBuffer)
            or converted.sample_rate != settings.target_rate
            or original.channels != converted.channels):
        raise AudioContractError("Inconsistent resample report inputs")
    return {
        "source_hz": original.sample_rate,
        "target_hz": converted.sample_rate,
        "source_frames": original.frames,
        "target_frames": converted.frames,
        "source_seconds": original.seconds,
        "target_seconds": converted.seconds,
        "duration_difference_seconds": converted.seconds - original.seconds,
        "method": settings.method,
        "anti_alias_filter_applied": settings.method == "sinc",
        "neural_training_data_modified": False,
    }
