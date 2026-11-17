"""Bounded audio shift estimation and honest aligned-region accounting.

Sample-clock alignment avoids corrupting spectral comparisons with a
one-sample playback offset, but can also mask real latency regressions.
Every result therefore records original lag and finite overlap metrics.
No time stretching, speaker identification or neural model inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from sonumio.audio_core.contracts import AudioBuffer, make_buffer, finite_float
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


def shift_preserving_length(audio: AudioBuffer, frames: int) -> AudioBuffer:
    """Shift PCM right for positive frames, left for negative, pad with zero."""
    if not isinstance(audio, AudioBuffer) or type(frames) is not int:
        raise AudioContractError("Expected audio and integral shift")
    if abs(frames) > audio.frames:
        raise AudioContractError("Shift cannot exceed sound duration")
    out = np.zeros_like(audio.samples)
    if frames > 0:
        out[:, frames:] = audio.samples[:, :-frames]
    elif frames < 0:
        out[:, :frames] = audio.samples[:, -frames:]
    else:
        out[:] = audio.samples
    return audio.copy_with(out, source_note="explicit zero-padding frame shift")


def _mono_detector(source: AudioBuffer) -> np.ndarray:
    if not isinstance(source, AudioBuffer):
        raise AudioContractError("Expected canonical source audio")
    data = source.samples.astype(np.float64)
    return np.mean(data, axis=0)


def estimate_lag(reference: AudioBuffer, candidate: AudioBuffer, *,
                 max_lag_frames: int = 1024,
                 minimum_rms: float = 1e-7) -> int:
    """Positive lag means candidate arrives LATER than reference.

    FFT cross-correlation computes correlation of reference with candidate.
    To avoid false agreement on near-silence, both sources must have
    sufficient energy and a unique strong peak; ambiguous ties return 0.
    """
    if not isinstance(reference, AudioBuffer) or not isinstance(candidate, AudioBuffer):
        raise AudioContractError("Expected reference and candidate AudioBuffers")
    if not reference.spec.same_shape(candidate.spec):
        raise AudioContractError("Lag estimation needs equal clocks and source shapes")
    if (type(max_lag_frames) is not int or max_lag_frames < 0
            or max_lag_frames > min(reference.frames - 1, 100000)):
        raise AudioContractError("Lag search exceeds bounded sample geometry")
    rms = finite_float(minimum_rms, "minimum alignment RMS",
                       minimum=0, maximum=1)
    x, y = _mono_detector(reference), _mono_detector(candidate)
    if math.sqrt(float(np.mean(x*x))) <= rms or math.sqrt(
        float(np.mean(y*y))
    ) <= rms:
        return 0
    # Correlate candidate against reference: for candidate later,
    # cross-correlation maximum occurs at a positive candidate shift.
    n = len(x)
    nfft = 1 << ((2 * n - 2).bit_length())
    if nfft * 16 > 128 * 1024 * 1024:
        raise AudioLimitError("Alignment FFT exceeds memory limit")
    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)
    corr = np.fft.irfft(Y * np.conj(X), n=nfft)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    wrapped = np.mod(lags, nfft)
    scores = corr[wrapped]
    max_score = float(np.max(scores))
    if not math.isfinite(max_score) or max_score <= 0:
        return 0
    matches = np.flatnonzero(
        np.abs(scores - max_score) <= max(1e-12, abs(max_score) * 1e-10)
    )
    if len(matches) != 1:
        return 0
    return int(lags[int(matches[0])])


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    reference: AudioBuffer
    candidate_aligned: AudioBuffer
    estimated_lag_frames: int
    valid_start: int
    valid_stop: int
    applied_shift_frames: int

    def __post_init__(self):
        if not self.reference.spec.same_shape(self.candidate_aligned.spec):
            raise AudioContractError("Aligned output source shape mismatch")
        if (type(self.estimated_lag_frames) is not int or
                type(self.applied_shift_frames) is not int or
                type(self.valid_start) is not int or type(self.valid_stop) is not int
                or not 0 <= self.valid_start < self.valid_stop <= self.reference.frames):
            raise AudioContractError("Invalid aligned evaluation region")

    @property
    def overlap_frames(self) -> int:
        return self.valid_stop - self.valid_start

    @property
    def overlap_fraction(self) -> float:
        return self.overlap_frames / self.reference.frames

    def report(self) -> dict:
        return {
            "source_sample_rate": self.reference.sample_rate,
            "original_frames": self.reference.frames,
            "lag_frames": self.estimated_lag_frames,
            "lag_seconds": self.estimated_lag_frames / self.reference.sample_rate,
            "shift_applied": self.applied_shift_frames,
            "valid_overlap_frames": self.overlap_frames,
            "valid_overlap_fraction": self.overlap_fraction,
            "time_stretch_applied": False,
            "silent_zero_padding_applied": self.applied_shift_frames != 0,
        }


def align_pair(reference: AudioBuffer, candidate: AudioBuffer, *,
               max_lag_frames: int = 1024,
               apply_alignment: bool = True) -> AlignmentResult:
    if type(apply_alignment) is not bool:
        raise AudioContractError("apply_alignment must be boolean")
    lag = estimate_lag(reference, candidate, max_lag_frames=max_lag_frames)
    shift = -lag if apply_alignment else 0
    aligned = shift_preserving_length(candidate, shift)
    if shift > 0:
        valid_start, valid_stop = shift, reference.frames
    elif shift < 0:
        valid_start, valid_stop = 0, reference.frames + shift
    else:
        valid_start, valid_stop = 0, reference.frames
    if valid_start >= valid_stop:
        raise AudioContractError("Alignment removed entire comparison signal")
    return AlignmentResult(reference, aligned, lag,
                           valid_start, valid_stop, shift)


def score_with_overlap(aligned: AlignmentResult,
                       metric, *, name: str = "metric") -> dict:
    """Evaluate only true overlapping samples; retain overlap disclosure."""
    if not isinstance(aligned, AlignmentResult) or not callable(metric):
        raise AudioContractError("Expected aligned pair and metric callback")
    a, b = aligned.valid_start, aligned.valid_stop
    x = aligned.reference.samples[:, a:b]
    y = aligned.candidate_aligned.samples[:, a:b]
    value = float(metric(x, y))
    if not math.isfinite(value):
        raise AudioContractError("Alignment metric returned nonfinite result")
    return {
        "name": name,
        "value": value,
        "overlap_frames": aligned.overlap_frames,
        "overlap_fraction": aligned.overlap_fraction,
        "estimated_lag_frames": aligned.estimated_lag_frames,
        "not_comparable_to_unaligned_full_length_without_disclosure": True,
    }
