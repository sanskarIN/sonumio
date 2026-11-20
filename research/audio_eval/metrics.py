"""Reproducible bounded sample and multi-resolution spectral comparisons.

All metrics require equal clocks/channels and finite PCM. No metric on its
own is a reliable human quality score, speech intelligibility measure or
music adherence score. The spectral reconstruction uses explicit windows
and avoids external pretrained embedding models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from sonumio.audio_core.contracts import AudioBuffer, as_channel_first
from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.numerics import dbfs, stable_peak, stable_rms
from sonumio.audio_core.stft import StftConfig, stft
from .reference import EvaluationPair

MAX_METRIC_DB = 160.0


def _pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = as_channel_first(x, float64=True), as_channel_first(y, float64=True)
    if a.shape != b.shape:
        raise AudioContractError("Reference and candidate sample arrays differ in shape")
    return a, b


def _finite_decibels(numerator: float, denominator: float,
                     *, floor: float = -160) -> float:
    if numerator < 0 or denominator < 0:
        raise AudioContractError("Expected nonnegative signal energies")
    if denominator <= 1e-30:
        return MAX_METRIC_DB if numerator > 1e-30 else floor
    if numerator <= 1e-30:
        return floor
    return float(np.clip(10 * math.log10(numerator / denominator),
                         floor, MAX_METRIC_DB))


def signal_to_error_ratio_db(reference: np.ndarray, candidate: np.ndarray
                             ) -> float:
    a, b = _pair(reference, candidate)
    numerator = float(np.sum(a * a, dtype=np.float64))
    denominator = float(np.sum((a - b) ** 2, dtype=np.float64))
    return _finite_decibels(numerator, denominator)


def si_sdr_db(reference: np.ndarray, candidate: np.ndarray, *,
              remove_dc: bool = True) -> float:
    """Scale-invariant source-to-distortion ratio, capped for finite JSON."""
    if type(remove_dc) is not bool:
        raise AudioContractError("remove_dc must be boolean")
    a, b = _pair(reference, candidate)
    if remove_dc:
        a = a - np.mean(a, axis=1, keepdims=True)
        b = b - np.mean(b, axis=1, keepdims=True)
    scores = []
    for src, predicted in zip(a, b):
        energy = float(np.dot(src, src))
        if energy <= 1e-20:
            scores.append(-MAX_METRIC_DB if np.any(np.abs(predicted) > 1e-10) else 0)
            continue
        scale = float(np.dot(src, predicted)) / energy
        target = src * scale
        error = predicted - target
        scores.append(_finite_decibels(float(np.dot(target, target)),
                                        float(np.dot(error, error))))
    return float(np.mean(scores))


def normalized_correlation(reference: np.ndarray, candidate: np.ndarray,
                           *, remove_dc: bool = True) -> float:
    a, b = _pair(reference, candidate)
    if remove_dc:
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
    ab = float(np.sum(a * b))
    magnitude = math.sqrt(float(np.sum(a*a)) * float(np.sum(b*b)))
    if magnitude <= 1e-20:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.clip(ab / magnitude, -1, 1))


def sample_abs_error(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a, b = _pair(reference, candidate)
    difference = a - b
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "sample_count": int(difference.size),
        "different_samples": int(np.count_nonzero(difference)),
    }


def spectral_log_distance(reference: AudioBuffer, candidate: AudioBuffer, *,
                          n_fft: int = 512, hop_length: int = 128,
                          epsilon: float = 1e-7) -> float:
    if not isinstance(reference, AudioBuffer) or not isinstance(candidate, AudioBuffer):
        raise AudioContractError("Expected immutable signal buffers")
    if not reference.spec.same_shape(candidate.spec):
        raise AudioContractError("Spectral distance needs equal clocks and shape")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 1e-12 <= epsilon <= 1e-2:
        raise AudioContractError("Invalid numerical spectral magnitude floor")
    config = StftConfig(n_fft=n_fft, hop_length=hop_length)
    aa, bb = stft(reference, config).magnitude(), stft(candidate, config).magnitude()
    diff = np.log(np.maximum(aa, epsilon)) - np.log(np.maximum(bb, epsilon))
    return float(np.sqrt(np.mean(diff**2)))


def spectral_convergence(reference: AudioBuffer, candidate: AudioBuffer, *,
                         n_fft: int = 512, hop_length: int = 128
                         ) -> float:
    if not reference.spec.same_shape(candidate.spec):
        raise AudioContractError("Spectral convergence needs equal signal clocks")
    config = StftConfig(n_fft=n_fft, hop_length=hop_length)
    a = stft(reference, config).magnitude()
    b = stft(candidate, config).magnitude()
    numerator = float(np.linalg.norm(a - b))
    denominator = float(np.linalg.norm(a))
    if denominator <= 1e-15:
        return 0 if numerator <= 1e-15 else MAX_METRIC_DB
    return numerator / denominator


def multi_resolution_distance(reference: AudioBuffer, candidate: AudioBuffer,
                              *, fft_sizes: tuple[int, ...] = (256, 512, 1024)
                              ) -> dict:
    if (not isinstance(fft_sizes, tuple) or not 1 <= len(fft_sizes) <= 5
            or len(set(fft_sizes)) != len(fft_sizes)):
        raise AudioContractError("Expected 1..5 distinct FFT sizes")
    rows = []
    for fft in fft_sizes:
        hop = fft // 4
        rows.append({
            "n_fft": fft, "hop": hop,
            "log_spectral_distance": spectral_log_distance(
                reference, candidate, n_fft=fft, hop_length=hop
            ),
            "spectral_convergence": spectral_convergence(
                reference, candidate, n_fft=fft, hop_length=hop
            ),
        })
    return {
        "resolutions": rows,
        "mean_log_spectral_distance": float(np.mean([
            x["log_spectral_distance"] for x in rows
        ])),
        "mean_spectral_convergence": float(np.mean([
            x["spectral_convergence"] for x in rows
        ])),
        "not_a_human_perceptual_rating": True,
    }


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Finite report of relative errors, not generative model rankings."""

    case_id: str
    reference_sha256: str
    candidate_sha256: str
    sample_rate: int
    channels: int
    frames: int
    reference_peak_dbfs: float
    candidate_peak_dbfs: float
    reference_rms_dbfs: float
    candidate_rms_dbfs: float
    max_sample_absolute_error: float
    mean_sample_absolute_error: float
    sample_rmse: float
    signal_error_ratio_db: float
    si_sdr_db: float
    normalized_correlation: float
    multiresolution_log_spectral_distance: float
    multiresolution_spectral_convergence: float
    same_bytes: bool

    def as_dict(self) -> dict:
        report = asdict(self)
        report["human_audio_quality_assessed"] = False
        report["rights_verified_automatically"] = False
        report["model_training_performed"] = False
        return report


def compare_signals(pair: EvaluationPair, *,
                    fft_sizes: tuple[int, ...] = (256, 512, 1024)
                    ) -> MetricSet:
    if not isinstance(pair, EvaluationPair):
        raise AudioContractError("Expected EvaluationPair")
    from sonumio.audio_core.hashing import digest_audio
    ref, can = pair.reference.buffer, pair.candidate
    sample = sample_abs_error(ref.samples, can.samples)
    spectral = multi_resolution_distance(ref, can, fft_sizes=fft_sizes)
    return MetricSet(
        case_id=pair.case_id,
        reference_sha256=ref_sha if (ref_sha := pair.reference.source_sha256)
        else digest_audio(ref),
        candidate_sha256=digest_audio(can),
        sample_rate=ref.sample_rate, channels=ref.channels, frames=ref.frames,
        reference_peak_dbfs=dbfs(stable_peak(ref.samples)),
        candidate_peak_dbfs=dbfs(stable_peak(can.samples)),
        reference_rms_dbfs=dbfs(stable_rms(ref.samples)),
        candidate_rms_dbfs=dbfs(stable_rms(can.samples)),
        max_sample_absolute_error=sample["max_abs"],
        mean_sample_absolute_error=sample["mean_abs"],
        sample_rmse=sample["rmse"],
        signal_error_ratio_db=signal_to_error_ratio_db(ref.samples, can.samples),
        si_sdr_db=si_sdr_db(ref.samples, can.samples),
        normalized_correlation=normalized_correlation(ref.samples, can.samples),
        multiresolution_log_spectral_distance=spectral["mean_log_spectral_distance"],
        multiresolution_spectral_convergence=spectral["mean_spectral_convergence"],
        same_bytes=np.array_equal(ref.samples, can.samples),
    )
