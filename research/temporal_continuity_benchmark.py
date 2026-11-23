"""Source-grounded temporal continuity benchmarks for original Sonumio audio.

Compare a predicted continuation against the ACTUAL neighboring samples
from the same 3-second, first-party DSP recording. Numerical controls:

* Oracle: exact ground truth future (all aligned error metrics must be zero).
* Silence: zero after the exact known previous-clip prefix.
* Wrong source: future from an unrelated acoustic recording, with the
  exact same correct known prefix grafted onto it.

Each source recording group is counted exactly once; bilingual English/
Hindi captions are NOT independent sounds. There is no perceptual score,
song/speech correctness assertion, trained model publication, file output
or assumption that low sample jump means an audible good continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from statistics import fmean

import numpy as np
import torch
from torch.nn import functional as F

from research.adjacent_audio import (
    AdjacentDataset, RECIPE, audit_adjacent_split,
    adjacent_bilingual_dataset,
)
from research.latent_audio import FRAMES, VALIDATION_SEEDS
from research.velocity_prefix_continuation import _prefix_frames
from research.wave_diffusion import SAMPLE_RATE

PROTOCOL = "sonumio-grounded-temporal-continuity-benchmark-v1"
WAVE_PROTOCOL = "sonumio-grounded-continuation-numerical-diagnostic-v1"
FFT_SIZES = (256, 512, 1024)
MAX_SIGNAL = 1e3
MAX_BATCH_RECORDINGS = 42
ARMS = ("true_adjacent_future", "silence_after_prefix", "unrelated_future")
ERROR_METRICS = (
    "future_sample_mae", "future_sample_rmse",
    "future_first_difference_mae",
    "future_log_spectral_l1",
    "future_spectral_convergence",
    "highband_power_fraction_error",
    "first_new_sample_jump_absolute_error",
    "first_new_sample_slope_change_absolute_error",
)


@dataclass(frozen=True, slots=True)
class SeamEvaluationConfig:
    """Fixed sample rate and bounded FFT protocol, no learned metrics."""

    prefix_frames: int = 1024
    fft_sizes: tuple[int, ...] = FFT_SIZES
    highband_cutoff_hz: int = 1600
    early_future_frames: int = 1024

    def __post_init__(self) -> None:
        _prefix_frames(self.prefix_frames)
        if (type(self.fft_sizes) is not tuple
                or not 1 <= len(self.fft_sizes) <= len(FFT_SIZES)
                or any(type(n) is not int or n not in FFT_SIZES
                       for n in self.fft_sizes)
                or len(set(self.fft_sizes)) != len(self.fft_sizes)
                or self.fft_sizes != tuple(sorted(self.fft_sizes))
                or type(self.highband_cutoff_hz) is not int
                or not 500 <= self.highband_cutoff_hz < 4000
                or type(self.early_future_frames) is not int
                or not 1024 <= self.early_future_frames <= FRAMES - self.prefix_frames
                or self.early_future_frames % 256):
            raise ValueError("Invalid bounded adjacent-continuation numerical protocol")

    def as_dict(self) -> dict:
        return {
            "prefix_frames": self.prefix_frames,
            "fft_sizes": list(self.fft_sizes),
            "highband_cutoff_hz": self.highband_cutoff_hz,
            "early_future_frames": self.early_future_frames,
            "sample_rate": SAMPLE_RATE,
        }


def _sha(audio: torch.Tensor) -> str:
    if (not isinstance(audio, torch.Tensor)
            or audio.device.type != "cpu"
            or audio.dtype != torch.float32 or audio.requires_grad
            or not torch.isfinite(audio).all()):
        raise ValueError("Invalid numeric audit waveform hash input")
    return hashlib.sha256(
        audio.detach().contiguous().numpy().tobytes()
    ).hexdigest()


def _wave(source: torch.Tensor, *, role: str) -> None:
    if (not isinstance(source, torch.Tensor)
            or source.shape != (FRAMES,)
            or source.dtype != torch.float32
            or source.device.type != "cpu"
            or source.requires_grad
            or not torch.isfinite(source).all()):
        raise ValueError("Source, candidate, and previous clips must be valid mono CPU audio")
    limit = 1.0001 if role in ("previous", "reference") else MAX_SIGNAL
    if source.abs().max().item() > limit:
        raise ValueError("Unbounded, wrong-range continuation sample input")


def _stft(mono: torch.Tensor, n_fft: int) -> torch.Tensor:
    if mono.ndim != 1 or mono.dtype != torch.float32 or n_fft not in FFT_SIZES:
        raise ValueError("Unsupported bounded spectral analysis frame")
    result = torch.stft(
        mono.view(1, -1), n_fft=n_fft, hop_length=n_fft // 4,
        window=torch.hann_window(n_fft, dtype=torch.float32),
        return_complex=True,
    ).abs().double()
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite waveform-magnitude reference")
    return result


def _spectrum(
    original: torch.Tensor, candidate: torch.Tensor,
    config: SeamEvaluationConfig,
) -> tuple[float, float, float]:
    spectral_errors = []
    convergences = []
    highband_errors = []
    for n_fft in config.fft_sizes:
        reference = _stft(original, n_fft)
        predicted = _stft(candidate, n_fft)
        spectral_errors.append(float(
            (torch.log1p(reference) - torch.log1p(predicted)).abs().mean()
        ))
        delta = predicted - reference
        convergences.append(float(
            torch.linalg.vector_norm(delta.flatten(1), dim=1).mean()
            / torch.linalg.vector_norm(reference.flatten(1), dim=1
                ).mean().clamp_min(1e-8)
        ))
        high = torch.fft.rfftfreq(
            n_fft, d=1 / SAMPLE_RATE
        ) >= config.highband_cutoff_hz
        def energy_ratio(magnitude):
            energy = magnitude.square()
            return (
                energy[:, high].sum()
                / energy.sum().clamp_min(1e-8)
            )
        highband_errors.append(float(
            (energy_ratio(predicted) - energy_ratio(reference)).abs()
        ))
    values = (fmean(spectral_errors), fmean(convergences),
              fmean(highband_errors))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Invalid numerical spectral-continuity result")
    return values


@torch.no_grad()
def score_continuation(
    previous: torch.Tensor, true_next: torch.Tensor,
    candidate_next: torch.Tensor, *,
    config: SeamEvaluationConfig | None = None,
) -> dict:
    """Compare unknown future frames to exact adjacent ground-truth samples.

    The known prefix is required to be exact in all inputs and is not
    counted as prediction error. First-unknown-sample jump, slope-change
    and spectral errors are measurements, NOT human listening scores.
    """
    spec = config if config is not None else SeamEvaluationConfig()
    if not isinstance(spec, SeamEvaluationConfig):
        raise ValueError("Unknown temporal-continuity protocol settings")
    _wave(previous, role="previous")
    _wave(true_next, role="reference")
    _wave(candidate_next, role="candidate")
    width = spec.prefix_frames
    known = previous[-width:]
    if (not torch.equal(true_next[:width], known)
            or not torch.equal(candidate_next[:width], known)):
        raise ValueError("Reference or predicted continuation lost exact source overlap")
    future = true_next[width:].double()
    candidate = candidate_next[width:].double()
    error = candidate - future
    if (len(error) < 1024
            or not torch.isfinite(error).all()):
        raise ValueError("Invalid unknown-future source or generated output")
    source_jump = float(true_next[width] - true_next[width - 1])
    model_jump = float(candidate_next[width] - candidate_next[width - 1])
    source_slope = float(
        (true_next[width + 1] - true_next[width])
        - (true_next[width] - true_next[width - 1])
    )
    model_slope = float(
        (candidate_next[width + 1] - candidate_next[width])
        - (candidate_next[width] - candidate_next[width - 1])
    )
    early = spec.early_future_frames
    log_stft, converge, highband = _spectrum(
        true_next[width:width + early],
        candidate_next[width:width + early], spec,
    )
    metrics = {
        "future_sample_mae": float(error.abs().mean()),
        "future_sample_rmse": float(error.square().mean().sqrt()),
        "future_first_difference_mae": float(
            (candidate[1:] - candidate[:-1]
             - (future[1:] - future[:-1])).abs().mean()
        ),
        "future_log_spectral_l1": log_stft,
        "future_spectral_convergence": converge,
        "highband_power_fraction_error": highband,
        "first_new_sample_jump_absolute_error": abs(model_jump - source_jump),
        "first_new_sample_slope_change_absolute_error": abs(
            model_slope - source_slope
        ),
    }
    if any(not math.isfinite(v) or v < 0 for v in metrics.values()):
        raise ValueError("Invalid nonfinite or negative numerical continuation error")
    return {
        "protocol": WAVE_PROTOCOL,
        "prefix_frames": width,
        "unknown_future_frames": FRAMES - width,
        "known_prefix_is_bitwise_equal": True,
        "previous_sha256": _sha(previous),
        "true_next_sha256": _sha(true_next),
        "candidate_next_sha256": _sha(candidate_next),
        "candidate_matches_true_future_exactly": bool(torch.equal(
            candidate_next, true_next
        )),
        "reference_first_unknown_sample_jump": source_jump,
        "candidate_first_unknown_sample_jump": model_jump,
        "metrics": metrics,
        "candidate_unclipped_peak": float(candidate_next.abs().max()),
        "candidate_outside_unit_range_fraction": float(
            (candidate_next.abs() > 1).float().mean()
        ),
        "sample_level_metric_only": True,
        "human_listening_or_semantic_continuity_measured": False,
        "model_weights_or_audio_written": False,
    }


def _controls(
    recording: AdjacentDataset, index: int,
) -> dict[str, torch.Tensor]:
    width = recording.prefix_frames
    correct = recording.future[index, 0].detach()
    unrelated_idx = (index + 2) % recording.examples
    if unrelated_idx == index or (
        recording.group_ids[unrelated_idx] == recording.group_ids[index]
    ):
        raise ValueError("Source-shuffle negative control must change acoustic group")
    unrelated = recording.future[unrelated_idx, 0].detach().clone()
    unrelated[:width] = correct[:width]
    muted = correct.clone()
    muted[width:] = 0
    if (torch.equal(unrelated, correct)
            or torch.equal(muted, correct)):
        raise ValueError("Negative controls accidentally match the ground truth")
    return {
        "true_adjacent_future": correct,
        "silence_after_prefix": muted,
        "unrelated_future": unrelated,
    }


def _group_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Cannot aggregate missing or nonfinite source metrics")
    return {
        "count": len(values),
        "mean": float(fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "positive_groups": sum(x > 0 for x in values),
        "zero_groups": sum(x == 0 for x in values),
    }


@torch.no_grad()
def benchmark_adjacent_dataset(
    data: AdjacentDataset, *,
    config: SeamEvaluationConfig | None = None,
) -> dict:
    """Evaluate all independent recordings once, with two negative controls."""
    spec = config if config is not None else SeamEvaluationConfig()
    if not isinstance(data, AdjacentDataset) or not isinstance(
        spec, SeamEvaluationConfig
    ):
        raise ValueError("Adjacent dataset and validated metric protocol required")
    data.check()
    if data.prefix_frames != spec.prefix_frames:
        raise ValueError("Evaluation overlap must match the true source adjacency")
    if (not 2 <= data.group_count <= MAX_BATCH_RECORDINGS
            or len(set(data.source_hashes)) != data.group_count):
        raise ValueError("Invalid acoustic-group coverage or duplicate sound source")
    rows = []
    seen_groups = set()
    for index, group in enumerate(data.group_ids):
        if group in seen_groups:
            continue
        seen_groups.add(group)
        previous = data.previous[index, 0].detach()
        controls = _controls(data, index)
        observations = {
            name: score_continuation(
                previous, controls["true_adjacent_future"], candidate,
                config=spec,
            ) for name, candidate in controls.items()
        }
        oracle = observations["true_adjacent_future"]
        if (not oracle["candidate_matches_true_future_exactly"]
                or any(oracle["metrics"][metric] > 1e-9
                       for metric in ERROR_METRICS)):
            raise ValueError("Exact source oracle did not reconstruct zero error")
        for negative in ("silence_after_prefix", "unrelated_future"):
            probe = observations[negative]
            if (probe["candidate_matches_true_future_exactly"]
                    or probe["metrics"]["future_sample_mae"] <= 0):
                raise ValueError("False-negative temporal control was not independent")
        rows.append({
            "acoustic_recording_group_sha256": group,
            "source_full_recording_sha256": data.source_hashes[index],
            "recording_seed": data.recording_seeds[index],
            "caption_language_provenance": "same_audio_en_and_hi",
            "candidate_scores": observations,
        })
    if len(rows) != data.group_count or seen_groups != set(data.group_ids):
        raise ValueError("Not all independent temporal-continuity sources evaluated")
    summaries = {
        arm: {
            metric: _group_summary([
                row["candidate_scores"][arm]["metrics"][metric]
                for row in rows
            ]) for metric in ERROR_METRICS
        }
        for arm in ARMS
    }
    config_dict = spec.as_dict()
    settings = {
        "data_fingerprint": data.fingerprint,
        "source_recipe": RECIPE,
        "recording_groups": data.group_count,
        "bilingual_caption_examples": data.examples,
        **config_dict,
    }
    return {
        "protocol": PROTOCOL,
        "configuration": settings,
        "configuration_sha256": hashlib.sha256(json.dumps(
            settings, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "source": {
            "independent_recordings": data.group_count,
            "bilingual_caption_examples": data.examples,
            "sample_rate": SAMPLE_RATE,
            "known_prefix_exactly_overlapped_in_source": True,
            "english_hindi_audio_counted_once_per_acoustic_group": True,
            "related_procedural_source_families": True,
            "source_sha256": list(sorted(set(data.source_hashes))),
        },
        "per_recording": rows,
        "group_descriptive_error": summaries,
        "ground_truth_oracle_zero_error_verified": True,
        "deliberately_mismatched_future_negative_control_verified": True,
        "zero_future_negative_control_verified": True,
        "same_prefix_for_all_three_arms": True,
        "no_pretrained_or_perceptual_model_used": True,
        "human_quality_or_audio_coherence_claim": False,
        "no_model_weights_or_wav_written": True,
        "model_publication_authorized": False,
        "warning": (
            "Zero-error oracle merely repeats source samples. "
            "Mismatched/silent baselines expose numerical score behavior, "
            "not how real people perceive audio continuation. "
            "Synthetic sound families are related and limited."
        ),
    }


def compare_continuations(
    real: torch.Tensor, baseline: torch.Tensor, candidate: torch.Tensor,
    previous: torch.Tensor, *, prefix_frames: int,
) -> dict:
    """Direct paired model-vs-reference error difference, numerical only."""
    spec = SeamEvaluationConfig(prefix_frames=prefix_frames)
    reference = score_continuation(
        previous, real, baseline, config=spec,
    )
    new = score_continuation(
        previous, real, candidate, config=spec,
    )
    return {
        "protocol": PROTOCOL,
        "reference_error": reference["metrics"],
        "candidate_error": new["metrics"],
        "candidate_minus_reference_error": {
            name: new["metrics"][name] - reference["metrics"][name]
            for name in ERROR_METRICS
        },
        "same_exact_previous_clip_and_true_future": True,
        "unclipped_source_and_candidate_inputs": True,
        "perceptual_audio_quality_or_long_context_semantics_measured": False,
    }
