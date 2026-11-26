"""Private checkpoint replication screening against eligible training crops.

Time-resolved mel and aligned waveform similarities are separate diagnostics.
No threshold establishes plagiarism, memorization, or model release safety.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = "sonumio-private-replication-screen-v1"
SAMPLE_RATE = 8000
FRAMES = 2048
WINDOW = 256
HOP = 64
MEL_BANDS = 24
MEL_LAGS = (-2, -1, 0, 1, 2)
MIN_RMS = 1e-5
MAX_MAGNITUDE = 1e6
MAX_VALIDATION = 256
MAX_PAIRS = 100_000


@dataclass(frozen=True, slots=True)
class ScreenPolicy:
    pair_budget: int = 20_000
    waveform_threshold: float = .9995
    mel_threshold: float = .995

    def __post_init__(self):
        if type(self.pair_budget) is not int or not 1 <= self.pair_budget <= MAX_PAIRS:
            raise ValueError("Invalid replication pair budget")
        for value in (self.waveform_threshold, self.mel_threshold):
            if type(value) not in (int, float) or not math.isfinite(value) or not .9 <= value < 1:
                raise ValueError("Invalid replication review threshold")


def _mel_bank():
    hz = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    edges = 700 * (10 ** (np.linspace(0, mel, MEL_BANDS + 2) / 2595) - 1)
    left, middle, right = edges[:-2], edges[1:-1], edges[2:]
    weights = np.maximum(0, np.minimum(
        (hz[:, None] - left) / (middle - left),
        (right - hz[:, None]) / (right - middle)))
    if np.any(weights.sum(axis=0) <= 0):
        raise RuntimeError("Empty mel filter")
    return weights / weights.sum(axis=0)


MEL_BANK = _mel_bank()
TAPER = np.hanning(WINDOW + 1)[:-1]


def _unit(x):
    centered = x.astype(np.float64) - np.mean(x, dtype=np.float64)
    norm = np.linalg.norm(centered)
    return centered.ravel() / norm if norm > 1e-12 else None


@dataclass(frozen=True, slots=True)
class Signature:
    waveform: np.ndarray
    # Each lag is stored in both orientations so comparison is symmetric.
    mel_left: tuple[np.ndarray | None, ...]
    mel_right: tuple[np.ndarray | None, ...]


def signature(waveform: np.ndarray) -> Signature | None:
    if (not isinstance(waveform, np.ndarray) or waveform.shape != (FRAMES,)
            or waveform.dtype != np.float32 or not np.isfinite(waveform).all()
            or np.max(np.abs(waveform)) > MAX_MAGNITUDE):
        raise ValueError("Expected bounded finite float32 8 kHz crop")
    x = waveform.astype(np.float64)
    x -= x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    if rms < MIN_RMS:
        return None
    x /= rms
    windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW)[::HOP]
    power = np.abs(np.fft.rfft(windows*TAPER, axis=1))**2
    mel = power @ MEL_BANK
    # Global normalization keeps the relative temporal envelope rather
    # than independently normalizing away each frame's energy.
    logmel = np.log1p(10 * mel / mel.mean())
    left, right = [], []
    for lag in MEL_LAGS:
        count = len(logmel) - abs(lag)
        a, b = max(-lag, 0), max(lag, 0)
        left.append(_unit(logmel[a:a+count]))
        right.append(_unit(logmel[b:b+count]))
    return Signature(x / np.linalg.norm(x), tuple(left), tuple(right))


def similarities(a: Signature, b: Signature) -> tuple[float, float | None]:
    aligned = min(1., abs(float(np.dot(a.waveform, b.waveform))))
    mel_scores = [float(np.clip(np.dot(x, y), -1., 1.))
                  for x, y in zip(a.mel_left, b.mel_right, strict=True)
                  if x is not None and y is not None]
    return aligned, max(mel_scores) if mel_scores else None


@dataclass(slots=True)
class _Nearest:
    panel: str
    signature: Signature | None
    waveform: float | None = None
    mel: float | None = None

    def compare(self, other):
        if self.signature is None or other is None:
            return
        wave, mel = similarities(self.signature, other)
        self.waveform = wave if self.waveform is None else max(self.waveform, wave)
        if mel is not None:
            self.mel = mel if self.mel is None else max(self.mel, mel)


def _summarize(queries, policy):
    result = {"queries": len(queries)}
    for attr, threshold in (("waveform", policy.waveform_threshold), ("mel", policy.mel_threshold)):
        values = [getattr(q, attr) for q in queries if getattr(q, attr) is not None]
        result[attr] = {
            "scorable_queries": len(values),
            "unscorable_queries": len(queries)-len(values),
            "nearest_similarity_median": float(np.median(values)) if values else None,
            "nearest_similarity_maximum": max(values) if values else None,
            "review_candidates": sum(x >= threshold for x in values),
        }
    return result


def evaluate_replication(reader, ready, model, schedule, *, generation=None, policy=None):
    """Use exactly the same eligible training gallery for each query panel.

    Validation is the nonmember comparison baseline. Null guidance is
    sampled once per seed, not six times. All positive guidance categories
    share the same initial noise. No test crops enter similarity scoring.
    """
    from research.archive_generated_assay import (
        GenerationConfig, initial_noise, reverse_indices, sample_unclipped,
    )
    from research.archive_signal_reference import _audited_rows, _require_ready
    from research.archive_validation import model_fingerprint
    from research.wave_diffusion import CATEGORIES, CLIP_FRAMES

    generation = GenerationConfig(reverse_steps=8) if generation is None else generation
    policy = ScreenPolicy() if policy is None else policy
    if not isinstance(generation, GenerationConfig) or not isinstance(policy, ScreenPolicy):
        raise ValueError("Invalid replication screen configuration")
    _require_ready(reader, ready)
    if CLIP_FRAMES != FRAMES or reader.report()["sample_rate"] != SAMPLE_RATE:
        raise ValueError("Unsupported replication crop clock")
    reverse_indices(schedule.steps, generation.reverse_steps)
    train_count = len(ready.indices("train"))
    validation_count = len(ready.indices("validation"))
    positive = tuple(g for g in generation.guidances if g != 0)
    generated_count = len(generation.seeds) * (1 + len(CATEGORIES)*len(positive))
    expected_pairs = train_count * (validation_count + generated_count)
    if (not train_count or not 1 <= validation_count <= MAX_VALIDATION
            or expected_pairs > policy.pair_budget):
        raise ValueError("Replication screen exceeds full-panel budget")
    before = model_fingerprint(model)
    queries = [_Nearest("heldout_validation", signature(clip))
               for _, clip in _audited_rows(reader, ready, "validation")]
    if len(queries) != validation_count:
        raise ValueError("Validation panel count mismatch")
    for seed in generation.seeds:
        noise = initial_noise(seed)
        null = sample_unclipped(model, schedule, noise, 0, guidance=0,
                                reverse_steps=generation.reverse_steps)
        queries.append(_Nearest("guidance_0.0", signature(null)))
        for guidance in positive:
            for category in range(len(CATEGORIES)):
                samples = sample_unclipped(model, schedule, noise, category, guidance=guidance,
                                           reverse_steps=generation.reverse_steps)
                queries.append(_Nearest(f"guidance_{float(guidance)}", signature(samples)))
    if len(queries) != validation_count + generated_count:
        raise ValueError("Generated panel count mismatch")

    gallery = scorable_gallery = 0
    for _, clip in _audited_rows(reader, ready, "train"):
        gallery += 1
        reference = signature(clip)
        if reference is not None:
            scorable_gallery += 1
        for query in queries:
            query.compare(reference)
    if gallery != train_count or model_fingerprint(model) != before:
        raise ValueError("Reference gallery or model identity changed")
    panels = {name: _summarize([q for q in queries if q.panel == name], policy)
              for name in sorted({q.panel for q in queries})}
    reader.report()
    return {
        "protocol": PROTOCOL,
        "archive_sha256": ready.archive_sha256,
        "eligible_training_sha256": ready.eligible_fingerprints["train"],
        "eligible_validation_sha256": ready.eligible_fingerprints["validation"],
        "model_sha256": before,
        "sample_rate": SAMPLE_RATE, "clip_frames": FRAMES,
        "mel_window": WINDOW, "mel_hop": HOP, "mel_bands": MEL_BANDS,
        "mel_lags": list(MEL_LAGS),
        "review_thresholds": {"waveform": policy.waveform_threshold, "mel": policy.mel_threshold},
        "pair_budget": policy.pair_budget, "planned_pairs": expected_pairs,
        "gallery_crops": gallery, "scorable_gallery_crops": scorable_gallery,
        "generated_queries": generated_count, "validation_queries": validation_count,
        "panels": panels, "generation_seeds": list(generation.seeds),
        "reverse_steps": generation.reverse_steps, "diffusion_profile": schedule.profile,
        "diffusion_steps": schedule.steps,
        "test_split_scored": False, "gallery_is_full_eligible_training_split": True,
        "thresholds_empirically_calibrated": False,
        "model_state_unchanged": True, "memorization_established": False,
        "memorization_absence_established": False, "files_written": 0,
    }


def screen_checkpoint(archive, *, archive_sha256, checkpoint, checkpoint_sha256,
                      training, generation=None, policy=None):
    import torch
    from research.archive_checkpoint import restore_training_checkpoint
    from research.archive_readiness import audit_archive
    from research.archive_training_contracts import ArchiveTrainingConfig, make_archive_optimizer
    from research.archive_validation import model_fingerprint
    from research.corpus_archive import CorpusArchiveReader
    from research.corpus_archive_format import require_sha256
    from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule

    require_sha256(archive_sha256)
    require_sha256(checkpoint_sha256)
    if not isinstance(training, ArchiveTrainingConfig):
        raise ValueError("Expected archive training configuration")
    with CorpusArchiveReader(archive, expected_sha256=archive_sha256) as reader:
        ready = audit_archive(reader)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(training.seed)
            with torch.device("cpu"):
                model = TinyWaveDenoiser(training.channels)
                schedule = DiffusionSchedule(training.diffusion_steps, profile=training.diffusion_profile)
            optimizer = make_archive_optimizer(model)
            restored = restore_training_checkpoint(
                checkpoint, expected_sha256=checkpoint_sha256, config=training,
                readiness=ready, initial_model_sha256=model_fingerprint(model),
                model=model, optimizer=optimizer)
            report = evaluate_replication(reader, ready, model, schedule,
                                          generation=generation, policy=policy)
    return {**report, "checkpoint_sha256": restored.sha256,
            "completed_training_steps": restored.progress.completed_steps}


def main(argv=None):
    from research.archive_generated_assay import GenerationConfig
    from research.archive_study_plan import parse_seeds
    from research.archive_training_contracts import ArchiveTrainingConfig
    from research.wave_diffusion import DiffusionSchedule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    defaults = ArchiveTrainingConfig()
    for field in ("seed", "steps", "batch_size", "channels", "diffusion_steps",
                  "validation_seed", "validation_repeats", "validation_batch_size"):
        parser.add_argument("--"+field.replace("_", "-"), type=int, default=getattr(defaults, field))
    parser.add_argument("--diffusion-profile", choices=DiffusionSchedule.PROFILES,
                        default=defaults.diffusion_profile)
    parser.add_argument("--reverse-steps", type=int, default=8)
    parser.add_argument("--generation-seeds", default="701,709,727")
    parser.add_argument("--pair-budget", type=int, default=20_000)
    args = parser.parse_args(argv)
    try:
        training = ArchiveTrainingConfig(**{key: getattr(args, key)
                                             for key in ArchiveTrainingConfig.__dataclass_fields__})
        generation = GenerationConfig(seeds=parse_seeds(args.generation_seeds),
                                      reverse_steps=args.reverse_steps)
        report = screen_checkpoint(args.archive, archive_sha256=args.archive_sha256,
                                   checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                                   training=training, generation=generation,
                                   policy=ScreenPolicy(pair_budget=args.pair_budget))
    except (ValueError, OSError, RuntimeError):
        parser.exit(2, "Private replication screen failed verification or budget.\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
