"""Run the open-set probe on fixed held-out controls and optional DDIM samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from platform import python_version

import numpy as np
import torch

from research.architecture_compare import _audio_hashes
from research.generated_source_grounding import (
    ARMS, LANGUAGES, REFERENCE_SEEDS, _sample_assay, _summary,
)
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves,
)
from research.open_set_source_probe import (
    PROTOCOL, NOISE_SEEDS, NOISE_FAMILIES, baseline_comparison,
    evaluate_probe, fit_probe, phase_surrogate, reference_panels,
)
from research.velocity_ddim_study import _model_state_sha256, _train_models
from research.wave_diffusion import CATEGORIES, DiffusionSchedule, SAMPLE_RATE


def _summaries(cases: list[dict]) -> dict:
    return {
        arm: {
            language: {
                condition: _summary([
                    row[condition] for row in cases
                    if row["arm"] == arm and row["language"] == language
                ])
                for condition in ("matched", "rotated", "null_guidance")
            }
            for language in LANGUAGES
        }
        for arm in ARMS
    }


def run_study(
    *, train: bool = False, seed: int = 17, train_steps: int = 16,
    codec_steps: int = 8, evaluations: int = 4,
    sampling_seeds: tuple[int, ...] = (301,),
) -> dict:
    if (type(train) is not bool or type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 64
            or type(codec_steps) is not int or not 1 <= codec_steps <= 64
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)):
        raise ValueError("Invalid bounded open-set study configuration")
    sources, noises, hashes = reference_panels()
    training = synthetic_bilingual_waves(TRAIN_SEEDS)
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, val_hashes = _audio_hashes(training.waves), _audio_hashes(validation.waves)
    panel_hashes = {digest for domain in hashes.values() for split in domain.values() for digest in split}
    if train_hashes & val_hashes or panel_hashes.intersection(train_hashes | val_hashes):
        raise ValueError("Exact audio leakage across generator and probe panels")
    fingerprints = (training.fingerprint, validation.fingerprint)
    del training, validation

    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    old = baseline_comparison(sources, noises)
    heldout = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    # These examples preserve full-clip spectral power and deliberately disrupt
    # temporal phase. They are ambiguous and excluded from the strict noise gate.
    surrogates = []
    for label, category in enumerate(CATEGORIES):
        for recording_seed, waveform in zip(REFERENCE_SEEDS["holdout"], sources["holdout"][label], strict=True):
            surrogate = phase_surrogate(waveform, recording_seed + 2000)
            surrogates.append({
                "source": category, "seed": recording_seed,
                **probe.detailed_score(surrogate).as_dict(label),
            })
    split_report = {
        domain: {
            split: {
                "seeds": list((REFERENCE_SEEDS if domain == "source" else NOISE_SEEDS)[split]),
                "recordings": len(digests),
                "ordered_sha256": hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
            }
            for split, digests in panels.items()
        }
        for domain, panels in hashes.items()
    }
    cases, checks, diagnostics, model_sha = [], [], None, None
    status = "probe_only" if not train else "reference_gate_failed"
    if train and heldout["gate"]["passed"]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
            models, codec, diagnostics = _train_models(
                seed=seed, train_steps=train_steps, codec_steps=codec_steps,
                schedule=schedule, channels=16,
            )
            if (diagnostics["training_fingerprint"], diagnostics["validation_fingerprint"]) != fingerprints:
                raise ValueError("Training data changed after overlap verification")
            model_sha = _model_state_sha256(models, codec)
            cases, checks = _sample_assay(
                probe, models, codec, schedule, sampling_seeds=sampling_seeds,
                evaluations=evaluations, guidance=1.0,
            )
            if _model_state_sha256(models, codec) != model_sha:
                raise ValueError("Generation mutated trained parameters")
        status = "generated"
    return {
        "protocol": PROTOCOL,
        "runtime": {"python": python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "configuration": {
            "train_requested": train, "seed": seed, "train_steps": train_steps,
            "codec_steps": codec_steps, "evaluations": evaluations,
            "sampling_seeds": list(sampling_seeds), "sample_rate": SAMPLE_RATE, "frames": FRAMES,
        },
        "status": status,
        "source_categories": list(CATEGORIES),
        "noise_families": list(NOISE_FAMILIES),
        "reference_splits": split_report,
        "overlap_with_generator_data": False,
        "calibration": {
            "source_radii": dict(zip(CATEGORIES, map(float, probe.radii), strict=True)),
            "strict_noise_margin_thresholds": dict(zip(CATEGORIES, map(float, probe.margin_thresholds), strict=True)),
        },
        "old_probe_on_same_holdout": old,
        "open_set_holdout": heldout,
        "phase_surrogate_challenge": {
            "included_in_noise_gate": False,
            "cases": surrogates,
            "correct_accepted_source": sum(
                row["within_reference_radius"] and row["target_rank"] == 1 for row in surrogates
            ),
        },
        "training": diagnostics,
        "trained_model_sha256": model_sha,
        "generated_cases": cases,
        "null_guidance_checks": checks,
        "generated_summaries": _summaries(cases) if cases else {},
        "model_readiness_established": False,
        "weights_or_audio_saved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--codec-steps", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301])
    args = parser.parse_args()
    print(json.dumps(run_study(
        train=args.train, seed=args.seed, train_steps=args.train_steps,
        codec_steps=args.codec_steps, evaluations=args.evaluations,
        sampling_seeds=tuple(args.sampling_seeds),
    ), allow_nan=False, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
