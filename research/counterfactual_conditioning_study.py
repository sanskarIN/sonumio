"""Private paired CPU intervention study for original synthetic audio research.

Compare the existing v-objective against the optional matched-only margin
objective using identical initialization, data, corruption, and codec training.
No audio or weights are written. Neither teacher-forced error nor synthetic
reference similarity is proof of semantic quality or Sonumio-1 readiness.
"""
from __future__ import annotations

import argparse
import json
import math
from platform import python_version

import numpy as np
import torch

from research.counterfactual_conditioning import (
    CaptionInterventions, heldout_interventions, RELATIVE_MARGIN, validate_weight,
)
from research.generated_source_grounding import (
    ARMS, LANGUAGES, REFERENCE_SEEDS, _fit, _reference_controls,
    _reference_data, _sample_assay, _summary,
)
from research.architecture_compare import _audio_hashes
from research.latent_audio import (
    TRAIN_SEEDS, VALIDATION_SEEDS, FRAMES, synthetic_bilingual_waves,
)
from research.velocity_ddim_study import _model_state_sha256, _train_models
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE

PROTOCOL = "sonumio-counterfactual-velocity-conditioning-v1"


def _codec_equal(first, second) -> bool:
    a, b = first.state_dict(), second.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _summaries(cases: list[dict], confounds: frozenset[str]) -> dict:
    return {
        arm: {
            language: {
                key: _summary(
                    [case[key] for case in cases
                     if case["arm"] == arm and case["language"] == language],
                    noise_confounded=confounds,
                )
                for key in ("matched", "rotated", "null_guidance")
            }
            for language in LANGUAGES
        }
        for arm in ARMS
    }


def _teacher_forced_deltas(baseline: dict, treatment: dict) -> dict:
    keys = ("matched_mse", "wrong_mse", "null_mse",
            "wrong_minus_matched_mse", "null_minus_matched_mse")
    return {
        arm: {
            level: {
                stratum: {
                    key: treatment[arm]["by_timestep_and_caption_stratum"][level][stratum][key] - row[key]
                    for key in keys
                }
                for stratum, row in strata.items()
            }
            for level, strata in baseline[arm]["by_timestep_and_caption_stratum"].items()
        }
        for arm in ARMS
    }


def run_study(
    *, seed: int = 17, train_steps: int = 16, codec_steps: int = 8,
    weight: float = 0.5, evaluations: int = 4,
    sampling_seeds: tuple[int, ...] = (301,),
) -> dict:
    validate_weight(weight)
    if (not weight or type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 64
            or type(codec_steps) is not int or not 1 <= codec_steps <= 64
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)):
        raise ValueError("Invalid bounded paired conditioning study configuration")
    all_seeds = TRAIN_SEEDS + VALIDATION_SEEDS + tuple(
        value for panel in REFERENCE_SEEDS.values() for value in panel
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Recording seeds overlap")
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    train_hashes = _audio_hashes(train.waves)
    val_hashes = _audio_hashes(validation.waves)
    waves, reference_hashes = _reference_data()
    if train_hashes & val_hashes or (train_hashes | val_hashes).intersection(
        digest for values in reference_hashes.values() for digest in values
    ):
        raise ValueError("Exact training, validation or reference recording overlap")
    probe = _fit(waves["fit"], waves["calibration"])
    reference = _reference_controls(probe, waves["holdout"])
    confounds = frozenset(reference["categories_accepted_for_noise"])
    table = CaptionInterventions.build(validation.tokens, VALIDATION_SEEDS)
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    results, codec_reference = {}, None
    with torch.random.fork_rng(devices=[]):
        for name, conditioning_weight in (("baseline", 0.0), ("treatment", weight)):
            torch.manual_seed(seed)
            models, codec, diagnostics = _train_models(
                seed=seed, train_steps=train_steps, codec_steps=codec_steps,
                schedule=schedule, channels=16, conditioning_weight=conditioning_weight,
            )
            if (diagnostics["training_fingerprint"] != train.fingerprint
                    or diagnostics["validation_fingerprint"] != validation.fingerprint):
                raise ValueError("Training corpus changed after overlap check")
            fingerprint = _model_state_sha256(models, codec)
            if codec_reference is None:
                codec_reference = codec
            elif not _codec_equal(codec_reference, codec):
                raise ValueError("Paired codec training diverged")
            with torch.no_grad():
                latent = torch.cat([codec.encode(batch) for batch in validation.waves.split(8)])
            heldout = {
                arm: heldout_interventions(
                    models[arm], validation.waves if arm == "raw" else latent,
                    validation.tokens, table, schedule, noise_seed=seed + 1024,
                )
                for arm in ARMS
            }
            cases, null_checks = [], []
            if reference["supports_interpreting_generated_scores"]:
                cases, null_checks = _sample_assay(
                    probe, models, codec, schedule, sampling_seeds=sampling_seeds,
                    evaluations=evaluations, guidance=1.0,
                )
            if _model_state_sha256(models, codec) != fingerprint:
                raise ValueError("Evaluation modified trained parameters")
            results[name] = {
                "training": diagnostics,
                "trained_model_sha256": fingerprint,
                "teacher_forced": heldout,
                "generated_reference_summaries": _summaries(cases, confounds) if cases else {},
                "generated_case_count": len(cases),
                "generated_cases": cases,
                "null_guidance_checks": null_checks,
            }
    control, treatment = results["baseline"], results["treatment"]
    equal_fields = (
        "initial_model_state_sha256", "training_input_trace_sha256", "retained_caption_examples",
        "training_fingerprint", "validation_fingerprint",
    )
    if any(control["training"][key] != treatment["training"][key] for key in equal_fields):
        raise ValueError("Paired training initialization or input trace diverged")
    return {
        "protocol": PROTOCOL,
        "runtime": {"python": python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "configuration": {
            "seed": seed, "train_steps": train_steps, "codec_steps": codec_steps,
            "conditioning_weight": weight, "relative_margin": RELATIVE_MARGIN,
            "evaluations": evaluations, "sampling_seeds": list(sampling_seeds),
            "sample_rate": SAMPLE_RATE, "frames": FRAMES,
        },
        "same_initial_parameters": True,
        "same_training_input_trace": True,
        "same_frozen_codec": True,
        "retained_caption_examples": control["training"]["retained_caption_examples"],
        "reference_probe": reference,
        "arms": results,
        "treatment_minus_baseline_teacher_forced": _teacher_forced_deltas(
            control["teacher_forced"], treatment["teacher_forced"],
        ),
        "generated_scores_available": bool(control["generated_case_count"]),
        "semantic_quality_established": False,
        "weights_or_audio_saved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--codec-steps", type=int, default=8)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[301])
    args = parser.parse_args()
    result = run_study(
        seed=args.seed, train_steps=args.train_steps, codec_steps=args.codec_steps,
        weight=args.weight, evaluations=args.evaluations,
        sampling_seeds=tuple(args.sampling_seeds),
    )
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
