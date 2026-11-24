"""Compare logical-batch training across microbatch sizes and explicit devices.

Use one original text waveform architecture and the same CPU-generated
training inputs. Train on synthetic recordings only. All held-out metrics
and DDIM sampling run on CPU so backend differences are not confounded
with evaluation-device changes. No model weights or waveforms are saved.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.device_velocity_training import TrainConfig, resolve_device, train_model
from research.velocity_loss_weights import POLICIES
from research.semantic_training import synthetic_source_targets
from research.generated_source_grounding import LANGUAGES, _summary
from research.latent_audio import TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves
from research.open_set_source_probe import evaluate_probe, fit_probe, reference_panels
from research.text_audio import TextWaveDenoiser, _synth_prompts
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_prediction import decode_velocity, velocity_target
from research.wave_diffusion import CATEGORIES, DiffusionSchedule

PROTOCOL = "sonumio-device-velocity-training-study-v1"


@torch.no_grad()
def heldout(model, data, schedule, *, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    levels = {}
    for level in sorted({0, schedule.steps // 2, schedule.steps - 1}):
        totals = {"matched_mse": 0.0, "null_mse": 0.0,
                  "matched_clean_mae": 0.0, "null_clean_mae": 0.0}
        count = 0
        for start in range(0, len(data.waves), 4):
            clean = data.waves[start:start + 4]
            tokens = data.tokens[start:start + len(clean)]
            # Paired English/Hindi caption rows see the same noise.
            noise = torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0)
            step = torch.full((len(clean),), level, dtype=torch.long)
            noisy = schedule.corrupt(clean, step, noise)
            target = velocity_target(clean, noise, step, schedule)
            for kind, caption in (("matched", tokens), ("null", torch.zeros_like(tokens))):
                predicted = model(noisy, step, caption)
                estimate, _ = decode_velocity(noisy, predicted, step, schedule)
                totals[kind + "_mse"] += float(F.mse_loss(predicted, target, reduction="sum"))
                totals[kind + "_clean_mae"] += float(F.l1_loss(estimate, clean, reduction="sum"))
            count += clean.numel()
        levels[str(level)] = {key: value / count for key, value in totals.items()}
    return {"by_timestep": levels, "unique_recordings": len(data.waves) // 2,
            "caption_rows_per_timestep": len(data.waves), "evaluation_device": "cpu"}


@torch.no_grad()
def generate(model, schedule, probe, *, seed, evaluations):
    settings = {"seed": seed, "evaluations": evaluations}
    a = sample_velocity_ddim(model, schedule, _synth_prompts((0,), "en", seed), guidance=0, **settings)
    b = sample_velocity_ddim(model, schedule, _synth_prompts((1,), "hi", seed), guidance=0, **settings)
    if not torch.equal(a, b):
        raise ValueError("Unconditional output depends on caption")
    null = probe.detailed_score(a.clamp(-1, 1))
    rows = []
    for language in LANGUAGES:
        for label, category in enumerate(CATEGORIES):
            wave = sample_velocity_ddim(
                model, schedule, _synth_prompts((label,), language, seed), guidance=1, **settings,
            )
            decision = probe.detailed_score(wave.clamp(-1, 1))
            rows.append({"language": language, "category": category,
                         "matched": decision.as_dict(label),
                         "rotated": decision.as_dict((label + 1) % len(CATEGORIES)),
                         "null_guidance": null.as_dict(label),
                         "unclamped_rms": float(wave.square().mean().sqrt()),
                         "out_of_range_fraction": float((wave.abs() > 1).float().mean())})
    return {"cases": rows, "null_caption_invariant": True,
            "by_language": {
                language: {condition: _summary([row[condition] for row in rows if row["language"] == language])
                           for condition in ("matched", "rotated", "null_guidance")}
                for language in LANGUAGES
            }}


def compare_parameters(left, right):
    a, b = left.state_dict(), right.state_dict()
    if a.keys() != b.keys():
        raise ValueError("Model parameter layouts differ")
    maximum = 0.0
    squared_sum = 0.0
    count = 0
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError("Model parameter shapes differ")
        diff = a[key].double() - b[key].double()
        maximum = max(maximum, float(diff.abs().max()))
        squared_sum += float(diff.square().sum())
        count += diff.numel()
    return {"maximum_absolute_parameter_difference": maximum,
            "parameter_rmse": math.sqrt(squared_sum / count), "scalar_parameters_and_buffers": count}


def run_study(*, config=TrainConfig(), alternative_microbatch=1, sampling_seed=301, evaluations=4,
              compare_objectives=False, compare_semantics=False):
    if not isinstance(config, TrainConfig):
        raise ValueError("Expected validated training configuration")
    # A smaller fixed experiment budget than the underlying training runner.
    if (config.steps > 256 or config.schedule_steps > 32
            or type(alternative_microbatch) is not int or not 1 <= alternative_microbatch <= min(8, config.batch_size)
            or alternative_microbatch == config.microbatch_size
            or type(sampling_seed) is not int or not 0 <= sampling_seed < 2**32
            or type(evaluations) is not int or not 2 <= evaluations <= config.schedule_steps):
        raise ValueError("Invalid bounded device study settings")
    if type(compare_objectives) is not bool or (compare_objectives and config.loss_policy != 'uniform'):
        raise ValueError('Objective comparison requires uniform reference policy')
    if type(compare_semantics) is not bool or (compare_semantics and config.semantic_weight != 0):
        raise ValueError('Semantic comparison requires unsupervised reference')
    resolve_device(config.device)
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, val_hashes = _audio_hashes(train.waves), _audio_hashes(validation.waves)
    sources, noises, hashes = reference_panels()
    reference_hashes = {h for domain in hashes.values() for panel in domain.values() for h in panel}
    if train_hashes & val_hashes or reference_hashes & (train_hashes | val_hashes):
        raise ValueError("Exact training/evaluation recording overlap")
    if (not torch.equal(train.waves[::2], train.waves[1::2])
            or not torch.equal(validation.waves[::2], validation.waves[1::2])):
        raise ValueError("Unexpected bilingual corpus ordering")
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    gate = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    schedule = DiffusionSchedule(config.schedule_steps, profile="cosine_zero_terminal_v1")
    variants = {"reference_cpu": replace(config, device="cpu"),
                "alternative_cpu": replace(config, microbatch_size=alternative_microbatch, device="cpu")}
    if config.device != "cpu":
        variants["requested_cuda"] = config
    if compare_objectives:
        for policy in ("min_snr", "clean_mse"):
            variants[policy + "_cpu"] = replace(config, device="cpu", loss_policy=policy)
    if compare_semantics:
        variants["semantic_cpu"] = replace(config, device="cpu", semantic_weight=0.5)
    training_targets = synthetic_source_targets(train.tokens, TRAIN_SEEDS) if any(v.semantic_weight for v in variants.values()) else None
    validation_targets = synthetic_source_targets(validation.tokens, VALIDATION_SEEDS) if training_targets is not None else None
    models, reports = {}, {}
    with torch.random.fork_rng(devices=[]):
        for name, variant in variants.items():
            # Seed the CPU default generator only, not every CUDA generator.
            torch.random.default_generator.manual_seed(config.seed)
            model = TextWaveDenoiser(16)
            started = perf_counter()
            kwargs = ({"semantic_targets": training_targets, "semantic_validation": (validation.tokens, validation_targets)}
                      if variant.semantic_weight else {})
            report = train_model(model, train.waves, train.tokens, variant, **kwargs)
            report["training_wall_seconds"] = perf_counter() - started
            # Includes transfers, hashing and synchronization, not just kernels.
            models[name] = model
            reports[name] = {
                "training": report,
                "heldout": heldout(model, validation, schedule, seed=config.seed + 2),
                "generated": (generate(model, schedule, probe, seed=sampling_seed, evaluations=evaluations)
                              if gate["gate"]["passed"] else None),
            }
    for key in ("initial_state_sha256",):
        if len({row["training"][key] for row in reports.values()}) != 1:
            raise ValueError("Paired initial model states differ")
    if len({row["training"]["batch_plan"]["input_trace_sha256"] for row in reports.values()}) != 1:
        raise ValueError("Paired CPU stochastic inputs differ")
    return {
        "protocol": PROTOCOL,
        "configuration": {**vars(config), "alternative_microbatch": alternative_microbatch,
                          "sampling_seed": sampling_seed, "evaluations": evaluations,
                          "compare_objectives": compare_objectives, "compare_semantics": compare_semantics},
        "training_fingerprint": train.fingerprint, "validation_fingerprint": validation.fingerprint,
        "unique_training_recordings": len(train_hashes), "unique_validation_recordings": len(val_hashes),
        "exact_split_overlap": False, "paired_initialization_and_inputs": True,
        "reference_probe": gate, "arms": reports,
        "parameter_comparisons": {
            name: compare_parameters(models["reference_cpu"], model)
            for name, model in models.items() if name != "reference_cpu"
        },
        "cuda_training_executed": "requested_cuda" in models,
        "weighted_objectives_not_directly_comparable": True,
        "cuda_performance_or_quality_established": False,
        "timing_includes_transfers_and_fingerprints": True,
        "weights_or_audio_saved": False, "model_readiness_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--alternative-microbatch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--schedule-steps", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--loss-policy", choices=POLICIES, default="uniform")
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--compare-objectives", action="store_true")
    parser.add_argument("--compare-semantics", action="store_true")
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--timestep-sampling", choices=("uniform", "stratified"), default="uniform")
    parser.add_argument("--caption-sampling", choices=("independent", "bilingual_pair"), default="independent")
    args = parser.parse_args()
    config = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                         microbatch_size=args.microbatch_size, device=args.device,
                         seed=args.seed, schedule_steps=args.schedule_steps,
                         loss_policy=args.loss_policy, min_snr_gamma=args.min_snr_gamma,
                         timestep_sampling=args.timestep_sampling, caption_sampling=args.caption_sampling,
                         semantic_weight=args.semantic_weight)
    print(json.dumps(run_study(config=config, alternative_microbatch=args.alternative_microbatch,
                               evaluations=args.evaluations, compare_objectives=args.compare_objectives,
                               compare_semantics=args.compare_semantics), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
