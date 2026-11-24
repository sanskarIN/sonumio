"""Compare lossless patch coordinates with matched initial waveform functions."""
from __future__ import annotations

import argparse
import json
import math

import torch

from research.architecture_compare import _audio_hashes
from research.counterfactual_conditioning import CaptionInterventions, heldout_interventions
from research.heldout_caption_panel import evaluate_caption_panel
from research.latent_audio import TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves
from research.multiscale_waveform_study import _train, _heldout, _generate, state_sha256
from research.open_set_source_probe import fit_probe, evaluate_probe, reference_panels
from research.paired_grounding_audit import _exact_assignment
from research.reversible_waveform_bases import BASES, ReversibleBasisDenoiser
from research.wave_diffusion import CATEGORIES, DiffusionSchedule

PROTOCOL = "sonumio-reversible-waveform-basis-study-v1"


def _configuration(seed, steps, evaluations, sampling_seeds, spectral_weight):
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(steps) is not int or not 1 <= steps <= 256
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)
            or type(spectral_weight) not in (int, float)
            or not math.isfinite(spectral_weight) or not 0 <= spectral_weight <= 2):
        raise ValueError("Invalid bounded reversible-basis experiment")


@torch.no_grad()
def initial_function_check(model, reference, noisy, steps, captions):
    model.eval()
    actual = model(noisy, steps, captions)
    diff = (actual - reference).abs()
    if not torch.allclose(actual, reference, atol=2e-5, rtol=2e-5):
        raise ValueError("Coordinate rotation failed to align the initial waveform function")
    return {"maximum_absolute_error": float(diff.max()), "mean_absolute_error": float(diff.mean()),
            "alignment_within_tolerance": True}


def assignment_panels(generated):
    panels = []
    for seed in sorted({row["seed"] for row in generated["cases"]}):
        for language in ("en", "hi"):
            rows = {row["category"]: row for row in generated["cases"]
                    if row["seed"] == seed and row["language"] == language}
            if set(rows) != set(CATEGORIES):
                raise ValueError("Incomplete generation panel")
            forced = tuple(rows[c]["matched"]["closest"] for c in CATEGORIES)
            accepted = tuple(rows[c]["matched"]["closest"] if rows[c]["matched"]["within_reference_radius"] else None
                             for c in CATEGORIES)
            panels.append({"seed": seed, "language": language,
                           "forced_rank_one": _exact_assignment(forced),
                           "accepted": _exact_assignment(accepted),
                           "distinct_accepted_categories": len(set(accepted) - {None})})
    return panels


def run_study(*, seed=17, train_steps=96, evaluations=8, sampling_seeds=(301,), spectral_weight=0.2):
    _configuration(seed, train_steps, evaluations, sampling_seeds, spectral_weight)
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    heldout = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, val_hashes = _audio_hashes(train.waves), _audio_hashes(heldout.waves)
    sources, noise, hashes = reference_panels()
    reference_hashes = {h for domain in hashes.values() for panel in domain.values() for h in panel}
    if train_hashes & val_hashes or reference_hashes & (train_hashes | val_hashes):
        raise ValueError("Exact waveform overlap across independent splits")
    if not torch.equal(heldout.waves[::2], heldout.waves[1::2]):
        raise ValueError("Invalid bilingual heldout ordering")
    probe = fit_probe(sources["fit"], sources["calibration"], noise["fit"], noise["calibration"])
    gate = evaluate_probe(probe, sources["holdout"], noise["holdout"])
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    interventions = CaptionInterventions.build(heldout.tokens, VALIDATION_SEEDS)
    rng = torch.Generator(device="cpu").manual_seed(seed+3)
    steps = torch.tensor([0, 0, 7, 7])
    clean = heldout.waves[:4]
    noisy = schedule.corrupt(clean, steps, torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0))
    arms = {}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        baseline = ReversibleBasisDenoiser("identity").eval()
        with torch.no_grad():
            initial_reference = baseline(noisy, steps, heldout.tokens[:4])
        unrotated_hash = state_sha256(baseline)
        for name in BASES:
            torch.random.default_generator.manual_seed(seed)
            model = ReversibleBasisDenoiser(name)
            if state_sha256(model) != unrotated_hash:
                raise ValueError("Unrotated model initialization differs")
            model.align_initial_function()
            alignment = initial_function_check(model, initial_reference, noisy, steps, heldout.tokens[:4])
            training = _train(model, train, schedule, seed=seed+1, steps=train_steps,
                              spectral_weight=spectral_weight)
            metrics = _heldout(model, heldout, schedule, seed=seed+2)
            counterfactual = heldout_interventions(model, heldout.waves, heldout.tokens, interventions,
                                                   schedule, noise_seed=seed+2)
            generated = None
            lexical = None
            if gate["gate"]["passed"]:
                generated = _generate(model, schedule, probe, evaluations=evaluations, sampling_seeds=sampling_seeds)
                generated["balanced_assignment"] = assignment_panels(generated)
                lexical = evaluate_caption_panel(model, schedule, probe, evaluations=evaluations,
                                                 seed=sampling_seeds[0])
            if state_sha256(model) != training["trained_state_sha256"]:
                raise ValueError("Evaluation changed model state")
            arms[name] = {"basis": model.describe(), "initial_function_check": alignment,
                          "training": training, "heldout": metrics, "heldout_interventions": counterfactual,
                          "generated": generated, "heldout_caption_panel": lexical}
    if len({arm["training"]["input_trace_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("Paired training inputs diverged")
    if len({arm["training"]["parameter_count"] for arm in arms.values()}) != 1:
        raise ValueError("Trainable parameter budget differs")
    return {
        "protocol": PROTOCOL, "configuration": {"seed": seed, "train_steps": train_steps,
            "evaluations": evaluations, "sampling_seeds": list(sampling_seeds), "spectral_weight": spectral_weight},
        "training_fingerprint": train.fingerprint, "validation_fingerprint": heldout.fingerprint,
        "unique_training_recordings": len(train_hashes), "unique_validation_recordings": len(val_hashes),
        "exact_split_overlap": False, "same_initial_unrotated_state": True,
        "aligned_initial_waveform_functions": True, "same_training_input_trace": True,
        "equal_trainable_parameter_counts": True, "coefficient_count_preserved": True,
        "adamw_coordinate_invariance_claimed": False,
        "reference_probe": gate, "arms": arms, "weights_or_audio_saved": False,
        "model_readiness_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=96)
    parser.add_argument("--evaluations", type=int, default=8)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301])
    parser.add_argument("--spectral-weight", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(run_study(seed=args.seed, train_steps=args.train_steps,
        evaluations=args.evaluations, sampling_seeds=tuple(args.sampling_seeds),
        spectral_weight=args.spectral_weight), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
