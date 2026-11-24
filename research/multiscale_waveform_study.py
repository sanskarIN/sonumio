"""Compare raw waveform architectures and a spectral clean-estimate objective.

All models see the same synthetic recordings, captions, minibatches,
timesteps, noise, and condition dropout. The two multiscale arms share
initial parameters; the legacy arm has a different architecture and size.
No weights, audio files, external datasets, or release artifacts are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.generated_source_grounding import LANGUAGES, _summary
from research.heldout_caption_panel import evaluate_caption_panel
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves
from research.multiscale_waveform import MultiscaleWaveDenoiser, spectral_clean_loss
from research.open_set_source_probe import evaluate_probe, fit_probe, reference_panels
from research.text_audio import TextWaveDenoiser, _synth_prompts
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_prediction import decode_velocity, velocity_target
from research.wave_diffusion import CATEGORIES, DiffusionSchedule, SAMPLE_RATE

PROTOCOL = "sonumio-multiscale-waveform-study-v1"
ARMS = ("legacy_velocity", "multiscale_velocity", "multiscale_spectral")


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256(b"sonumio-multiscale-state-v1\0")
    for name, value in sorted(model.state_dict().items()):
        if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Unexpected model state")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train(model, data, schedule, *, seed, steps, spectral_weight):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    trace = hashlib.sha256(b"sonumio-multiscale-inputs-v1\0")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    initial = state_sha256(model)
    losses, spectral_losses = [], []
    for _ in range(steps):
        indices = torch.randint(len(data.tokens), (2,), generator=rng)
        timestep = torch.randint(schedule.steps, (2,), generator=rng)
        keep = torch.rand(2, generator=rng) >= 0.1
        clean = data.waves[indices]
        noise = torch.randn(clean.shape, generator=rng)
        for value in (indices, timestep, keep, noise):
            trace.update(value.numpy().tobytes())
        prompt = torch.where(keep[:, None], data.tokens[indices], torch.zeros_like(data.tokens[indices]))
        noisy = schedule.corrupt(clean, timestep, noise)
        target = velocity_target(clean, noise, timestep, schedule)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, timestep, prompt)
        base = F.mse_loss(prediction, target)
        auxiliary = (spectral_clean_loss(noisy, prediction, clean, timestep, schedule)
                     if spectral_weight else base.new_zeros(()))
        loss = base + spectral_weight * auxiliary
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training objective")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(base.detach()))
        spectral_losses.append(float(auxiliary.detach()))
    model.eval()
    return {
        "initial_state_sha256": initial,
        "trained_state_sha256": state_sha256(model),
        "input_trace_sha256": trace.hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "first_velocity_loss": losses[0], "last_velocity_loss": losses[-1],
        "mean_velocity_loss": mean(losses), "mean_spectral_auxiliary": mean(spectral_losses),
        "spectral_weight": spectral_weight,
    }


@torch.no_grad()
def _heldout(model, data, schedule, *, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for level in (0, schedule.steps // 2, schedule.steps - 1):
        sums = {"matched_velocity_mse": 0.0, "null_velocity_mse": 0.0,
                "matched_clean_mae": 0.0, "null_clean_mae": 0.0}
        count = 0
        for offset in range(0, len(data.tokens), 4):
            clean = data.waves[offset:offset + 4]
            tokens = data.tokens[offset:offset + len(clean)]
            # Both caption languages of a recording use identical noise.
            noise = torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0)
            step = torch.full((len(clean),), level, dtype=torch.long)
            noisy = schedule.corrupt(clean, step, noise)
            target = velocity_target(clean, noise, step, schedule)
            for kind, caption in (("matched", tokens), ("null", torch.zeros_like(tokens))):
                prediction = model(noisy, step, caption)
                restored, _ = decode_velocity(noisy, prediction, step, schedule)
                sums[kind + "_velocity_mse"] += float(F.mse_loss(prediction, target, reduction="sum"))
                sums[kind + "_clean_mae"] += float(F.l1_loss(restored, clean, reduction="sum"))
            count += clean.numel()
        result[str(level)] = {key: value / count for key, value in sums.items()}
    return {"by_timestep": result, "caption_rows_per_timestep": len(data.tokens),
            "unique_recordings": len(data.tokens) // 2}


@torch.no_grad()
def _generate(model, schedule, probe, *, evaluations, sampling_seeds):
    cases, checks = [], []
    for seed in sampling_seeds:
        kwargs = {"seed": seed, "evaluations": evaluations}
        a = sample_velocity_ddim(model, schedule, _synth_prompts((0,), "en", seed), guidance=0, **kwargs)
        b = sample_velocity_ddim(model, schedule, _synth_prompts((1,), "hi", seed), guidance=0, **kwargs)
        if not torch.equal(a, b):
            raise ValueError("Zero-guidance output depends on caption")
        null = probe.detailed_score(a.clamp(-1, 1))
        checks.append({"seed": seed, "caption_invariant": True})
        for language in LANGUAGES:
            for label, category in enumerate(CATEGORIES):
                caption = _synth_prompts((label,), language, seed)
                wave = sample_velocity_ddim(model, schedule, caption, guidance=1, **kwargs)
                clamped = wave.clamp(-1, 1)
                decision = probe.detailed_score(clamped)
                cases.append({
                    "seed": seed, "language": language, "category": category,
                    "matched": decision.as_dict(label),
                    "rotated": decision.as_dict((label + 1) % len(CATEGORIES)),
                    "null_guidance": null.as_dict(label),
                    "waveform_sha256": hashlib.sha256(clamped.numpy().tobytes()).hexdigest(),
                    "unclamped_rms": float(wave.square().mean().sqrt()),
                    "out_of_range_fraction": float((wave.abs() > 1).float().mean()),
                })
    return {"cases": cases, "null_checks": checks, "by_language": {
        language: {
            condition: _summary([row[condition] for row in cases if row["language"] == language])
            for condition in ("matched", "rotated", "null_guidance")
        } for language in LANGUAGES
    }}


def run_study(*, seed=17, train_steps=64, evaluations=8, sampling_seeds=(301,), spectral_weight=0.2):
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 256
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)
            or type(spectral_weight) not in (int, float) or not 0 < spectral_weight <= 2):
        raise ValueError("Invalid bounded multiscale study configuration")
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, val_hashes = _audio_hashes(train.waves), _audio_hashes(validation.waves)
    source, noise, hashes = reference_panels()
    reference_hashes = {h for domain in hashes.values() for split in domain.values() for h in split}
    if train_hashes & val_hashes or reference_hashes.intersection(train_hashes | val_hashes):
        raise ValueError("Exact audio overlap between study splits")
    if (not torch.equal(train.waves[::2], train.waves[1::2])
            or not torch.equal(validation.waves[::2], validation.waves[1::2])):
        raise ValueError("Expected bilingual recording pairs")
    probe = fit_probe(source["fit"], source["calibration"], noise["fit"], noise["calibration"])
    holdout = evaluate_probe(probe, source["holdout"], noise["holdout"])
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    arms = {}
    with torch.random.fork_rng(devices=[]):
        for name in ARMS:
            torch.manual_seed(seed)
            model = TextWaveDenoiser(16) if name == "legacy_velocity" else MultiscaleWaveDenoiser()
            training = _train(
                model, train, schedule, seed=seed + 1, steps=train_steps,
                spectral_weight=spectral_weight if name == "multiscale_spectral" else 0.0,
            )
            heldout = _heldout(model, validation, schedule, seed=seed + 2)
            generated = (_generate(model, schedule, probe, evaluations=evaluations, sampling_seeds=sampling_seeds)
                         if holdout["gate"]["passed"] else None)
            lexical = (evaluate_caption_panel(model, schedule, probe, evaluations=evaluations,
                                              seed=sampling_seeds[0])
                       if holdout["gate"]["passed"] else None)
            if state_sha256(model) != training["trained_state_sha256"]:
                raise ValueError("Evaluation mutated parameters")
            arms[name] = {"training": training, "heldout": heldout, "generated": generated,
                          "heldout_caption_panel": lexical}
    if len({arm["training"]["input_trace_sha256"] for arm in arms.values()}) != 1:
        raise ValueError("Paired training inputs diverged")
    if (arms["multiscale_velocity"]["training"]["initial_state_sha256"]
            != arms["multiscale_spectral"]["training"]["initial_state_sha256"]):
        raise ValueError("Spectral ablation initialization diverged")
    return {
        "protocol": PROTOCOL,
        "configuration": {"seed": seed, "train_steps": train_steps, "evaluations": evaluations,
                          "sampling_seeds": list(sampling_seeds), "spectral_weight": spectral_weight,
                          "frames": FRAMES, "sample_rate": SAMPLE_RATE},
        "training_fingerprint": train.fingerprint, "validation_fingerprint": validation.fingerprint,
        "unique_training_recordings": len(train_hashes), "unique_validation_recordings": len(val_hashes),
        "same_training_input_trace": True, "same_multiscale_initialization": True,
        "equal_parameter_comparison": False, "reference_probe": holdout, "arms": arms,
        "weights_or_audio_saved": False, "model_readiness_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--evaluations", type=int, default=8)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301])
    parser.add_argument("--spectral-weight", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(run_study(
        seed=args.seed, train_steps=args.train_steps, evaluations=args.evaluations,
        sampling_seeds=tuple(args.sampling_seeds), spectral_weight=args.spectral_weight,
    ), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
