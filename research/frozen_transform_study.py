"""Gate a frozen-transform latent diffusion study on reconstruction fidelity.

Choose a 16-coefficient block transform on validation recordings only.
Require fresh source-preservation and artificial-noise controls before any
denoiser optimization. Compare native vs train-normalized continuous latents.
No audio, checkpoint, payload files or model weights leave memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.frozen_transform_latent import (
    METHODS, FrozenTransformCodec, latent_statistics, state_sha256,
)
from research.generated_source_grounding import LANGUAGES, REFERENCE_SEEDS, _summary
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, TextLatentDenoiser, synthetic_bilingual_waves,
)
from research.open_set_source_probe import (
    NOISE_FAMILIES, NOISE_SEEDS, evaluate_probe, fit_probe, noise_wave, reference_panels,
)
from research.quantized_transform_study import CONFIRMATION_SEEDS as QUANTIZED_CONFIRMATION_SEEDS
from research.text_audio import _synth_prompts
from research.transform_audio_codec import TransformFit
from research.transform_codec_study import FIT_SEEDS, _panel, score_decoded_panel
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_prediction import decode_velocity, velocity_target
from research.wave_diffusion import CATEGORIES, DiffusionSchedule

PROTOCOL = "sonumio-frozen-transform-latent-study-v1"
SOURCE_CONFIRMATION_SEEDS = (1510, 1511, 1512, 1513)
NOISE_CONFIRMATION_SEEDS = (1901, 1902, 1903, 1904)
MIN_CORRECT = 16


def _fingerprint(hashes):
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


@torch.no_grad()
def _encode(codec, waves):
    return torch.cat([codec.encode(batch) for batch in waves.split(8)])


@torch.no_grad()
def _reconstruct(codec, waves):
    tensor = torch.from_numpy(np.asarray(waves, dtype=np.float32).copy()).view(-1, 1, FRAMES)
    return torch.cat([codec.decode(codec.encode(batch)) for batch in tensor.split(8)])[:, 0].numpy()


def prepare(seed):
    # Fit/denoiser training share some recordings by design. Selection/heldout
    # also share some singles. No fit or training recording may enter either
    # evaluation domain; no confirmation or reference recording may enter either.
    fit_waves, fit_labels, fit_hashes = _panel(FIT_SEEDS)
    val_waves, val_labels, val_hashes = _panel(VALIDATION_SEEDS)
    confirm_waves, confirm_labels, confirm_hashes = _panel(SOURCE_CONFIRMATION_SEEDS)
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, validation_hashes = _audio_hashes(train.waves), _audio_hashes(validation.waves)
    sources, noises, hashes = reference_panels()
    reference_hashes = {h for domain in hashes.values() for panel in domain.values() for h in panel}
    new_noises = np.stack([
        [noise_wave(family, seed) for seed in NOISE_CONFIRMATION_SEEDS]
        for family in NOISE_FAMILIES
    ])
    new_noise_hashes = tuple(
        hashlib.sha256(wave.tobytes()).hexdigest() for wave in new_noises.reshape(-1, FRAMES)
    )
    groups = [
        train_hashes | set(fit_hashes),
        validation_hashes | set(val_hashes),
        set(confirm_hashes), reference_hashes, set(new_noise_hashes),
    ]
    if len(set(new_noise_hashes)) != len(new_noise_hashes) or any(
        a & b for i, a in enumerate(groups) for b in groups[i + 1:]
    ):
        raise ValueError("Exact recording overlap between training, validation or confirmation domains")
    all_seeds = (
        FIT_SEEDS + VALIDATION_SEEDS + SOURCE_CONFIRMATION_SEEDS + QUANTIZED_CONFIRMATION_SEEDS
        + NOISE_CONFIRMATION_SEEDS + tuple(
            value for domain in (REFERENCE_SEEDS, NOISE_SEEDS) for values in domain.values() for value in values
        )
    )
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Recording seed overlap")
    if (not torch.equal(train.waves[::2], train.waves[1::2])
            or not torch.equal(validation.waves[::2], validation.waves[1::2])):
        raise ValueError("Unexpected bilingual recording layout")
    fitted = TransformFit.fit(fit_waves, fit_labels)
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    reference = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    training_waves = train.waves[::2]
    selection = {}
    for method in METHODS:
        codec = FrozenTransformCodec(fitted, method, training_waves, calibrate=False)
        selection[method] = score_decoded_panel(
            val_waves, _reconstruct(codec, val_waves), val_labels, probe,
        )
    use_source_scores = reference["gate"]["passed"]
    def rank(method):
        row = selection[method]
        correct = row["reconstructed_correct_accepted"] if use_source_scores else 0
        coverage = sum(v["reconstructed_correct_accepted"] > 0 for v in row["by_category"].values()) if use_source_scores else 0
        return (-correct, -coverage, row["numerical"]["relative_squared_error"], method)
    selected = min(METHODS, key=rank)
    native = FrozenTransformCodec(fitted, selected, training_waves, calibrate=False)
    calibrated = FrozenTransformCodec(fitted, selected, training_waves, calibrate=True)
    reconstructed = _reconstruct(native, confirm_waves)
    normalized_reconstruction = _reconstruct(calibrated, confirm_waves)
    np.testing.assert_allclose(reconstructed, normalized_reconstruction, atol=2e-6, rtol=1e-5)
    confirmation = score_decoded_panel(confirm_waves, reconstructed, confirm_labels, probe)
    noise_rows = []
    for family, waves in zip(NOISE_FAMILIES, new_noises, strict=True):
        decoded = _reconstruct(native, waves)
        original = [probe.detailed_score(wave).source.accepted for wave in waves]
        reconstruction = [
            probe.detailed_score(np.clip(wave, -1, 1).astype(np.float32)).source.accepted
            for wave in decoded
        ]
        noise_rows.append({
            "family": family, "recordings": len(waves),
            "original_accepted": sum(original), "reconstructed_accepted": sum(reconstruction),
        })
    gate = {
        "reference_probe_passed": use_source_scores,
        "minimum_correct_accepted": MIN_CORRECT,
        "minimum_per_category": 1,
        "maximum_noise_acceptances": 0,
        "original_correct_accepted": confirmation["original_correct_accepted"],
        "reconstruction_correct_accepted": confirmation["reconstructed_correct_accepted"],
        "all_categories_preserved": all(v["reconstructed_correct_accepted"] >= 1 for v in confirmation["by_category"].values()),
        "original_noise_accepted": sum(row["original_accepted"] for row in noise_rows),
        "reconstructed_noise_accepted": sum(row["reconstructed_accepted"] for row in noise_rows),
    }
    gate["passed"] = (
        gate["reference_probe_passed"] and gate["original_correct_accepted"] >= MIN_CORRECT
        and gate["reconstruction_correct_accepted"] >= MIN_CORRECT
        and gate["all_categories_preserved"]
        and gate["original_noise_accepted"] == gate["reconstructed_noise_accepted"] == 0
    )
    report = {
        "selected_transform": selected, "selection_validation": selection,
        "confirmation": confirmation, "noise_confirmation": noise_rows,
        "gate": gate, "reference_probe": reference,
        "fit_fingerprint": _fingerprint(fit_hashes),
        "selection_fingerprint": _fingerprint(val_hashes),
        "confirmation_fingerprint": _fingerprint(confirm_hashes),
        "noise_confirmation_fingerprint": _fingerprint(new_noise_hashes),
        "training_fingerprint": train.fingerprint,
        "validation_fingerprint": validation.fingerprint,
        "confirmation_used_for_selection": False,
        "exact_cross_domain_overlap": False,
        "training_and_fit_may_share_recordings": True,
        "selection_and_validation_may_share_recordings": True,
    }
    return report, {"native": native, "calibrated": calibrated}, train, validation, probe


def train_denoiser(model, clean, tokens, schedule, *, seed, steps):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    trace = hashlib.sha256(b"sonumio-transform-latent-training-input-v1\0")
    initial = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    losses = []
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(tokens), (2,), generator=rng)
        step = torch.randint(schedule.steps, (2,), generator=rng)
        keep = torch.rand(2, generator=rng) >= 0.1
        audio = clean[indices]
        noise = torch.randn(audio.shape, generator=rng)
        for item in (indices, step, keep, noise):
            trace.update(item.numpy().tobytes())
        captions = torch.where(keep[:, None], tokens[indices], torch.zeros_like(tokens[indices]))
        optimizer.zero_grad(set_to_none=True)
        target = velocity_target(audio, noise, step, schedule)
        loss = F.mse_loss(model(schedule.corrupt(audio, step, noise), step, captions), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite transform latent training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {"initial_state_sha256": initial, "input_trace_sha256": trace.hexdigest(),
            "trained_state_sha256": state_sha256(model), "mean_loss": mean(losses),
            "first_loss": losses[0], "last_loss": losses[-1]}


@torch.no_grad()
def heldout(model, codec, data, schedule, *, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for level in (0, 4, 7):
        sums = {"matched_waveform_mae": 0.0, "null_waveform_mae": 0.0,
                "matched_velocity_mse": 0.0, "null_velocity_mse": 0.0}
        waveform_count, latent_count = 0, 0
        for offset in range(0, len(data.tokens), 4):
            wave = data.waves[offset:offset + 4]
            clean = codec.encode(wave)
            tokens = data.tokens[offset:offset + len(wave)]
            noise = torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0)
            step = torch.full((len(wave),), level, dtype=torch.long)
            noisy = schedule.corrupt(clean, step, noise)
            target = velocity_target(clean, noise, step, schedule)
            for kind, caption in (("matched", tokens), ("null", torch.zeros_like(tokens))):
                prediction = model(noisy, step, caption)
                restored, _ = decode_velocity(noisy, prediction, step, schedule)
                decoded = codec.decode(restored)
                sums[kind + "_waveform_mae"] += float(F.l1_loss(decoded, wave, reduction="sum"))
                sums[kind + "_velocity_mse"] += float(F.mse_loss(prediction, target, reduction="sum"))
            waveform_count += wave.numel()
            latent_count += clean.numel()
        result[str(level)] = {
            key: value / (waveform_count if "waveform" in key else latent_count)
            for key, value in sums.items()
        }
    return {"by_timestep": result, "unique_recordings": len(data.tokens) // 2,
            "caption_rows_per_timestep": len(data.tokens),
            "velocity_mse_across_coordinate_systems_not_directly_comparable": True}


@torch.no_grad()
def generate(model, codec, schedule, probe, *, evaluations, sampling_seeds):
    rows, checks = [], []
    for seed in sampling_seeds:
        settings = {"seed": seed, "evaluations": evaluations, "latent": True, "codec": codec}
        null = sample_velocity_ddim(model, schedule, _synth_prompts((0,), "en", seed), guidance=0, **settings)
        other = sample_velocity_ddim(model, schedule, _synth_prompts((1,), "hi", seed), guidance=0, **settings)
        if not torch.equal(null, other):
            raise ValueError("Null generation changed with caption")
        null_decision = probe.detailed_score(null.clamp(-1, 1))
        checks.append({"seed": seed, "caption_invariant": True})
        for language in LANGUAGES:
            for index, category in enumerate(CATEGORIES):
                waveform = sample_velocity_ddim(
                    model, schedule, _synth_prompts((index,), language, seed), guidance=1, **settings,
                )
                decision = probe.detailed_score(waveform.clamp(-1, 1))
                rows.append({
                    "seed": seed, "language": language, "category": category,
                    "matched": decision.as_dict(index),
                    "rotated": decision.as_dict((index + 1) % len(CATEGORIES)),
                    "null_guidance": null_decision.as_dict(index),
                    "unclamped_rms": float(waveform.square().mean().sqrt()),
                    "out_of_range_fraction": float((waveform.abs() > 1).float().mean()),
                })
    return {"cases": rows, "null_checks": checks, "by_language": {
        language: {
            condition: _summary([row[condition] for row in rows if row["language"] == language])
            for condition in ("matched", "rotated", "null_guidance")
        } for language in LANGUAGES
    }}


def run_study(*, seed=17, train_steps=96, evaluations=8, sampling_seeds=(301,)):
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 256
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(value) is not int or not 0 <= value < 2**32 for value in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)):
        raise ValueError("Invalid bounded frozen-transform study configuration")
    preparation, codecs, train, validation, probe = prepare(seed)
    arms = {}
    if preparation["gate"]["passed"]:
        schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
        with torch.random.fork_rng(devices=[]):
            for name, codec in codecs.items():
                before = state_sha256(codec)
                with torch.no_grad():
                    latents = _encode(codec, train.waves)
                torch.manual_seed(seed)
                model = TextLatentDenoiser(16)
                training = train_denoiser(model, latents, train.tokens, schedule, seed=seed+1, steps=train_steps)
                evaluation = heldout(model, codec, validation, schedule, seed=seed+2)
                generated = generate(model, codec, schedule, probe, evaluations=evaluations,
                                     sampling_seeds=sampling_seeds)
                if state_sha256(codec) != before or state_sha256(model) != training["trained_state_sha256"]:
                    raise ValueError("Evaluation mutated frozen codec or trained denoiser")
                arms[name] = {"codec": codec.describe(), "training": training, "heldout": evaluation,
                              "generated": generated}
        for key in ("initial_state_sha256", "input_trace_sha256"):
            if len({arm["training"][key] for arm in arms.values()}) != 1:
                raise ValueError("Paired latent ablation inputs diverged")
    return {
        "protocol": PROTOCOL,
        "configuration": {"seed": seed, "train_steps": train_steps, "evaluations": evaluations,
                          "sampling_seeds": list(sampling_seeds)},
        **preparation,
        "codec_diagnostics": {
            name: {"codec": codec.describe(), "training_latent_statistics": latent_statistics(codec, train.waves[::2]),
                   "validation_latent_statistics": latent_statistics(codec, validation.waves[::2])}
            for name, codec in codecs.items()
        },
        "status": "trained" if arms else "reconstruction_gate_failed",
        "arms": arms,
        "weights_or_audio_saved": False, "model_readiness_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=96)
    parser.add_argument("--evaluations", type=int, default=8)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301])
    args = parser.parse_args()
    print(json.dumps(run_study(seed=args.seed, train_steps=args.train_steps,
                               evaluations=args.evaluations, sampling_seeds=tuple(args.sampling_seeds)),
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
