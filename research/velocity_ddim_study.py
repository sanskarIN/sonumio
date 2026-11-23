"""Controlled v-trained, same-initial-noise DDIM budget study (CPU only).

Train a tiny raw and latent velocity denoiser on original Sonumio DSP audio,
then compare deterministic DDIM inference at reduced and full timestep
budgets. Training, encoder, sampler, PCM and checkpoints remain experimental;
this does NOT imply usable sound, text understanding, or Sonumio-1 readiness.

No user recordings or pretrained weights; no audio, WAV, model weights or
checkpoints leave memory. CLI prints *numeric JSON diagnostics only*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from statistics import mean
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.generation_audit import _compare_waveforms, PAIRS
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, LatentResearchModel,
    synthetic_bilingual_waves, reconstruction_loss,
)
from research.text_audio import TextWaveDenoiser
from research.velocity_ddim import ddim_plan, sample_velocity_ddim
from research.velocity_prediction import velocity_target
from research.velocity_ablation import _record_unclipped
from research.wave_diffusion import DiffusionSchedule

PROTOCOL = "sonumio-velocity-sampler-budget-study-v1"
_PROMPTS = tuple(
    (language, category, caption)
    for en, hi, category in PAIRS
    for language, caption in (("en", en), ("hi", hi))
)


def _settings(*, seed: int, train_steps: int, codec_steps: int,
              schedule_steps: int, channels: int, budgets: tuple[int, ...],
              sample_seeds: tuple[int, ...], guidance: float) -> None:
    """Bound cost and reject misleading or invalid sampling configurations."""
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 64
            or type(codec_steps) is not int or not 1 <= codec_steps <= 64
            or type(schedule_steps) is not int or not 8 <= schedule_steps <= 32
            or type(channels) is not int or channels != 16):
        raise ValueError("Invalid bounded v-DDIM synthetic research training config")
    if (type(budgets) is not tuple or not 2 <= len(budgets) <= 4
            or any(type(b) is not int for b in budgets)
            or len(set(budgets)) != len(budgets)
            or tuple(sorted(budgets)) != budgets
            or budgets[-1] != schedule_steps
            or any(not 2 <= b <= schedule_steps for b in budgets)):
        raise ValueError("Budgets must be 2..4 strictly increasing steps ending at T")
    if (type(sample_seeds) is not tuple
            or not 1 <= len(sample_seeds) <= 3
            or len(set(sample_seeds)) != len(sample_seeds)
            or any(type(s) is not int or not 0 <= s < 2**32
                   for s in sample_seeds)):
        raise ValueError("Use 1..3 unique unsigned sampling seeds")
    if (type(guidance) not in (float, int) or not math.isfinite(guidance)
            or not 0 <= guidance <= 4):
        raise ValueError("Unsupported v-DDIM guidance value")


def _train_models(*, seed: int, train_steps: int, codec_steps: int,
                  schedule: DiffusionSchedule, channels: int):
    """Train two v-objective arms without exposing weights or persistent data."""
    train = synthetic_bilingual_waves(TRAIN_SEEDS)
    validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
    training_hashes, validation_hashes = (
        _audio_hashes(train.waves), _audio_hashes(validation.waves)
    )
    if (training_hashes & validation_hashes
            or train.fingerprint == validation.fingerprint):
        raise ValueError("Synthetic train/held-out waveform leakage")
    raw = TextWaveDenoiser(channels)
    latent = LatentResearchModel(channels)
    codec = latent.codec
    index_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    codec_optim = torch.optim.AdamW(codec.parameters(), lr=0.001)
    unique = train.waves[::2]
    codec.train()
    codec_losses = []
    for _ in range(codec_steps):
        index = torch.randint(len(unique), (2,), generator=index_rng)
        codec_optim.zero_grad(set_to_none=True)
        loss = reconstruction_loss(unique[index], codec(unique[index]))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite codec reconstruction loss")
        loss.backward()
        nn.utils.clip_grad_norm_(codec.parameters(), 1)
        codec_optim.step()
        codec_losses.append(float(loss.detach()))
    codec.eval()
    codec.requires_grad_(False)

    with torch.no_grad():
        latents = torch.cat([
            codec.encode(batch) for batch in train.waves.split(8)
        ])
        heldout_latents = torch.cat([
            codec.encode(batch) for batch in validation.waves.split(8)
        ])
        reconstruction_sum = sum(
            F.l1_loss(codec(batch), batch, reduction="sum").item()
            for batch in validation.waves.split(8)
        )
        reconstruction_mae = float(
            reconstruction_sum / validation.waves.numel()
        )

    optimizers = {
        "raw": torch.optim.AdamW(raw.parameters(), lr=0.0005),
        "latent": torch.optim.AdamW(latent.denoiser.parameters(), lr=0.0005),
    }
    models = {"raw": raw, "latent": latent.denoiser}
    losses = {"raw": [], "latent": []}
    for _ in range(train_steps):
        indices = torch.randint(len(train.waves), (2,), generator=index_rng)
        timestep = torch.randint(
            schedule.steps, (len(indices),), generator=index_rng,
        )
        keep = torch.rand(len(indices), generator=index_rng) >= 0.1
        tokens = torch.where(
            keep[:, None], train.tokens[indices],
            torch.zeros_like(train.tokens[indices]),
        )
        for arm in ("raw", "latent"):
            clean = train.waves[indices] if arm == "raw" else latents[indices]
            noise = torch.randn(clean.shape, generator=index_rng)
            xt = schedule.corrupt(clean, timestep, noise)
            target = velocity_target(clean, noise, timestep, schedule)
            optimizers[arm].zero_grad(set_to_none=True)
            estimate = models[arm](xt, timestep, tokens)
            loss = F.mse_loss(estimate, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite velocity denoising loss")
            loss.backward()
            nn.utils.clip_grad_norm_(models[arm].parameters(), 1)
            optimizers[arm].step()
            losses[arm].append(float(loss.detach()))

    # Complete validation split (42 source recordings x 2 bilingual captions):
    # do NOT infer independent examples from duplicated underlying waveforms.
    # Uses fresh RNG independent of optimization and per-arm minibatch bounds.
    # This v-objective MSE is a noise-prediction diagnostic, NOT audio quality.
    heldout_losses: dict[str, float] = {}
    eval_rng = torch.Generator(device="cpu").manual_seed(seed + 171)
    for arm in ("raw", "latent"):
        models[arm].eval()
        source = validation.waves if arm == "raw" else heldout_latents
        error_sum, element_count = 0.0, 0
        with torch.no_grad():
            for offset in range(0, len(source), 4):
                clean = source[offset:offset + 4]
                tokens = validation.tokens[offset:offset + len(clean)]
                timestep = torch.randint(
                    schedule.steps, (len(clean),), generator=eval_rng,
                )
                noise = torch.randn(clean.shape, generator=eval_rng)
                prediction = models[arm](
                    schedule.corrupt(clean, timestep, noise), timestep, tokens,
                )
                target = velocity_target(clean, noise, timestep, schedule)
                error_sum += float(F.mse_loss(
                    prediction, target, reduction="sum",
                ))
                element_count += target.numel()
        if element_count < 1:
            raise ValueError("Empty held-out velocity evaluation")
        heldout_losses[arm] = error_sum / element_count
        if not math.isfinite(heldout_losses[arm]):
            raise ValueError("Nonfinite held-out objective diagnostic")
    report = {
        "training_fingerprint": train.fingerprint,
        "validation_fingerprint": validation.fingerprint,
        "unique_training_waveforms": len(training_hashes),
        "unique_heldout_waveforms": len(validation_hashes),
        "bilingual_caption_examples_per_split": train.examples,
        "independent_recordings_per_split": len(training_hashes),
        "heldout_captions_scored_for_velocity": validation.examples,
        "heldout_recordings_scored_for_velocity": len(validation_hashes),
        "heldout_metric_scope": "all captions; source pairs share waveforms",
        "training_loss_initial": {arm: losses[arm][0] for arm in losses},
        "training_loss_final": {arm: losses[arm][-1] for arm in losses},
        "heldout_velocity_mse": heldout_losses,
        "heldout_codec_waveform_mae": reconstruction_mae,
        "codec_first_loss": codec_losses[0],
        "codec_last_loss": codec_losses[-1],
        "train_and_validation_exact_overlap": False,
    }
    return models, codec, report


def _evaluate(models: dict, codec: nn.Module,
              schedule: DiffusionSchedule, *,
              budgets: tuple[int, ...], sample_seeds: tuple[int, ...],
              guidance: float) -> tuple[list[dict], list[dict]]:
    """Evaluate identical trained weights, captions, and initial noise."""
    records: list[dict] = []
    contrasts: list[dict] = []
    full = budgets[-1]
    for arm in ("raw", "latent"):
        model = models[arm]
        latent = arm == "latent"
        for seed in sample_seeds:
            # Generate the full-step deterministic control ONCE, reused by
            # every reduced-budget comparison. No stochastic DDPM reference.
            references = {}
            for language, category, caption in _PROMPTS:
                generated = sample_velocity_ddim(
                    model, schedule, caption, evaluations=full,
                    seed=seed, guidance=guidance, latent=latent,
                    codec=codec if latent else None,
                    return_preactivation=latent,
                )
                references[(category, language)] = generated
            for budget in budgets:
                outputs = {}
                for language, category, caption in _PROMPTS:
                    reference = references[(category, language)]
                    generation = (
                        reference if budget == full else sample_velocity_ddim(
                            model, schedule, caption, evaluations=budget,
                            seed=seed, guidance=guidance, latent=latent,
                            codec=codec if latent else None,
                            return_preactivation=latent,
                        )
                    )
                    waveform, pre = (
                        generation if isinstance(generation, tuple)
                        else (generation, None)
                    )
                    reference_wave = (
                        reference[0] if isinstance(reference, tuple) else reference
                    )
                    health = _record_unclipped(
                        waveform, decoder_preactivation=pre,
                    )
                    post = waveform.clamp(-1, 1)
                    full_post = reference_wave.clamp(-1, 1)
                    delta = _compare_waveforms(post, full_post)
                    if budget == full and not delta["exact_same_waveform"]:
                        raise ValueError("Full-step DDIM control does not reproduce itself")
                    outputs[(category, language)] = post
                    records.append({
                        "arm": arm, "budget": budget, "seed": seed,
                        "category": category, "language": language,
                        "sampler_nfe": ddim_plan(
                            schedule, evaluations=budget,
                        ).as_dict(guidance=guidance)["model_evaluations"],
                        "peak_before_external_limiter": health["preclip_peak"],
                        "rms_before_external_limiter": health["preclip_rms"],
                        "out_of_range_fraction": health["preclip_out_of_range_fraction"],
                        "postclamp_rms": health["postclip"]["rms"],
                        "postclamp_clipping_fraction": health["postclip"]["clipping_fraction"],
                        "postclamp_sha256": health["postclip"]["sha256"],
                        "mae_to_full_step_ddim": delta["mean_absolute_difference"],
                        "cosine_to_full_step_ddim": delta["waveform_cosine"],
                        "identical_to_full_step": delta["exact_same_waveform"],
                        **({"latent_pretanh_saturation_fraction":
                            health["codec_tanh_saturation_fraction"]}
                           if latent else {}),
                    })
                for language in ("en", "hi"):
                    difference = _compare_waveforms(
                        outputs[("rain", language)],
                        outputs[("wind", language)],
                    )
                    contrasts.append({
                        "arm": arm, "budget": budget, "seed": seed,
                        "language": language,
                        "rain_vs_wind_waveform_difference": difference,
                        "semantic_sound_recognition_evaluated": False,
                    })
    return records, contrasts


def _summary(records: list[dict], *, budgets: tuple[int, ...]) -> dict:
    results = {}
    for arm in ("raw", "latent"):
        results[arm] = {}
        for budget in budgets:
            selected = [
                row for row in records
                if row["arm"] == arm and row["budget"] == budget
            ]
            if not selected:
                raise ValueError("Sampler study summary lacks expected cases")
            results[arm][str(budget)] = {
                "samples": len(selected),
                "model_evaluations_per_sample": selected[0]["sampler_nfe"],
                "mean_mae_to_full_step": mean(
                    row["mae_to_full_step_ddim"] for row in selected
                ),
                "mean_clipping_fraction": mean(
                    row["postclamp_clipping_fraction"] for row in selected
                ),
                "mean_rms": mean(row["postclamp_rms"] for row in selected),
                "mean_out_of_range_fraction": mean(
                    row["out_of_range_fraction"] for row in selected
                ),
                "unique_sample_hashes": len({
                    row["postclamp_sha256"] for row in selected
                }),
            }
    return results


def run_study(*, seed: int = 17, train_steps: int = 4,
              codec_steps: int = 2, schedule_steps: int = 8,
              channels: int = 16, budgets: tuple[int, ...] = (2, 4, 8),
              sample_seeds: tuple[int, ...] = (201, 202),
              guidance: float = 1.0) -> dict:
    """Actual training plus deterministic inference, no external artifacts."""
    _settings(
        seed=seed, train_steps=train_steps, codec_steps=codec_steps,
        schedule_steps=schedule_steps, channels=channels, budgets=budgets,
        sample_seeds=sample_seeds, guidance=guidance,
    )
    started = perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        schedule = DiffusionSchedule(
            schedule_steps, profile="cosine_zero_terminal_v1",
        )
        models, codec, diagnostics = _train_models(
            seed=seed, train_steps=train_steps, codec_steps=codec_steps,
            schedule=schedule, channels=channels,
        )
        records, contrasts = _evaluate(
            models, codec, schedule, budgets=budgets,
            sample_seeds=sample_seeds, guidance=guidance,
        )
    report = {
        "protocol": PROTOCOL,
        "configuration": {
            "seed": seed, "train_steps": train_steps,
            "codec_steps": codec_steps,
            "schedule_steps": schedule_steps, "channels": channels,
            "budgets": list(budgets), "sample_seeds": list(sample_seeds),
            "guidance": guidance,
            "objective": "velocity", "sampler": "deterministic_ddim_eta_zero",
            "sample_rate": 8000, "frames": FRAMES,
            "training_corpus": "sonumio-first-party-synthetic-dsp",
        },
        "training": diagnostics,
        "timesteps": {
            str(budget): ddim_plan(schedule, evaluations=budget).as_dict(
                guidance=guidance
            ) for budget in budgets
        },
        "samples": records,
        "prompt_change_controls": contrasts,
        "by_arm_budget": _summary(records, budgets=budgets),
        "sample_count": len(records),
        "sampler_wall_seconds_including_training": round(
            perf_counter() - started, 3,
        ),
        "same_starting_gaussian_for_equal_seeds": True,
        "same_trained_parameters_across_budgets": True,
        "heldout_audio_used_for_training": False,
        "weights_written": False,
        "audio_files_written": False,
        "release_authorized": False,
        "human_listening_or_prompt_semantics_evaluated": False,
        "limitations": [
            "Relative full-step DDIM error is numerical trajectory difference, not quality",
            "Cross-domain raw/latent v-MSE is not directly comparable",
            "Model may not converge with these short synthetic training budgets",
            "Only 8 kHz 1.024-second synthetic sound effects, no speech or music",
            "No independent listening, real-world licensing or semantic correctness",
        ],
    }
    # Deterministic configuration identity excludes variable runtime timings.
    report["configuration_sha256"] = hashlib.sha256(
        json.dumps(report["configuration"], sort_keys=True,
                   separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if any(type(x) is float and not math.isfinite(x)
           for row in records for x in row.values()):
        raise ValueError("Nonfinite generated-audio experiment result")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=4)
    parser.add_argument("--codec-steps", type=int, default=2)
    parser.add_argument("--schedule-steps", type=int, default=8)
    parser.add_argument("--budgets", default="2,4,8")
    parser.add_argument("--sample-seeds", default="201,202")
    parser.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()
    budgets = tuple(int(s) for s in args.budgets.split(","))
    seeds = tuple(int(s) for s in args.sample_seeds.split(","))
    result = run_study(
        seed=args.seed, train_steps=args.train_steps,
        codec_steps=args.codec_steps, schedule_steps=args.schedule_steps,
        budgets=budgets, sample_seeds=seeds, guidance=args.guidance,
    )
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
