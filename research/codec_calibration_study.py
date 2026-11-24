"""Paired codec and latent-calibration experiments with actual DDIM generation.

Compare legacy uncalibrated, legacy calibrated, and residual calibrated
representations. The two legacy arms share exactly the same frozen codec.
All denoisers start with identical weights and see identical caption indices,
timesteps, noise tensors and classifier-free dropout. The residual codec
has a different parameter count; this is not an equal-capacity comparison.
Only first-party synthetic recordings and in-memory weights are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from platform import python_version
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.calibrated_audio_codec import (
    CalibratedCodec, LatentCalibration, ResidualAudioCodec, latent_statistics,
    reconstruction_diagnostics,
)
from research.generated_source_grounding import LANGUAGES, REFERENCE_SEEDS, _summary
from research.latent_audio import (
    FRAMES, LATENT_CHANNELS, LATENT_FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS,
    TextLatentDenoiser, TinyAudioAutoencoder, reconstruction_loss,
    synthetic_bilingual_waves,
)
from research.open_set_source_probe import evaluate_probe, fit_probe, reference_panels
from research.text_audio import SYNTHETIC_COMBINATIONS, _synth_prompts
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_prediction import velocity_target
from research.wave_diffusion import CATEGORIES, SAMPLE_RATE, DiffusionSchedule

PROTOCOL = "sonumio-codec-latent-calibration-study-v1"
ARMS = ("legacy_native", "legacy_calibrated", "residual_calibrated")


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256(b"sonumio-codec-calibration-state-v1\0")
    for name, value in sorted(module.state_dict().items()):
        if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Unexpected research module state")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_codec(codec: nn.Module, unique: torch.Tensor, *, seed: int, steps: int) -> dict:
    rng = torch.Generator().manual_seed(seed)
    trace = hashlib.sha256(b"sonumio-codec-minibatch-v1\0")
    codec.train()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=0.001)
    losses = []
    for _ in range(steps):
        indices = torch.randint(len(unique), (2,), generator=rng)
        trace.update(indices.numpy().tobytes())
        batch = unique[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = reconstruction_loss(batch, codec(batch))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite codec training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(codec.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    codec.eval().requires_grad_(False)
    return {
        "parameter_count": sum(p.numel() for p in codec.parameters()),
        "training_input_trace_sha256": trace.hexdigest(),
        "first_loss": losses[0], "last_loss": losses[-1],
        "mean_loss": mean(losses),
        "trained_state_sha256": state_sha256(codec),
    }


@torch.no_grad()
def _encoded(codec, waves: torch.Tensor) -> torch.Tensor:
    return torch.cat([codec.encode(batch) for batch in waves.split(8)])


def _average(rows: list[dict]) -> dict:
    result = {"recordings": len(rows)}
    for name in rows[0]:
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = mean(values) if values else None
    return result


@torch.no_grad()
def _codec_evaluation(codec, validation: torch.Tensor, source_holdout: np.ndarray, probe) -> dict:
    unique = validation[::2]
    rows = []
    for batch in unique.split(8):
        rows.extend(reconstruction_diagnostics(batch, codec.decode(codec.encode(batch))))
    group_size = len(VALIDATION_SEEDS)
    if len(rows) != len(SYNTHETIC_COMBINATIONS) * group_size:
        raise ValueError("Unexpected validation recording order")
    by_cardinality = {
        str(cardinality): _average([
            rows[i * group_size + j]
            for i, members in enumerate(SYNTHETIC_COMBINATIONS)
            if len(members) == cardinality for j in range(group_size)
        ])
        for cardinality in (1, 2)
    }
    oracle = []
    for label, category in enumerate(CATEGORIES):
        for seed, audio in zip(REFERENCE_SEEDS["holdout"], source_holdout[label], strict=True):
            wave = torch.from_numpy(audio.copy()).view(1, 1, FRAMES)
            reconstructed = codec.decode(codec.encode(wave))[0, 0]
            oracle.append({
                "category": category, "recording_seed": seed,
                "original": probe.detailed_score(wave[0, 0]).as_dict(label),
                "reconstruction": probe.detailed_score(reconstructed).as_dict(label),
                "fidelity": reconstruction_diagnostics(wave, reconstructed.view(1, 1, FRAMES))[0],
            })
    def accepted(key):
        return sum(row[key]["within_reference_radius"] and row[key]["target_rank"] == 1 for row in oracle)
    return {
        "validation_unique_recordings": len(rows),
        "validation_metrics": _average(rows),
        "validation_by_source_count": by_cardinality,
        "source_holdout_oracle": {
            "original_correct_accepted": accepted("original"),
            "reconstructed_correct_accepted": accepted("reconstruction"),
            "cases": oracle,
        },
    }


def _train_denoiser(
    model: nn.Module, train_latents: torch.Tensor, tokens: torch.Tensor,
    schedule: DiffusionSchedule, *, seed: int, steps: int,
) -> dict:
    model.train()
    rng = torch.Generator().manual_seed(seed)
    trace = hashlib.sha256(b"sonumio-latent-denoising-trace-v1\0")
    initial = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    losses = []
    for _ in range(steps):
        indices = torch.randint(len(tokens), (2,), generator=rng)
        timestep = torch.randint(schedule.steps, (len(indices),), generator=rng)
        keep = torch.rand(len(indices), generator=rng) >= 0.1
        clean = train_latents[indices]
        noise = torch.randn(clean.shape, generator=rng)
        for value in (indices, timestep, keep, noise):
            trace.update(value.numpy().tobytes())
        prompt = torch.where(keep[:, None], tokens[indices], torch.zeros_like(tokens[indices]))
        noisy = schedule.corrupt(clean, timestep, noise)
        target = velocity_target(clean, noise, timestep, schedule)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(noisy, timestep, prompt), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite latent denoising loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {
        "initial_state_sha256": initial,
        "training_input_trace_sha256": trace.hexdigest(),
        "trained_state_sha256": state_sha256(model),
        "first_loss": losses[0], "last_loss": losses[-1],
        "mean_loss": mean(losses),
    }


@torch.no_grad()
def _heldout_objective(model, source, tokens, schedule, *, seed: int) -> dict:
    rng = torch.Generator().manual_seed(seed)
    rows = {}
    for timestep in (0, schedule.steps // 2, schedule.steps - 1):
        errors = {"matched": 0.0, "null": 0.0}
        count = 0
        for offset in range(0, len(tokens), 4):
            clean = source[offset:offset + 4]
            prompt = tokens[offset:offset + len(clean)]
            # Bilingual rows share clean latents and their corruption noise.
            noise = torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0)
            steps = torch.full((len(clean),), timestep, dtype=torch.long)
            noisy = schedule.corrupt(clean, steps, noise)
            target = velocity_target(clean, noise, steps, schedule)
            errors["matched"] += float(F.mse_loss(model(noisy, steps, prompt), target, reduction="sum"))
            errors["null"] += float(F.mse_loss(model(noisy, steps, torch.zeros_like(prompt)), target, reduction="sum"))
            count += target.numel()
        rows[str(timestep)] = {key + "_mse": value / count for key, value in errors.items()}
    return {
        "by_timestep": rows, "caption_cases_per_timestep": len(tokens),
        "independent_recordings": len(tokens) // 2,
        "cross_representation_mse_not_directly_comparable": True,
    }


@torch.no_grad()
def _generate(model, codec, schedule, probe, *, evaluations, sampling_seeds) -> dict:
    cases, checks = [], []
    for seed in sampling_seeds:
        common = {"seed": seed, "evaluations": evaluations, "latent": True, "codec": codec}
        a = sample_velocity_ddim(model, schedule, _synth_prompts((0,), "en", seed), guidance=0, **common)
        b = sample_velocity_ddim(model, schedule, _synth_prompts((1,), "hi", seed), guidance=0, **common)
        if not torch.equal(a, b):
            raise ValueError("Unconditional waveform changed with caption")
        null = probe.detailed_score(a)
        checks.append({"seed": seed, "caption_invariant": True})
        for language in LANGUAGES:
            for label, category in enumerate(CATEGORIES):
                caption = _synth_prompts((label,), language, seed)
                wave = sample_velocity_ddim(model, schedule, caption, guidance=1, **common)
                decision = probe.detailed_score(wave)
                cases.append({
                    "seed": seed, "language": language, "category": category,
                    "matched": decision.as_dict(label),
                    "rotated": decision.as_dict((label + 1) % len(CATEGORIES)),
                    "null_guidance": null.as_dict(label),
                    "waveform_sha256": hashlib.sha256(wave.numpy().tobytes()).hexdigest(),
                    "rms": float(wave.square().mean().sqrt()),
                    "clipping_fraction": float((wave.abs() >= 0.999).float().mean()),
                })
    return {
        "cases": cases, "null_checks": checks,
        "by_language": {
            language: {
                condition: _summary([row[condition] for row in cases if row["language"] == language])
                for condition in ("matched", "rotated", "null_guidance")
            } for language in LANGUAGES
        },
    }


def run_study(
    *, seed: int = 17, codec_steps: int = 32, train_steps: int = 32,
    evaluations: int = 4, sampling_seeds: tuple[int, ...] = (301,),
) -> dict:
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(codec_steps) is not int or not 1 <= codec_steps <= 128
            or type(train_steps) is not int or not 1 <= train_steps <= 128
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)):
        raise ValueError("Invalid bounded codec study configuration")
    train, validation = synthetic_bilingual_waves(TRAIN_SEEDS), synthetic_bilingual_waves(VALIDATION_SEEDS)
    train_hashes, val_hashes = _audio_hashes(train.waves), _audio_hashes(validation.waves)
    sources, noises, hashes = reference_panels()
    references = {h for domain in hashes.values() for panel in domain.values() for h in panel}
    if train_hashes & val_hashes or references.intersection(train_hashes | val_hashes):
        raise ValueError("Exact waveform leakage across study splits")
    if not torch.equal(train.waves[::2], train.waves[1::2]) or not torch.equal(validation.waves[::2], validation.waves[1::2]):
        raise ValueError("Expected paired bilingual recordings")
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    holdout = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    report = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        legacy = TinyAudioAutoencoder()
        torch.manual_seed(seed)
        residual = ResidualAudioCodec()
        codec_reports = {
            name: train_codec(codec, train.waves[::2], seed=seed + 1, steps=codec_steps)
            for name, codec in (("legacy", legacy), ("residual", residual))
        }
        if codec_reports["legacy"]["training_input_trace_sha256"] != codec_reports["residual"]["training_input_trace_sha256"]:
            raise ValueError("Codec minibatch trace diverged")
        legacy_calibration = LatentCalibration.fit(_encoded(legacy, train.waves[::2]))
        residual_calibration = LatentCalibration.fit(_encoded(residual, train.waves[::2]))
        variants = (
            ("legacy_native", legacy, None),
            ("legacy_calibrated", legacy, legacy_calibration),
            ("residual_calibrated", residual, residual_calibration),
        )
        for name, native, calibration in variants:
            codec = CalibratedCodec(native, calibration)
            codec_hash = state_sha256(codec)
            train_latents, validation_latents = _encoded(codec, train.waves), _encoded(codec, validation.waves)
            torch.manual_seed(seed + 2)
            model = TextLatentDenoiser(16)
            denoiser = _train_denoiser(model, train_latents, train.tokens, schedule,
                                       seed=seed + 3, steps=train_steps)
            reconstruction = _codec_evaluation(codec, validation.waves, sources["holdout"], probe)
            objective = _heldout_objective(model, validation_latents, validation.tokens, schedule, seed=seed + 4)
            generated = (_generate(model, codec, schedule, probe, evaluations=evaluations,
                                   sampling_seeds=sampling_seeds) if holdout["gate"]["passed"] else None)
            if state_sha256(model) != denoiser["trained_state_sha256"] or state_sha256(codec) != codec_hash:
                raise ValueError("Evaluation mutated frozen model or codec")
            report[name] = {
                "native_codec_sha256": state_sha256(native),
                "calibration": calibration.describe() if calibration else None,
                "training_latent_statistics": latent_statistics(train_latents[::2]),
                "validation_latent_statistics": latent_statistics(validation_latents[::2]),
                "reconstruction": reconstruction, "denoising": denoiser,
                "heldout_velocity": objective, "generated": generated,
            }
    for key in ("initial_state_sha256", "training_input_trace_sha256"):
        if len({report[name]["denoising"][key] for name in ARMS}) != 1:
            raise ValueError("Paired denoising control diverged")
    if report["legacy_native"]["native_codec_sha256"] != report["legacy_calibrated"]["native_codec_sha256"]:
        raise ValueError("Legacy codec parameters differ across calibration arms")
    return {
        "protocol": PROTOCOL,
        "runtime": {"python": python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "configuration": {
            "seed": seed, "codec_steps": codec_steps, "train_steps": train_steps,
            "evaluations": evaluations, "sampling_seeds": list(sampling_seeds),
            "sample_rate": SAMPLE_RATE, "waveform_frames": FRAMES,
            "latent_shape": [LATENT_CHANNELS, LATENT_FRAMES],
            "scalar_compression_ratio": FRAMES / (LATENT_CHANNELS * LATENT_FRAMES),
        },
        "training_fingerprint": train.fingerprint, "validation_fingerprint": validation.fingerprint,
        "unique_training_recordings": len(train_hashes), "unique_validation_recordings": len(val_hashes),
        "exact_split_overlap": False, "same_denoiser_initialization_and_noise": True,
        "codec_training": codec_reports, "reference_probe": holdout, "arms": report,
        "weights_or_audio_saved": False, "model_readiness_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--codec-steps", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301])
    args = parser.parse_args()
    print(json.dumps(run_study(
        seed=args.seed, codec_steps=args.codec_steps, train_steps=args.train_steps,
        evaluations=args.evaluations, sampling_seeds=tuple(args.sampling_seeds),
    ), allow_nan=False, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
