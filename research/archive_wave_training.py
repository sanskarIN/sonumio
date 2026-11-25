"""Bounded CPU waveform training from a private, reviewed corpus archive.

Requires an independently retained archive SHA256 and passed numerical
readiness. This is an experimental single-class waveform DDPM, not
Sonumio-1. No weights or audio files are written by default. Explicit
checkpoint outputs are private local numeric training state.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from research.archive_readiness import audit_archive
from research.archive_sampler import CounterGroupSampler
from research.archive_training_contracts import (
    ArchiveTrainingConfig, TrainingProgress, make_archive_optimizer, runtime_identity,
)
from research.archive_checkpoint import (
    CheckpointOptions, plan_checkpoint_outputs, save_training_checkpoint,
    restore_training_checkpoint,
)
from research.archive_validation import evaluate_archive_validation, model_fingerprint
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, require_sha256
from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule, training_step

PROTOCOL = "sonumio-private-archive-wave-training-v1"


def train_archive(
    path: str | Path, *, expected_sha256: str,
    config: ArchiveTrainingConfig | None = None,
    checkpoint: CheckpointOptions | None = None,
) -> dict:
    """Audit, train in bounded CPU minibatches, and evaluate fixed controls.

    The source archive is never modified. Only training rows enter optimizer
    updates; validation is read separately. Test rows participate solely in
    preflight integrity/readiness coverage, not optimization or validation.
    """
    require_sha256(expected_sha256)
    if config is None:
        config = ArchiveTrainingConfig()
    if not isinstance(config, ArchiveTrainingConfig):
        raise ArchiveError("Expected ArchiveTrainingConfig")
    options = CheckpointOptions() if checkpoint is None else checkpoint
    if not isinstance(options, CheckpointOptions):
        raise ArchiveError("Expected CheckpointOptions")
    runtime_identity()
    started = perf_counter()
    with CorpusArchiveReader(path, expected_sha256=expected_sha256) as reader:
        readiness = audit_archive(reader)
        if readiness.report["readiness_gate_passed"] is not True:
            raise ArchiveError("Archive failed group-balanced training readiness")
        sampler = CounterGroupSampler(readiness, seed=config.seed)
        eligible = {clip.row.index: clip for clip in readiness.eligible["train"]}
        # CPU-only RNG scope: preserve caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            with torch.device("cpu"):
                model = TinyWaveDenoiser(config.channels)
                schedule = DiffusionSchedule(config.diffusion_steps, profile=config.diffusion_profile)
            optimizer = make_archive_optimizer(model)
            initial_state = model_fingerprint(model)
            progress = TrainingProgress()
            resumed_sha = None
            if options.resume_from is not None:
                restored = restore_training_checkpoint(
                    options.resume_from, expected_sha256=options.resume_sha256,
                    config=config, readiness=readiness,
                    initial_model_sha256=initial_state, model=model, optimizer=optimizer,
                )
                sampler, progress = restored.sampler, restored.progress
                torch.set_rng_state(restored.cpu_rng_state)
                resumed_sha = restored.sha256
            resumed_steps = progress.completed_steps
            plan = plan_checkpoint_outputs(options, resumed_steps, config.steps, model)
            receipts = []

            def save_at(destination):
                receipts.append(save_training_checkpoint(
                    destination, config=config, readiness=readiness, sampler=sampler,
                    progress=progress, initial_model_sha256=initial_state,
                    model=model, optimizer=optimizer,
                ).as_dict())

            model.train()
            for _ in range(progress.completed_steps, config.steps):
                indices = sampler.draw(config.batch_size).tolist()
                batch = reader.read_train_batch(indices)
                for index, row, clip in zip(indices, batch.rows, batch.clips, strict=True):
                    item = eligible.get(index)
                    if (item is None or item.row != row
                            or item.pcm_sha256 != _cropped_clip_digest(clip[0])):
                        raise ArchiveError("Training batch differs from audited eligibility")
                optimizer.zero_grad(set_to_none=True)
                loss = training_step(
                    model, schedule, torch.from_numpy(batch.clips), torch.from_numpy(batch.labels),
                )
                if not torch.isfinite(loss):
                    raise ArchiveError("Nonfinite archive training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                progress = progress.advance(float(loss.detach()))
                if progress.completed_steps in plan.periodic:
                    save_at(plan.periodic[progress.completed_steps])
            trained_state = model_fingerprint(model)
            validation = evaluate_archive_validation(
                reader, readiness, model, schedule, seed=config.validation_seed,
                repeats=config.validation_repeats, batch_size=config.validation_batch_size,
            )
            if model_fingerprint(model) != trained_state:
                raise ArchiveError("Validation altered trained model state")
            if trained_state == initial_state:
                raise ArchiveError("Experimental optimizer did not change model state")
            if progress.completed_steps != config.steps:
                raise ArchiveError("Archive training did not reach target steps")
            if plan.final_path is not None:
                save_at(plan.final_path)
            report = {
                "protocol": PROTOCOL, "configuration": asdict(config),
                "archive_sha256": readiness.archive_sha256,
                "readiness": readiness.report,
                "eligible_split_fingerprints": readiness.eligible_fingerprints.copy(),
                "sampler": sampler.describe(), "training_examples": len(eligible),
                "initial_step_noise_mse": progress.first_loss, "last_step_noise_mse": progress.last_loss,
                "mean_training_noise_mse": progress.loss_total/config.steps,
                "completed_steps": progress.completed_steps,
                "resumed_steps": resumed_steps, "new_steps": config.steps-resumed_steps,
                "resume_checkpoint_sha256": resumed_sha, "checkpoint_receipts": receipts,
                "initial_model_sha256": initial_state, "trained_model_sha256": trained_state,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "validation": validation,
                "elapsed_seconds": round(perf_counter()-started, 3),
                "device": "cpu", "training_uses_bounded_archive_batches": True,
                "test_split_used_for_training_or_validation": False,
                "weights_or_audio_files_written": bool(receipts),
                "private_checkpoint_files_written": bool(receipts), "audio_files_written": False,
                "model_publication_authorized": False, "original_sonumio_1": False,
                "model_quality_established": False,
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--diffusion-profile", choices=DiffusionSchedule.PROFILES, default="linear_beta_v1")
    parser.add_argument("--validation-seed", type=int, default=101)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int)
    args = parser.parse_args(argv)
    try:
        config = ArchiveTrainingConfig(
            seed=args.seed, steps=args.steps, batch_size=args.batch_size,
            channels=args.channels, diffusion_steps=args.diffusion_steps,
            diffusion_profile=args.diffusion_profile, validation_seed=args.validation_seed,
            validation_repeats=args.validation_repeats, validation_batch_size=args.validation_batch_size,
        )
        checkpoint = CheckpointOptions(
            resume_from=args.resume, resume_sha256=args.resume_sha256,
            final_path=args.checkpoint_out, periodic_dir=args.checkpoint_dir,
            interval=args.checkpoint_every,
        )
        report = train_archive(
            args.archive, expected_sha256=args.expected_sha256,
            config=config, checkpoint=checkpoint,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Private archive waveform training: {exc}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
