"""Read-only verification and same-step comparison for private training state.

Both the archive and each checkpoint require independently retained digests.
Verification reconstructs known model/optimizer objects, replays the sampler
and checks the same resume contract as the trainer. It never executes an
optimizer step, emits weights, or establishes data rights or model quality.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.archive_checkpoint import restore_training_checkpoint
from research.archive_readiness import audit_archive, ArchiveReadinessResult
from research.archive_training_contracts import (
    ArchiveTrainingConfig, make_archive_optimizer, require_optimizer_contract,
)
from research.archive_validation import model_fingerprint
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, require_sha256
from research.wave_diffusion import TinyWaveDenoiser

PROTOCOL = "sonumio-private-training-checkpoint-audit-v1"


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    path: str | Path
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)):
            raise ArchiveError("Expected a local checkpoint path")
        require_sha256(self.sha256)


def _hash_optimizer(model, optimizer) -> str:
    """Canonical digest of all restored AdamW numeric values, not objects."""
    digest = hashlib.sha256(b"sonumio-private-adamw-state-v1\0")
    for name, parameter in sorted(require_optimizer_contract(model, optimizer)):
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "little"))
        digest.update(label)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            array = optimizer.state[parameter][field].detach().numpy()
            digest.update(len(array.shape).to_bytes(4, "little"))
            for dimension in array.shape:
                digest.update(dimension.to_bytes(8, "little"))
            digest.update(np.asarray(array, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


def _verify(reference: CheckpointReference, config: ArchiveTrainingConfig,
            readiness: ArchiveReadinessResult) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        with torch.device("cpu"):
            model = TinyWaveDenoiser(config.channels)
        optimizer = make_archive_optimizer(model)
        initial = model_fingerprint(model)
        restored = restore_training_checkpoint(
            reference.path, expected_sha256=reference.sha256, config=config,
            readiness=readiness, initial_model_sha256=initial,
            model=model, optimizer=optimizer,
        )
        return {
            "checkpoint_sha256": restored.sha256,
            "archive_sha256": readiness.archive_sha256,
            "eligible_split_fingerprints": readiness.eligible_fingerprints.copy(),
            "configuration": config.trajectory_identity(),
            "initial_model_sha256": initial,
            "model_sha256": model_fingerprint(model),
            "optimizer_sha256": _hash_optimizer(model, optimizer),
            "cpu_rng_sha256": hashlib.sha256(restored.cpu_rng_state.numpy().tobytes()).hexdigest(),
            "sampler": restored.sampler.describe(),
            "progress": restored.progress.as_dict(),
            "resume_contract_verified": True,
        }


def audit_checkpoints(
    archive: str | Path, *, archive_sha256: str,
    primary: CheckpointReference, config: ArchiveTrainingConfig,
    comparison: CheckpointReference | None = None,
) -> dict:
    """Audit one or two digest-pinned checkpoints against one fresh corpus audit.

    Identical same-step state is evidence of numerical equivalence, not proof
    of ancestry. Different-step snapshots are explicitly not compared as
    equivalent trajectories; no model quality or performance ranking is made.
    """
    require_sha256(archive_sha256)
    if not isinstance(config, ArchiveTrainingConfig):
        raise ArchiveError("Expected ArchiveTrainingConfig")
    if not isinstance(primary, CheckpointReference) or (
        comparison is not None and not isinstance(comparison, CheckpointReference)
    ):
        raise ArchiveError("Expected digest-pinned checkpoint references")
    with CorpusArchiveReader(archive, expected_sha256=archive_sha256) as reader:
        readiness = audit_archive(reader)
        if readiness.report["readiness_gate_passed"] is not True:
            raise ArchiveError("Archive failed group-balanced training readiness")
        first = _verify(primary, config, readiness)
        second = _verify(comparison, config, readiness) if comparison is not None else None
        report = {
            "protocol": PROTOCOL,
            "primary": first,
            "comparison": second,
            "same_step": None,
            "identical_same_step_state": None,
            "component_matches": None,
            "ancestry_proven": False,
            "optimizer_steps_executed": 0,
            "files_written": 0,
            "model_publication_authorized": False,
            "original_sonumio_1": False,
            "model_quality_established": False,
        }
        if second is not None:
            same_step = first["progress"]["completed_steps"] == second["progress"]["completed_steps"]
            report["same_step"] = same_step
            if same_step:
                components = {
                    name: first[name] == second[name]
                    for name in ("model_sha256", "optimizer_sha256", "cpu_rng_sha256",
                                 "sampler", "progress", "initial_model_sha256",
                                 "archive_sha256", "eligible_split_fingerprints", "configuration")
                }
                report["component_matches"] = components
                report["identical_same_step_state"] = all(components.values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--compare-sha256")
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--diffusion-profile", default="linear_beta_v1")
    parser.add_argument("--validation-seed", type=int, default=101)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if (args.compare is None) != (args.compare_sha256 is None):
            raise ArchiveError("Comparison requires both checkpoint path and trusted SHA256")
        if args.require_identical and args.compare is None:
            raise ArchiveError("Identical-state gate requires a comparison checkpoint")
        config = ArchiveTrainingConfig(
            steps=args.steps, seed=args.seed, batch_size=args.batch_size,
            channels=args.channels, diffusion_steps=args.diffusion_steps,
            diffusion_profile=args.diffusion_profile, validation_seed=args.validation_seed,
            validation_repeats=args.validation_repeats, validation_batch_size=args.validation_batch_size,
        )
        report = audit_checkpoints(
            args.archive, archive_sha256=args.archive_sha256,
            primary=CheckpointReference(args.checkpoint, args.checkpoint_sha256), config=config,
            comparison=CheckpointReference(args.compare, args.compare_sha256)
            if args.compare is not None else None,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Private checkpoint audit: {exc}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if args.require_identical and report["identical_same_step_state"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
