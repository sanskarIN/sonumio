"""Private, digest-pinned CPU training checkpoint save and exact-state restore.

Only the known experimental waveform model, AdamW moments, sampler trace,
CPU RNG bytes and finite scalar progress are serialized. This format never
uses pickle, torch.load, dynamic code, or untrusted tensor allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from research.archive_readiness import ArchiveReadinessResult
from research.archive_sampler import CounterGroupSampler
from research.archive_training_contracts import (
    ArchiveTrainingConfig, TrainingProgress, MAX_TRAINING_STEPS,
    optimizer_identity, require_optimizer_contract, runtime_identity,
)
from research.archive_checkpoint_format import (
    TensorSpec, encode_checkpoint, decode_checkpoint, maximum_encoded_bytes,
    MAX_CHECKPOINT_BYTES,
)
from research.archive_validation import model_fingerprint
from research.checkpoint_snapshot import read_regular_snapshot
from research.corpus_archive_format import (
    ArchiveError, canonical_json, digest, require_sha256, SPLITS,
)
from research.wave_diffusion import TinyWaveDenoiser

PROTOCOL = "sonumio-private-archive-training-checkpoint-v1"
SUFFIX = ".stcheckpoint"
MAX_PERIODIC_CHECKPOINTS = 64
MAX_CHECKPOINT_RUN_BYTES = 512 * 1024 * 1024


def _require_cpu_model(model: TinyWaveDenoiser) -> None:
    if not isinstance(model, TinyWaveDenoiser) or any(
        value.device.type != "cpu" or value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise ArchiveError("Checkpoint requires the CPU float32 waveform model")


def tensor_specs(model: TinyWaveDenoiser) -> dict[str, TensorSpec]:
    _require_cpu_model(model)
    result = {
        "model/"+name: TensorSpec("f32", tuple(tensor.shape))
        for name, tensor in model.state_dict().items()
    }
    for name, parameter in model.named_parameters():
        result["optimizer/"+name+"/step"] = TensorSpec("f32", ())
        for moment in ("exp_avg", "exp_avg_sq"):
            result["optimizer/"+name+"/"+moment] = TensorSpec("f32", tuple(parameter.shape))
    result["rng/cpu"] = TensorSpec("u8", tuple(torch.random.get_rng_state().shape))
    return result


def _model_array_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    """Use precisely the archive validation model fingerprint protocol."""
    hasher = hashlib.sha256(b"sonumio-archive-model-state-v1\0")
    for key in sorted(name for name in arrays if name.startswith("model/")):
        name = key.removeprefix("model/").encode("utf-8")
        value = arrays[key]
        hasher.update(len(name).to_bytes(4, "little"))
        hasher.update(name)
        hasher.update(len(value.shape).to_bytes(4, "little"))
        for size in value.shape:
            hasher.update(size.to_bytes(8, "little"))
        hasher.update(np.asarray(value, dtype="<f4", order="C").tobytes())
    return hasher.hexdigest()


def _check_sampler(
    sampler: CounterGroupSampler, progress: TrainingProgress,
    config: ArchiveTrainingConfig, readiness: ArchiveReadinessResult,
) -> dict:
    if not isinstance(sampler, CounterGroupSampler):
        raise ArchiveError("Expected counter-based group-balanced sampler")
    state = sampler.describe()
    if (sampler.seed != config.seed
            or sampler.eligible_sha256 != readiness.eligible_fingerprints["train"]
            or sampler.position != progress.completed_steps*config.batch_size):
        raise ArchiveError("Sampler state does not match completed optimizer steps")
    return state


def _require_readiness(readiness: ArchiveReadinessResult) -> dict[str, str]:
    if (not isinstance(readiness, ArchiveReadinessResult)
            or readiness.report.get("readiness_gate_passed") is not True
            or set(readiness.eligible_fingerprints) != set(SPLITS)):
        raise ArchiveError("Checkpoint requires a freshly audited ready archive")
    require_sha256(readiness.archive_sha256)
    result = {split: require_sha256(readiness.eligible_fingerprints[split]) for split in SPLITS}
    if readiness.report.get("eligible_split_fingerprints") != result:
        raise ArchiveError("Checkpoint eligibility fingerprints disagree")
    return result


def _optimizer_arrays(
    model: TinyWaveDenoiser, optimizer: torch.optim.AdamW, completed_steps: int,
) -> dict[str, np.ndarray]:
    parameters = require_optimizer_contract(model, optimizer)
    if set(optimizer.state) != {p for _, p in parameters}:
        raise ArchiveError("Optimizer state does not cover every model parameter")
    result = {}
    for name, parameter in parameters:
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ArchiveError("Unexpected AdamW parameter state")
        for field, tensor in state.items():
            expected = () if field == "step" else tuple(parameter.shape)
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32 or tuple(tensor.shape) != expected
                    or not torch.isfinite(tensor).all()):
                raise ArchiveError("Invalid AdamW checkpoint tensor")
            if field == "step" and float(tensor) != completed_steps:
                raise ArchiveError("AdamW step disagrees with completed training steps")
            if field == "exp_avg_sq" and torch.any(tensor < 0):
                raise ArchiveError("AdamW second moment cannot be negative")
            result["optimizer/"+name+"/"+field] = tensor.detach().numpy()
    return result


def _output_path(path: str | Path) -> Path:
    target = Path(path)
    if (target.suffix != SUFFIX or not target.parent.is_dir()
            or target.parent.is_symlink()):
        raise ArchiveError("Checkpoint needs .stcheckpoint in an ordinary existing directory")
    if target.exists() or target.is_symlink():
        raise ArchiveError("Checkpoint destination already exists")
    return target


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    completed_steps: int
    filename: str
    sha256: str
    bytes_written: int

    def as_dict(self) -> dict:
        return {
            "completed_steps": self.completed_steps, "filename": self.filename,
            "sha256": self.sha256, "bytes_written": self.bytes_written,
            "private_model_optimizer_rng_state": True,
        }


@dataclass(frozen=True, slots=True)
class CheckpointOptions:
    resume_from: str | Path | None = None
    resume_sha256: str | None = None
    final_path: str | Path | None = None
    periodic_dir: str | Path | None = None
    interval: int | None = None

    def __post_init__(self):
        if (self.resume_from is None) != (self.resume_sha256 is None):
            raise ArchiveError("Resume requires both checkpoint path and trusted SHA256")
        if self.resume_sha256 is not None:
            require_sha256(self.resume_sha256)
        if (self.periodic_dir is None) != (self.interval is None):
            raise ArchiveError("Periodic checkpoints require directory and interval")
        if self.interval is not None and (
            type(self.interval) is not int or not 1 <= self.interval <= MAX_TRAINING_STEPS
        ):
            raise ArchiveError("Invalid periodic checkpoint interval")
        for value in (self.resume_from, self.final_path, self.periodic_dir):
            if value is not None and not isinstance(value, (str, Path)):
                raise ArchiveError("Invalid checkpoint path")


@dataclass(frozen=True, slots=True)
class CheckpointPlan:
    periodic: dict[int, Path]
    final_path: Path | None


def plan_checkpoint_outputs(
    options: CheckpointOptions, completed_steps: int, target_steps: int,
    model: TinyWaveDenoiser,
) -> CheckpointPlan:
    if not isinstance(options, CheckpointOptions):
        raise ArchiveError("Expected checkpoint options")
    if (type(completed_steps) is not int or type(target_steps) is not int
            or not 0 <= completed_steps <= target_steps <= MAX_TRAINING_STEPS):
        raise ArchiveError("Invalid checkpoint plan step range")
    periodic = {}
    if options.periodic_dir is not None:
        directory = Path(options.periodic_dir)
        if not directory.is_dir() or directory.is_symlink():
            raise ArchiveError("Periodic checkpoint directory must already exist")
        interval = options.interval
        first = (completed_steps//interval+1)*interval
        count = max(0, (target_steps-first)//interval+1)
        if count > MAX_PERIODIC_CHECKPOINTS:
            raise ArchiveError("Too many periodic checkpoint files in one run")
        for step in range(first, target_steps+1, interval):
            periodic[step] = _output_path(directory/f"step-{step:08d}{SUFFIX}")
    final = _output_path(options.final_path) if options.final_path is not None else None
    paths = [*periodic.values()]
    if final is not None:
        paths.append(final)
    if len({path.resolve(strict=False) for path in paths}) != len(paths):
        raise ArchiveError("Checkpoint output paths collide")
    if paths and len(paths)*maximum_encoded_bytes(tensor_specs(model)) > MAX_CHECKPOINT_RUN_BYTES:
        raise ArchiveError("Planned checkpoint files exceed per-run byte budget")
    return CheckpointPlan(periodic, final)


def save_training_checkpoint(
    path: str | Path, *, config: ArchiveTrainingConfig,
    readiness: ArchiveReadinessResult, sampler: CounterGroupSampler,
    progress: TrainingProgress, initial_model_sha256: str,
    model: TinyWaveDenoiser, optimizer: torch.optim.AdamW,
) -> CheckpointReceipt:
    """Write one private complete-step state; never replace an existing file."""
    target = _output_path(path)
    if not isinstance(config, ArchiveTrainingConfig) or not isinstance(progress, TrainingProgress):
        raise ArchiveError("Invalid checkpoint training configuration or progress")
    if not 1 <= progress.completed_steps <= config.steps:
        raise ArchiveError("Checkpoint requires at least one completed optimizer step")
    eligible = _require_readiness(readiness)
    sampler_state = _check_sampler(sampler, progress, config, readiness)
    initial = require_sha256(initial_model_sha256)
    specs = tensor_specs(model)
    arrays = {"model/"+name: value.detach().numpy()
              for name, value in model.state_dict().items()}
    arrays.update(_optimizer_arrays(model, optimizer, progress.completed_steps))
    arrays["rng/cpu"] = torch.random.get_rng_state().numpy()
    trained = model_fingerprint(model)
    if trained == initial:
        raise ArchiveError("Checkpoint model did not change from initialization")
    metadata = {
        "protocol": PROTOCOL, "configuration": config.trajectory_identity(),
        "runtime": runtime_identity(), "optimizer": optimizer_identity(),
        "archive_sha256": readiness.archive_sha256,
        "eligible_split_fingerprints": eligible,
        "sampler": sampler_state, "progress": progress.as_dict(),
        "initial_model_sha256": initial, "model_sha256": trained,
        "cpu_rng_sha256": digest(arrays["rng/cpu"].tobytes()),
        "private_checkpoint": True, "model_publication_authorized": False,
        "original_sonumio_1": False,
    }
    blob = encode_checkpoint(metadata, arrays, specs)
    checksum = digest(blob)
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".sonumio-train-",
            suffix=".partial", delete=False,
        ) as stream:
            scratch = Path(stream.name)
            if stream.write(blob) != len(blob):
                raise ArchiveError("Short checkpoint write")
            stream.flush()
            os.fsync(stream.fileno())
        written = read_regular_snapshot(scratch, max_bytes=MAX_CHECKPOINT_BYTES)
        if digest(written) != checksum:
            raise ArchiveError("Checkpoint scratch SHA256 mismatch")
        try:
            os.link(scratch, target)
        except OSError as exc:
            raise ArchiveError("Atomic no-clobber checkpoint publication failed") from exc
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return CheckpointReceipt(progress.completed_steps, target.name, checksum, len(blob))


@dataclass(frozen=True, slots=True)
class RestoredCheckpoint:
    progress: TrainingProgress
    sampler: CounterGroupSampler
    cpu_rng_state: torch.Tensor
    sha256: str


def _validate_metadata(
    metadata: dict, config: ArchiveTrainingConfig, readiness: ArchiveReadinessResult,
    initial_model_sha256: str,
) -> TrainingProgress:
    fields = {
        "protocol", "configuration", "runtime", "optimizer", "archive_sha256",
        "eligible_split_fingerprints", "sampler", "progress", "initial_model_sha256",
        "model_sha256", "cpu_rng_sha256", "private_checkpoint",
        "model_publication_authorized", "original_sonumio_1",
    }
    if set(metadata) != fields or metadata["protocol"] != PROTOCOL:
        raise ArchiveError("Unsupported training checkpoint metadata")
    if (metadata["private_checkpoint"] is not True
            or metadata["model_publication_authorized"] is not False
            or metadata["original_sonumio_1"] is not False):
        raise ArchiveError("Unsafe private checkpoint provenance declarations")
    # Canonical bytes distinguish e.g. Boolean and integer metadata values.
    if canonical_json(metadata["configuration"]) != canonical_json(config.trajectory_identity()):
        raise ArchiveError("Checkpoint training configuration changed")
    if canonical_json(metadata["runtime"]) != canonical_json(runtime_identity()):
        raise ArchiveError("Checkpoint runtime differs; exact continuation is not supported")
    if canonical_json(metadata["optimizer"]) != canonical_json(optimizer_identity()):
        raise ArchiveError("Checkpoint optimizer protocol changed")
    if require_sha256(metadata["archive_sha256"]) != readiness.archive_sha256:
        raise ArchiveError("Checkpoint belongs to a different private archive")
    if metadata["eligible_split_fingerprints"] != _require_readiness(readiness):
        raise ArchiveError("Checkpoint eligible split fingerprints changed")
    if require_sha256(metadata["initial_model_sha256"]) != require_sha256(initial_model_sha256):
        raise ArchiveError("Checkpoint initial model fingerprint changed")
    if require_sha256(metadata["model_sha256"]) == initial_model_sha256:
        raise ArchiveError("Checkpoint contains unchanged initial model")
    require_sha256(metadata["cpu_rng_sha256"])
    progress = TrainingProgress.parse(metadata["progress"])
    if not 1 <= progress.completed_steps <= config.steps:
        raise ArchiveError("Resume target precedes completed checkpoint steps")
    return progress


def restore_training_checkpoint(
    path: str | Path, *, expected_sha256: str, config: ArchiveTrainingConfig,
    readiness: ArchiveReadinessResult, initial_model_sha256: str,
    model: TinyWaveDenoiser, optimizer: torch.optim.AdamW,
) -> RestoredCheckpoint:
    """Validate complete private state before loading into fresh CPU objects.

    CPU RNG is returned, not installed globally. The caller must install it
    inside an isolated training RNG scope, after fresh initialization.
    """
    expected_sha256 = require_sha256(expected_sha256)
    if not isinstance(config, ArchiveTrainingConfig):
        raise ArchiveError("Expected archive training configuration")
    parameters = require_optimizer_contract(model, optimizer)
    if optimizer.state or model_fingerprint(model) != initial_model_sha256:
        raise ArchiveError("Restore requires freshly initialized model and optimizer")
    blob = read_regular_snapshot(path, max_bytes=MAX_CHECKPOINT_BYTES)
    if digest(blob) != expected_sha256:
        raise ArchiveError("Private training checkpoint SHA256 mismatch")
    metadata, arrays = decode_checkpoint(blob, tensor_specs(model))
    progress = _validate_metadata(metadata, config, readiness, initial_model_sha256)
    sampler = CounterGroupSampler.restore(readiness, metadata["sampler"])
    _check_sampler(sampler, progress, config, readiness)
    if _model_array_fingerprint(arrays) != metadata["model_sha256"]:
        raise ArchiveError("Checkpoint model fingerprint mismatch")
    if digest(arrays["rng/cpu"].tobytes()) != metadata["cpu_rng_sha256"]:
        raise ArchiveError("Checkpoint CPU RNG fingerprint mismatch")
    rng = torch.from_numpy(arrays["rng/cpu"].copy())
    try:
        torch.Generator(device="cpu").set_state(rng)
    except RuntimeError as exc:
        raise ArchiveError("Invalid checkpoint CPU RNG state") from exc
    restored_model = {
        name: torch.from_numpy(arrays["model/"+name].copy())
        for name in model.state_dict()
    }
    restored_optimizer = {}
    for name, parameter in parameters:
        prefix = "optimizer/"+name+"/"
        if float(arrays[prefix+"step"]) != progress.completed_steps:
            raise ArchiveError("Checkpoint AdamW step disagrees with completed steps")
        if np.any(arrays[prefix+"exp_avg_sq"] < 0):
            raise ArchiveError("Checkpoint AdamW second moment cannot be negative")
        restored_optimizer[parameter] = {
            field: torch.from_numpy(arrays[prefix+field].copy())
            for field in ("step", "exp_avg", "exp_avg_sq")
        }
    # No untrusted checkpoint metadata is passed to optimizer.load_state_dict.
    # All names, tensor geometries, values and progress were checked above.
    model.load_state_dict(restored_model, strict=True)
    optimizer.state.update(restored_optimizer)
    if model_fingerprint(model) != metadata["model_sha256"]:
        raise ArchiveError("Restored model does not match verified checkpoint")
    return RestoredCheckpoint(progress, sampler, rng, expected_sha256)
