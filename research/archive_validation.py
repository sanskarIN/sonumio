"""Paired fixed-noise validation for experimental archive waveform models.

Noise and timestep assignments depend on the eligible validation fingerprint,
row index and repeat, not minibatch partition or model RNG state. This is a
numerical noise-prediction diagnostic, not a generated-audio listening score.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import struct

import numpy as np
import torch

from research.archive_readiness import ArchiveReadinessResult, eligible_fingerprint
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, TinyWaveDenoiser, DiffusionSchedule

PROTOCOL = "sonumio-archive-fixed-noise-validation-v1"


def _check_int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ArchiveError(f"Invalid validation {name}")


def _example_seed(eligible_hash: str, seed: int, index: int, repeat: int) -> int:
    raw = hashlib.sha256(
        b"sonumio-archive-validation-noise-v1\0" + bytes.fromhex(eligible_hash)
        + struct.pack("<III", seed, index, repeat),
    ).digest()
    return int.from_bytes(raw[:8], "little") & ((1 << 63)-1)


class _Errors:
    def __init__(self):
        self.count = 0
        self.model = 0.0
        self.zero = 0.0

    def add(self, model: float, zero: float):
        if not (math.isfinite(model) and math.isfinite(zero) and model >= 0 and zero >= 0):
            raise ArchiveError("Nonfinite validation error")
        self.count += 1
        self.model += model
        self.zero += zero

    def report(self):
        if not self.count:
            return {"examples": 0, "model_noise_mse": None, "zero_noise_mse": None,
                    "relative_improvement": None}
        model = self.model / self.count
        zero = self.zero / self.count
        return {"examples": self.count, "model_noise_mse": model, "zero_noise_mse": zero,
                "relative_improvement": None if zero == 0 else 1-model/zero}


def model_fingerprint(model: TinyWaveDenoiser) -> str:
    digest = hashlib.sha256(b"sonumio-archive-model-state-v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ArchiveError("Expected finite float32 model state")
        label = name.encode("utf-8")
        digest.update(len(label).to_bytes(4, "little"))
        digest.update(label)
        digest.update(len(value.shape).to_bytes(4, "little"))
        for size in value.shape:
            digest.update(size.to_bytes(8, "little"))
        digest.update(np.asarray(value, dtype="<f4", order="C").tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_archive_validation(
    reader: CorpusArchiveReader, readiness: ArchiveReadinessResult,
    model: TinyWaveDenoiser, schedule: DiffusionSchedule, *,
    seed: int, repeats: int = 2, batch_size: int = 8,
) -> dict:
    """Evaluate eligible validation rows only, against paired zero controls.

    The per-row noise trace is independent of batch and model state. Numerical
    reductions/model kernels can still differ slightly with batch geometry.
    No test split is accessed for validation. The model's mode is restored.
    """
    _check_int(seed, 0, 2**32-1, "seed")
    _check_int(repeats, 1, 4, "repeats")
    _check_int(batch_size, 1, 64, "batch size")
    if not isinstance(reader, CorpusArchiveReader) or not isinstance(readiness, ArchiveReadinessResult):
        raise ArchiveError("Expected verified archive and eligibility")
    if reader.report()["archive_sha256"] != readiness.archive_sha256:
        raise ArchiveError("Validation eligibility belongs to another archive")
    if readiness.report.get("readiness_gate_passed") is not True:
        raise ArchiveError("Archive failed readiness")
    if not isinstance(model, TinyWaveDenoiser) or not isinstance(schedule, DiffusionSchedule):
        raise ArchiveError("Expected experimental waveform model and diffusion schedule")
    if next(model.parameters()).device.type != "cpu" or schedule.alpha_bars.device.type != "cpu":
        raise ArchiveError("Archive validation is currently CPU-only")
    eligible = {clip.row.index: clip for clip in readiness.eligible["validation"]}
    eligible_hash = readiness.eligible_fingerprints["validation"]
    if (eligible_fingerprint(readiness.eligible["validation"]) != eligible_hash
            or readiness.report["eligible_split_fingerprints"]["validation"] != eligible_hash):
        raise ArchiveError("Validation eligibility fingerprint changed")
    if not eligible:
        raise ArchiveError("No eligible archive validation rows")
    model_before = model_fingerprint(model)
    mode = model.training
    model.eval()
    trace = hashlib.sha256(b"sonumio-archive-validation-trace-v1\0")
    totals = _Errors()
    groups = defaultdict(_Errors)
    categories = [_Errors() for _ in CATEGORIES]
    seen = set()
    pending = []

    def flush():
        if not pending:
            return
        clean = torch.from_numpy(np.stack([item[1] for item in pending]))
        labels = torch.tensor([item[0].label for item in pending], dtype=torch.long)
        timestep_values = []
        noise_values = []
        for row, _, repeat in pending:
            generator = torch.Generator(device="cpu").manual_seed(
                _example_seed(eligible_hash, seed, row.index, repeat)
            )
            t = int(torch.randint(schedule.steps, (1,), generator=generator).item())
            noise = torch.randn((1, CLIP_FRAMES), generator=generator, dtype=torch.float32)
            timestep_values.append(t)
            noise_values.append(noise)
            trace.update(struct.pack("<III", row.index, repeat, t))
            trace.update(np.asarray(noise.numpy(), dtype="<f4").tobytes())
        steps = torch.tensor(timestep_values, dtype=torch.long)
        noise = torch.stack(noise_values)
        prediction = model(schedule.corrupt(clean, steps, noise), steps, labels)
        error = (prediction.double()-noise.double()).square().mean((1, 2))
        baseline = noise.double().square().mean((1, 2))
        for (row, _, _), model_error, zero_error in zip(pending, error.tolist(), baseline.tolist(), strict=True):
            totals.add(model_error, zero_error)
            groups[row.group_id].add(model_error, zero_error)
            categories[row.label].add(model_error, zero_error)
        pending.clear()

    try:
        for shard in reader.iter_shards("validation"):
            for row, samples in zip(shard.rows, shard.clips, strict=True):
                item = eligible.get(row.index)
                if item is None:
                    continue
                if item.row != row or _cropped_clip_digest(samples[0]) != item.pcm_sha256:
                    raise ArchiveError("Audited validation clip changed")
                seen.add(row.index)
                for repeat in range(repeats):
                    # Copy at most batch_size individual rows so no pending
                    # view pins earlier complete shards in memory.
                    pending.append((row, samples.copy(), repeat))
                    if len(pending) == batch_size:
                        flush()
        flush()
        if seen != set(eligible) or totals.count != len(eligible)*repeats:
            raise ArchiveError("Validation did not cover eligible rows exactly")
        if model_fingerprint(model) != model_before:
            raise ArchiveError("Model changed during fixed-noise validation")
    finally:
        model.train(mode)
    group_reports = [value.report() for value in groups.values()]
    macro_model = math.fsum(x["model_noise_mse"] for x in group_reports)/len(group_reports)
    macro_zero = math.fsum(x["zero_noise_mse"] for x in group_reports)/len(group_reports)
    return {
        "protocol": PROTOCOL, "eligible_validation_sha256": eligible_hash,
        "seed": seed, "repeats": repeats, "batch_size": batch_size,
        "diffusion_steps": schedule.steps, "diffusion_profile": schedule.profile,
        "eligible_validation_clips": len(eligible), "recording_groups": len(groups),
        "noise_trace_sha256": trace.hexdigest(), "model_state_sha256": model_before,
        "clip_weighted": totals.report(),
        "group_macro": {"model_noise_mse": macro_model, "zero_noise_mse": macro_zero,
                        "relative_improvement": None if macro_zero == 0 else 1-macro_model/macro_zero},
        "by_category": {category: value.report()
                        for category, value in zip(CATEGORIES, categories, strict=True)},
        "test_split_used": False, "model_state_unchanged": True,
        "perceptual_audio_quality_measured": False,
    }
