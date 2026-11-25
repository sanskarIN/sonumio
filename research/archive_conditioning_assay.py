"""Paired held-out category intervention on private experimental waveform models.

All conditioning alternatives see exactly the same corrupted input and target.
This measures teacher-forced noise prediction, not generated audio semantics.
No test-split scores, waveform exports, optimizer updates or model publication.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

from research.archive_checkpoint import restore_training_checkpoint
from research.archive_readiness import (
    ArchiveReadinessResult, audit_archive, eligible_fingerprint,
)
from research.archive_training_contracts import ArchiveTrainingConfig, make_archive_optimizer
from research.archive_validation import model_fingerprint
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, require_sha256
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, DiffusionSchedule, TinyWaveDenoiser

PROTOCOL = "sonumio-archive-category-intervention-v1"
LEVELS = ("early", "middle", "late")
MAX_ASSAY_ROWS = 4096
MAX_PREDICTION_ROWS = 1_000_000
TIE_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class AssayConfig:
    seed: int = 301
    repeats: int = 2
    batch_size: int = 4

    def __post_init__(self):
        for name, value, lo, hi in (
            ("seed", self.seed, 0, 2**32-1),
            ("repeats", self.repeats, 1, 4),
            ("batch size", self.batch_size, 1, 16),
        ):
            if type(value) is not int or not lo <= value <= hi:
                raise ArchiveError("Invalid conditioning assay " + name)


def timestep_panel(steps: int) -> tuple[int, int, int]:
    """Fixed early, middle and late positions; no model-specific selection."""
    if type(steps) is not int or not 8 <= steps <= 1000:
        raise ArchiveError("Invalid assay diffusion step count")
    return (0, (steps-1)//2, steps-1)


def _noise_seed(fingerprint: str, seed: int, index: int, repeat: int, timestep: int) -> int:
    raw = hashlib.sha256(
        b"sonumio-category-assay-noise-v1\0" + bytes.fromhex(fingerprint)
        + struct.pack("<IIII", seed, index, repeat, timestep)
    ).digest()
    return int.from_bytes(raw[:8], "little") & ((1<<63)-1)


class _PairedSummary:
    def __init__(self):
        self.count = 0
        self.sums = np.zeros(7, dtype=np.float64)
        self.strict_wins = self.ties = 0
        self.rank_sum = 0.0

    def add(self, errors: np.ndarray, truth: int, zero: float):
        if (errors.shape != (len(CATEGORIES)+1,) or not np.isfinite(errors).all()
                or np.any(errors < 0) or not math.isfinite(zero) or zero < 0):
            raise ArchiveError("Invalid paired conditioning errors")
        correct = float(errors[truth])
        wrong = np.delete(errors[:-1], truth)
        hardest = float(wrong.min())
        mean_wrong = float(wrong.mean())
        null = float(errors[-1])
        self.sums += (correct, mean_wrong, hardest, null, zero,
                      mean_wrong-correct, hardest-correct)
        less = int(np.count_nonzero(wrong < correct-TIE_ATOL))
        tied = int(np.count_nonzero(np.abs(wrong-correct) <= TIE_ATOL))
        self.rank_sum += 1 + less + .5*tied
        self.strict_wins += int(less == 0 and tied == 0)
        self.ties += int(less == 0 and tied > 0)
        self.count += 1

    def report(self):
        names = ("matched_mse", "mean_wrong_mse", "hardest_wrong_mse", "null_mse",
                 "zero_predictor_mse", "mean_wrong_minus_matched", "hardest_wrong_minus_matched")
        if not self.count:
            return {"cases": 0, **dict.fromkeys(names), "mean_rank": None,
                    "strict_win_fraction": None, "best_tie_fraction": None,
                    "null_minus_matched": None}
        result = dict(zip(names, (self.sums/self.count).tolist(), strict=True))
        return {"cases": self.count, **result,
                "mean_rank": self.rank_sum/self.count,
                "strict_win_fraction": self.strict_wins/self.count,
                "best_tie_fraction": self.ties/self.count,
                "null_minus_matched": result["null_mse"]-result["matched_mse"]}


def _group_macro(groups: dict) -> dict:
    rows = [summary.report() for summary in groups.values()]
    keys = ("matched_mse", "mean_wrong_minus_matched", "hardest_wrong_minus_matched",
            "null_minus_matched", "mean_rank", "strict_win_fraction")
    return {"recording_groups": len(rows), **{
        key: math.fsum(row[key] for row in rows)/len(rows) if rows else None
        for key in keys
    }}


@torch.no_grad()
def evaluate_conditioning(
    reader: CorpusArchiveReader, readiness: ArchiveReadinessResult,
    model: TinyWaveDenoiser, schedule: DiffusionSchedule, *,
    config: AssayConfig | None = None,
) -> dict:
    config = AssayConfig() if config is None else config
    if not isinstance(config, AssayConfig):
        raise ArchiveError("Expected AssayConfig")
    if not isinstance(reader, CorpusArchiveReader) or not isinstance(readiness, ArchiveReadinessResult):
        raise ArchiveError("Expected verified archive and readiness audit")
    if reader.report()["archive_sha256"] != readiness.archive_sha256:
        raise ArchiveError("Conditioning eligibility belongs to another archive")
    if readiness.report["readiness_gate_passed"] is not True:
        raise ArchiveError("Archive failed readiness")
    if (not isinstance(model, TinyWaveDenoiser) or model.classes != len(CATEGORIES)
            or not isinstance(schedule, DiffusionSchedule)
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or schedule.alpha_bars.device.type != "cpu"):
        raise ArchiveError("Expected CPU single-class waveform research model")
    eligible = readiness.eligible["validation"]
    fingerprint = eligible_fingerprint(eligible)
    if (fingerprint != readiness.eligible_fingerprints["validation"]
            or fingerprint != readiness.report["eligible_split_fingerprints"]["validation"]):
        raise ArchiveError("Conditioning eligibility fingerprint changed")
    n = len(eligible)
    prediction_rows = n*config.repeats*len(LEVELS)*(len(CATEGORIES)+1)
    if not 1 <= n <= MAX_ASSAY_ROWS or prediction_rows > MAX_PREDICTION_ROWS:
        raise ArchiveError("Conditioning assay exceeds validation workload budget")
    by_index = {item.row.index: item for item in eligible}
    if len(by_index) != n:
        raise ArchiveError("Duplicate eligible validation row")
    panel = timestep_panel(schedule.steps)
    overall = _PairedSummary()
    by_level = {name: _PairedSummary() for name in LEVELS}
    by_category = {name: _PairedSummary() for name in CATEGORIES}
    by_group = defaultdict(_PairedSummary)
    by_level_group = {name: defaultdict(_PairedSummary) for name in LEVELS}
    matrix = np.zeros((len(LEVELS),len(CATEGORIES),len(CATEGORIES)+1), dtype=np.float64)
    counts = np.zeros((len(LEVELS),len(CATEGORIES)), dtype=np.int64)
    trace = hashlib.sha256(b"sonumio-category-assay-trace-v1\0")
    pending = []
    seen = set()
    before = model_fingerprint(model)
    mode = model.training
    model.eval()

    def flush():
        if not pending:
            return
        clean = torch.from_numpy(np.stack([item[1] for item in pending]))
        timesteps = torch.tensor([panel[item[2]] for item in pending], dtype=torch.long)
        noise = []
        for row, _, level, repeat in pending:
            t = panel[level]
            generator = torch.Generator(device="cpu").manual_seed(
                _noise_seed(fingerprint, config.seed, row.index, repeat, t)
            )
            item = torch.randn((1,CLIP_FRAMES), generator=generator, dtype=torch.float32)
            noise.append(item)
            trace.update(struct.pack("<III",row.index,repeat,t))
            trace.update(np.asarray(item.numpy(),dtype="<f4").tobytes())
        target = torch.stack(noise)
        noisy = schedule.corrupt(clean,timesteps,target)
        errors = []
        for label in range(len(CATEGORIES)+1):
            labels = torch.full((len(pending),),label,dtype=torch.long)
            prediction = model(noisy,timesteps,labels)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ArchiveError("Nonfinite or malformed conditioning prediction")
            errors.append((prediction.double()-target.double()).square().mean((1,2)).numpy())
        values = np.stack(errors,axis=1)
        baseline = target.double().square().mean((1,2)).numpy()
        for (row, _, level, _), error, zero in zip(pending,values,baseline,strict=True):
            category = CATEGORIES[row.label]
            level_name = LEVELS[level]
            for summary in (overall, by_level[level_name], by_category[category],
                            by_group[row.group_id], by_level_group[level_name][row.group_id]):
                summary.add(error,row.label,float(zero))
            matrix[level,row.label] += error
            counts[level,row.label] += 1
        pending.clear()

    try:
        for shard in reader.iter_shards("validation"):
            for row, samples in zip(shard.rows,shard.clips,strict=True):
                item = by_index.get(row.index)
                if item is None:
                    continue
                if row.index in seen or row != item.row or _cropped_clip_digest(samples[0]) != item.pcm_sha256:
                    raise ArchiveError("Audited conditioning validation clip changed")
                seen.add(row.index)
                for level in range(len(LEVELS)):
                    for repeat in range(config.repeats):
                        pending.append((row,samples.copy(),level,repeat))
                        if len(pending) == config.batch_size:
                            flush()
        flush()
        if seen != set(by_index) or overall.count != n*config.repeats*len(LEVELS):
            raise ArchiveError("Conditioning assay missed eligible validation cases")
        if model_fingerprint(model) != before:
            raise ArchiveError("Model changed during conditioning assay")
        reader.report()  # final descriptor verification
    finally:
        model.train(mode)
    return {
        "protocol": PROTOCOL, "archive_sha256": readiness.archive_sha256,
        "eligible_validation_sha256": fingerprint, "model_sha256": before,
        "seed": config.seed, "repeats": config.repeats, "batch_size": config.batch_size,
        "diffusion_profile": schedule.profile, "diffusion_steps": schedule.steps,
        "timesteps": dict(zip(LEVELS,panel,strict=True)),
        "eligible_validation_clips": n, "prediction_rows": prediction_rows,
        "noise_trace_sha256": trace.hexdigest(),
        "clip_weighted": overall.report(), "group_macro": _group_macro(by_group),
        "by_category": {key:value.report() for key,value in by_category.items()},
        "by_noise_level": {key:{"clip_weighted":by_level[key].report(),
                              "group_macro":_group_macro(by_level_group[key])} for key in LEVELS},
        "condition_columns": [*CATEGORIES,"null"],
        "truth_rows": list(CATEGORIES),
        "error_matrices_by_noise_level": {
            name:[(matrix[i,j]/counts[i,j]).tolist() if counts[i,j] else None
                  for j in range(len(CATEGORIES))] for i,name in enumerate(LEVELS)
        },
        "tie_absolute_tolerance": TIE_ATOL,
        "paired_corruption": True, "test_split_scored": False,
        "model_state_unchanged": True, "optimizer_steps_executed": 0,
        "generated_audio_semantics_established": False,
        "model_publication_authorized": False,
    }



def compare_initialization(initial: dict, trained: dict) -> dict:
    """Compute paired descriptive deltas; do not rank or infer significance."""
    pins = ("protocol", "archive_sha256", "eligible_validation_sha256", "seed",
            "repeats", "batch_size", "diffusion_profile", "diffusion_steps",
            "timesteps", "eligible_validation_clips", "prediction_rows",
            "noise_trace_sha256", "condition_columns", "truth_rows",
            "tie_absolute_tolerance")
    if any(initial[key] != trained[key] for key in pins):
        raise ArchiveError("Initialization comparison uses a different assay panel")
    keys = ("matched_mse", "mean_wrong_mse", "hardest_wrong_mse", "null_mse",
            "mean_wrong_minus_matched", "hardest_wrong_minus_matched",
            "null_minus_matched", "mean_rank", "strict_win_fraction")

    def difference(a, b):
        if a.get("cases") != b.get("cases") or a.get("recording_groups") != b.get("recording_groups"):
            raise ArchiveError("Initialization comparison coverage changed")
        return {key: (b[key]-a[key] if a[key] is not None and b[key] is not None else None)
                for key in keys if key in a and key in b}

    return {
        "initial_model_sha256": initial["model_sha256"],
        "trained_model_sha256": trained["model_sha256"],
        "noise_trace_sha256": trained["noise_trace_sha256"],
        "delta_convention": "trained_minus_initial",
        "clip_weighted": difference(initial["clip_weighted"], trained["clip_weighted"]),
        "group_macro": difference(initial["group_macro"], trained["group_macro"]),
        "by_category": {name: difference(initial["by_category"][name], trained["by_category"][name])
                        for name in CATEGORIES},
        "by_noise_level": {
            level: {kind: difference(initial["by_noise_level"][level][kind],
                                     trained["by_noise_level"][level][kind])
                    for kind in ("clip_weighted", "group_macro")}
            for level in LEVELS
        },
        "matched_corruption_trace_verified": True,
        "statistical_significance_claimed": False,
        "generated_audio_semantics_established": False,
    }


def assay_checkpoint(archive, *, archive_sha256, checkpoint, checkpoint_sha256,
                     training: ArchiveTrainingConfig, config: AssayConfig | None = None,
                     compare_initial: bool = False):
    """Evaluate one digest-pinned training checkpoint without updating it."""
    require_sha256(archive_sha256)
    require_sha256(checkpoint_sha256)
    if not isinstance(training,ArchiveTrainingConfig):
        raise ArchiveError("Expected ArchiveTrainingConfig")
    if type(compare_initial) is not bool:
        raise ArchiveError("Expected boolean initialization comparison flag")
    with CorpusArchiveReader(archive,expected_sha256=archive_sha256) as reader:
        readiness = audit_archive(reader)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(training.seed)
            with torch.device("cpu"):
                model = TinyWaveDenoiser(training.channels)
                schedule = DiffusionSchedule(training.diffusion_steps,profile=training.diffusion_profile)
            optimizer = make_archive_optimizer(model)
            initial_hash = model_fingerprint(model)
            initial = (evaluate_conditioning(reader, readiness, model, schedule, config=config)
                       if compare_initial else None)
            restored = restore_training_checkpoint(
                checkpoint,expected_sha256=checkpoint_sha256,config=training,
                readiness=readiness,initial_model_sha256=initial_hash,
                model=model,optimizer=optimizer,
            )
            result = evaluate_conditioning(reader,readiness,model,schedule,config=config)
    return {**result,"checkpoint_sha256":restored.sha256,
            "completed_training_steps":restored.progress.completed_steps,
            "files_written":0,"original_sonumio_1":False,
            "initialization_assay": initial,
            "initialization_comparison": compare_initialization(initial,result) if initial is not None else None}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--archive-sha256",required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    for field in ("seed","steps","batch_size","channels","diffusion_steps","validation_seed",
                  "validation_repeats","validation_batch_size"):
        parser.add_argument("--"+field.replace("_","-"),type=int,
                            default=getattr(ArchiveTrainingConfig(),field))
    parser.add_argument("--diffusion-profile",default="linear_beta_v1",choices=DiffusionSchedule.PROFILES)
    parser.add_argument("--assay-seed",type=int,default=301)
    parser.add_argument("--assay-repeats",type=int,default=2)
    parser.add_argument("--assay-batch-size",type=int,default=4)
    parser.add_argument("--compare-initial",action="store_true",help="Evaluate matched seeded initialization before checkpoint restore")
    args=parser.parse_args(argv)
    try:
        training=ArchiveTrainingConfig(**{key:getattr(args,key) for key in ArchiveTrainingConfig.__dataclass_fields__})
        result=assay_checkpoint(args.archive,archive_sha256=args.archive_sha256,
                               checkpoint=args.checkpoint,checkpoint_sha256=args.checkpoint_sha256,
                               training=training,config=AssayConfig(args.assay_seed,args.assay_repeats,args.assay_batch_size),
                               compare_initial=args.compare_initial)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Archive conditioning assay: {exc}\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
