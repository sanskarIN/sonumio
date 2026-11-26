"""Paired native 24 kHz codec high-frequency objective study.

All arms receive identical source samples, batches and model initialization
within each seed. The 8 kHz DSP control measures actual bandwidth removal,
not a matched neural model. No weights or audio are persisted.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
import struct

import numpy as np
from scipy.signal import resample_poly
import torch

from research.native_codec import (
    Native24kCodec, FRAMES, SAMPLE_RATE, LATENT_FRAMES, LATENT_CHANNELS,
    HIGHBAND_HZ, MAX_BATCH, native_loss, native_metrics, validate_wave,
)
from research.crop_source_data import WINDOWS, DURATION
from research.wave_diffusion import CATEGORIES
from research.weight_ema import model_state_sha256
from sonumio.engines.procedural import generate

PROTOCOL = "sonumio-native24k-codec-ablation-v1"
SOURCE_PROTOCOL = "sonumio-native24k-event-aware-v1"
ARMS = {"base": 0., "highband": .5}
MAX_UPDATES = 256
MAX_TRAIN_EXAMPLES = 8192


def _seeds(values, minimum, maximum):
    if (type(values) is not tuple or not minimum <= len(values) <= maximum
            or any(type(v) is not int or not 0 <= v < 2**32 for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("Invalid native codec seed panel")


@dataclass(frozen=True, slots=True)
class StudyPlan:
    model_seeds: tuple[int, ...] = (17, 29, 41)
    training_seeds: tuple[int, ...] = (0, 1, 2)
    validation_seeds: tuple[int, ...] = (101, 102)
    updates: int = 32
    batch_size: int = 4

    def __post_init__(self):
        _seeds(self.model_seeds, 2, 4)
        _seeds(self.training_seeds, 1, 6)
        _seeds(self.validation_seeds, 1, 4)
        if set(self.training_seeds) & set(self.validation_seeds):
            raise ValueError("Source seeds overlap")
        if (type(self.updates) is not int or not 1 <= self.updates <= MAX_UPDATES
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= MAX_BATCH
                or self.train_examples > MAX_TRAIN_EXAMPLES):
            raise ValueError("Native codec training budget exceeded")

    @property
    def train_examples(self):
        return len(self.model_seeds)*len(ARMS)*self.updates*self.batch_size


def dataset(seeds):
    _seeds(seeds, 1, 6)
    waves, labels, identities = [], [], []
    trace = hashlib.sha256(SOURCE_PROTOCOL.encode("ascii")+b"\0")
    for label, category in enumerate(CATEGORIES):
        lower, upper = WINDOWS[category]
        # Windows were originally expressed on the 8 kHz crop clock.
        # Preserve the event-aware placement but use original 24 kHz PCM.
        low = 3*lower
        high = min(3*upper, int(DURATION*SAMPLE_RATE)-FRAMES)
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, label, 0x4E3234]))
            offset = int(rng.integers(low, high+1))
            source = generate(category, DURATION, seed)
            if source.sample_rate != SAMPLE_RATE or source.samples.shape != (int(DURATION*SAMPLE_RATE),):
                raise ValueError("Native source clock changed")
            crop = np.ascontiguousarray(source.samples[offset:offset+FRAMES], dtype=np.float32)
            if (crop.shape != (FRAMES,) or not np.isfinite(crop).all()
                    or np.max(np.abs(crop)) > 1.0001
                    or float(np.std(crop, dtype=np.float64)) < 1e-4):
                raise ValueError("Native source crop is silent or invalid")
            payload = np.asarray(crop, dtype="<f4").tobytes()
            identities.append(hashlib.sha256(payload).hexdigest())
            trace.update(struct.pack("<III", label, seed, offset))
            trace.update(payload)
            waves.append(torch.from_numpy(crop.copy()).reshape(1, FRAMES))
            labels.append(label)
    if len(set(identities)) != len(identities):
        raise ValueError("Exact duplicate native source crop")
    return torch.stack(waves), tuple(labels), tuple(identities), trace.hexdigest()


def lowband_control(wave):
    """Actual anti-aliased 24 -> 8 -> 24 kHz resampling; no clipping."""
    validate_wave(wave)
    raw = wave.detach().numpy().astype(np.float64)
    low = resample_poly(raw, 1, 3, axis=-1, window=("kaiser", 8.6))
    restored = resample_poly(low, 3, 1, axis=-1, window=("kaiser", 8.6))
    result = torch.from_numpy(restored.astype(np.float32))
    validate_wave(result, normalized=False)
    return result


def batch_plan(plan, seed, count):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches, pending = [], []
    for _ in range(plan.updates):
        while len(pending) < plan.batch_size:
            pending.extend(torch.randperm(count, generator=generator).tolist())
        batches.append(pending[:plan.batch_size])
        del pending[:plan.batch_size]
    trace = hashlib.sha256(np.asarray(batches, dtype="<i8").tobytes()).hexdigest()
    return batches, trace


def mean_metrics(rows):
    if not rows:
        raise ValueError("Empty native metric panel")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("Native metric schema changed")
    return {k: statistics.fmean(row[k] for row in rows) for k in keys}


@torch.no_grad()
def evaluate(wave, labels, model=None, *, control=None):
    if (model is None) == (control is None):
        raise ValueError("Choose exactly one evaluation reconstruction")
    before = model_state_sha256(model) if model is not None else None
    mode = model.training if model is not None else None
    if model is not None:
        model.eval()
    rows = []
    try:
        for i, label in enumerate(labels):
            source = wave[i:i+1]
            if model is not None:
                raw = model.decode_raw(model.encode(source))
                restored = raw.tanh()
            else:
                restored = control(source)
                raw = restored
            metrics = {k: float(v[0]) for k,v in native_metrics(source, restored).items()}
            metrics["pre_output_peak"] = float(raw.abs().max())
            metrics["pre_output_overrange_fraction"] = float((raw.abs()>1).float().mean())
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("Nonfinite evaluation report")
            rows.append({"category": CATEGORIES[label], "metrics": metrics})
        if len(rows) != len(wave):
            raise ValueError("Incomplete evaluation")
        if model is not None and model_state_sha256(model) != before:
            raise ValueError("Model changed during evaluation")
    finally:
        if model is not None:
            model.train(mode)
    return {"model_sha256": before, "clips": len(rows),
            "mean": mean_metrics([r["metrics"] for r in rows]),
            "by_category": {c: mean_metrics([r["metrics"] for r in rows if r["category"]==c])
                            for c in CATEGORIES}}


def train_pair(plan, seed, train, validation, labels):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            initial = Native24kCodec()
        identity = model_state_sha256(initial)
        batches, trace = batch_plan(plan, seed, len(train))
        initial_metrics = evaluate(validation, labels, initial)
        arms = {}
        for arm, weight in ARMS.items():
            model = copy.deepcopy(initial)
            if model_state_sha256(model) != identity:
                raise ValueError("Paired initialization mismatch")
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False, fused=False)
            losses = []
            model.train()
            for ids in batches:
                source = train[ids]
                optimizer.zero_grad(set_to_none=True)
                loss = native_loss(source, model(source), weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            trained = evaluate(validation, labels, model)
            if trained["model_sha256"] == identity:
                raise ValueError("Optimizer did not update codec")
            arms[arm] = {"weight": weight, "trained": trained,
                         "first_loss": losses[0], "last_loss": losses[-1]}
        return {"seed": seed, "initial_model_sha256": identity,
                "batch_trace_sha256": trace, "initial": initial_metrics,
                "parameters": sum(p.numel() for p in initial.parameters()), "arms": arms}


def run_study(plan=None):
    plan = StudyPlan() if plan is None else plan
    if not isinstance(plan, StudyPlan):
        raise ValueError("Expected StudyPlan")
    train, _, train_ids, train_hash = dataset(plan.training_seeds)
    validation, labels, valid_ids, valid_hash = dataset(plan.validation_seeds)
    if set(train_ids) & set(valid_ids):
        raise ValueError("Exact cross-split native PCM overlap")
    control = evaluate(validation, labels, control=lowband_control)
    silence = evaluate(validation, labels, control=torch.zeros_like)
    runs = [train_pair(plan, seed, train, validation, labels) for seed in plan.model_seeds]
    summary = {}
    for metric in runs[0]["initial"]["mean"]:
        values = [r["arms"]["highband"]["trained"]["mean"][metric] -
                  r["arms"]["base"]["trained"]["mean"][metric] for r in runs]
        summary[metric] = {"highband_minus_base_mean": statistics.fmean(values),
                           "per_seed_deltas": values,
                           "sample_stddev": statistics.stdev(values)}
    return {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL, "plan": asdict(plan),
        "sample_rate": SAMPLE_RATE, "clip_frames": FRAMES, "clip_seconds": FRAMES/SAMPLE_RATE,
        "latent_frames": LATENT_FRAMES, "latent_channels": LATENT_CHANNELS,
        "latent_scalar_compression": FRAMES/(LATENT_FRAMES*LATENT_CHANNELS),
        "highband_start_hz": HIGHBAND_HZ, "train_sha256": train_hash, "validation_sha256": valid_hash,
        "train_clips": len(train), "validation_clips": len(validation),
        "exact_cross_split_overlap": False,
        "matched_initialization_batches_and_architecture": True,
        "lowband_resample_control": control, "silence_control": silence,
        "runs": runs, "paired_seed_deltas": summary,
        "training_examples": plan.train_examples,
        "source_is_synthetic": True, "perceptual_quality_established": False,
        "neural_generation_tested": False, "files_written": 0,
    }


def _parse_seeds(value):
    parts = value.split(",")
    if any(not s.isascii() or not s.isdecimal() or len(s)>10 or str(int(s))!=s for s in parts):
        raise ValueError("Invalid canonical seed list")
    return tuple(int(s) for s in parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = StudyPlan()
    for field in ("model_seeds", "training_seeds", "validation_seeds"):
        parser.add_argument("--"+field.replace("_","-"), default=",".join(map(str,getattr(defaults,field))))
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    args = parser.parse_args(argv)
    try:
        plan = StudyPlan(_parse_seeds(args.model_seeds), _parse_seeds(args.training_seeds),
                         _parse_seeds(args.validation_seeds), args.updates, args.batch_size)
        report = run_study(plan)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Native codec study: {exc}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
