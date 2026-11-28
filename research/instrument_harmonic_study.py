"""Replicated held-out waveform reconstruction study for learned instruments.

The study fixes a first-party synthetic corpus, source split, sampler order
and optimizer budget before training. It compares learned reconstruction
against initialization, a fixed sine and every wrong instrument condition.
It never saves audio, checkpoints or weights. None of these numerical
controls establishes human-perceived instrument or music quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import statistics

import numpy as np
import torch

from research.instrument_note_data import (
    INSTRUMENTS, PANELS, MAX_BATCH, MAX_UPDATES, GridPlan, NoteCorpus,
    build_corpus, collate, evaluation_batches, training_schedule,
)
from research.instrument_harmonic_model import (
    WIDTHS, HarmonicInstrument, initialize_model, fixed_sine, instrument_loss, note_metrics,
)
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-learned-instrument-reconstruction-study-v1"
EVALUATION_PANELS = PANELS[1:]
METRICS = ("relative_wave_mse", "relative_spectral_error", "rms_ratio",
           "waveform_cosine", "peak", "clipping_fraction")
ERROR_METRICS = METRICS[:2]
ARMS = ("fixed_sine", "initial", "trained", "wrong_instrument")
MAX_TRAIN_SAMPLES = 40_000_000
MAX_EVALUATION_SAMPLES = 80_000_000


def _int(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Invalid instrument study " + label)
    return value


def _seeds(values):
    if (type(values) is not tuple or not 2 <= len(values) <= 5
            or any(type(x) is not int or not 0 <= x < 2**32 for x in values)
            or len(set(values)) != len(values)):
        raise ValueError("Invalid independent instrument initialization seeds")


@dataclass(frozen=True, slots=True)
class StudyPlan:
    model_seeds: tuple[int, ...] = (17, 29, 41)
    sampler_seed: int = 271
    width: int = 48
    updates: int = 96
    batch_size: int = 8
    evaluation_batch_size: int = 8
    grid: GridPlan = GridPlan()

    def __post_init__(self):
        _seeds(self.model_seeds)
        _int(self.sampler_seed, 0, 2**32 - 1, "sampler seed")
        if type(self.width) is not int or self.width not in WIDTHS:
            raise ValueError("Invalid instrument study network width")
        _int(self.updates, 1, MAX_UPDATES, "optimizer updates")
        _int(self.batch_size, 1, MAX_BATCH, "batch size")
        _int(self.evaluation_batch_size, 1, MAX_BATCH, "evaluation batch size")
        if not isinstance(self.grid, GridPlan):
            raise ValueError("Expected instrument study GridPlan")
        if self.training_sample_budget > MAX_TRAIN_SAMPLES:
            raise ValueError("Instrument study exceeds training sample budget")
        if self.evaluation_sample_budget > MAX_EVALUATION_SAMPLES:
            raise ValueError("Instrument study exceeds evaluation sample budget")

    @property
    def training_sample_budget(self):
        return len(self.model_seeds) * self.updates * self.batch_size * max(self.grid.training_frames)

    @property
    def evaluation_sample_budget(self):
        total = sum(spec.frames for name in EVALUATION_PANELS
                    for spec in self.grid.specifications(name))
        # Each seed: initial + trained + all other identities. One shared
        # fixed sine evaluation. Count the actual waveform comparisons.
        return total * (1 + len(self.model_seeds) * (2 + len(INSTRUMENTS) - 1))


class _Accumulator:
    """Group-macro note metrics; every note and instrument counted once."""

    def __init__(self):
        self.counts = dict.fromkeys(INSTRUMENTS, 0)
        self.values = {name: {metric: [] for metric in METRICS} for name in INSTRUMENTS}

    def append(self, batch, metrics):
        if set(metrics) != set(METRICS):
            raise ValueError("Unexpected learned instrument metric schema")
        labels = batch.instruments.tolist()
        for metric, value in metrics.items():
            if (not isinstance(value, torch.Tensor) or value.shape != (len(labels),)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid per-note metric tensor")
            for label, number in zip(labels, value.tolist(), strict=True):
                self.values[INSTRUMENTS[label]][metric].append(float(number))
        for label in labels:
            self.counts[INSTRUMENTS[label]] += 1

    def report(self, expected):
        if any(count <= 0 for count in self.counts.values()) or sum(self.counts.values()) != expected:
            raise ValueError("Incomplete held-out instrument evaluation")
        groups = {}
        for name, count in self.counts.items():
            if any(len(rows) != count for rows in self.values[name].values()):
                raise ValueError("Incomplete instrument metric group")
            groups[name] = {"notes": count, **{
                metric: statistics.fmean(self.values[name][metric]) for metric in METRICS
            }}
        return {
            "notes": expected,
            "by_instrument": groups,
            "instrument_macro": {
                metric: statistics.fmean(group[metric] for group in groups.values())
                for metric in METRICS
            },
        }


def _wave(model, batch, instruments):
    return model(instruments, batch.pitches, batch.velocities, batch.frames)


def _wrong_metrics(model, batch):
    # Exhaust all four incorrect labels while preserving pitch, velocity,
    # target waveform and exact duration. Average per note, not per label.
    result = None
    for offset in range(1, len(INSTRUMENTS)):
        incorrect = (batch.instruments + offset) % len(INSTRUMENTS)
        values = note_metrics(_wave(model, batch, incorrect), batch.targets)
        result = ({name: value.clone() for name, value in values.items()} if result is None
                  else {name: result[name] + value for name, value in values.items()})
    return {name: value / (len(INSTRUMENTS) - 1) for name, value in result.items()}


@torch.no_grad()
def evaluate_panel(examples, batch_size, *, model=None, arm):
    if arm not in ARMS or (arm == "fixed_sine" and model is not None):
        raise ValueError("Invalid instrument evaluation arm")
    if arm != "fixed_sine" and not isinstance(model, HarmonicInstrument):
        raise ValueError("Expected HarmonicInstrument for learned evaluation")
    previous_mode = None if model is None else model.training
    before = None if model is None else model_state_sha256(model)
    if model is not None:
        model.eval()
    accumulator = _Accumulator()
    visited = []
    try:
        for indices in evaluation_batches(examples, batch_size):
            batch = collate(tuple(examples[i] for i in indices))
            visited.extend(indices)
            if arm == "fixed_sine":
                predicted = fixed_sine(batch.instruments, batch.pitches, batch.velocities, batch.frames)
                metrics = note_metrics(predicted, batch.targets)
            elif arm == "wrong_instrument":
                metrics = _wrong_metrics(model, batch)
            else:
                metrics = note_metrics(_wave(model, batch, batch.instruments), batch.targets)
            accumulator.append(batch, metrics)
    finally:
        if model is not None:
            model.train(previous_mode)
    if sorted(visited) != list(range(len(examples))):
        raise ValueError("Held-out instrument note coverage changed")
    if model is not None and before != model_state_sha256(model):
        raise ValueError("Instrument evaluation changed neural state")
    result = accumulator.report(len(examples))
    result["wrong_conditions_per_note"] = len(INSTRUMENTS) - 1 if arm == "wrong_instrument" else 0
    return result


def evaluate_corpus(corpus, plan, *, model=None, arm):
    if not isinstance(corpus, NoteCorpus) or not isinstance(plan, StudyPlan) or corpus.plan != plan.grid:
        raise ValueError("Invalid instrument study corpus")
    return {
        name: evaluate_panel(corpus.panel(name), plan.evaluation_batch_size, model=model, arm=arm)
        for name in EVALUATION_PANELS
    }


def train_model(model, corpus, plan, schedule):
    if (not isinstance(model, HarmonicInstrument) or not isinstance(corpus, NoteCorpus)
            or not isinstance(plan, StudyPlan) or corpus.plan != plan.grid):
        raise ValueError("Invalid learned instrument training context")
    train = corpus.panel("train")
    expected, schedule_sha = training_schedule(train, updates=plan.updates,
                                               batch_size=plan.batch_size, seed=plan.sampler_seed)
    if schedule != expected:
        raise ValueError("Instrument optimizer schedule differs from prescribed training")
    initial = model_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    model.train()
    losses = []
    sample_count = 0
    for indices in schedule:
        batch = collate(tuple(train[i] for i in indices))
        optimizer.zero_grad(set_to_none=True)
        prediction = _wave(model, batch, batch.instruments)
        loss = instrument_loss(prediction, batch.targets)
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise ValueError("Invalid learned instrument gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        # Fail immediately rather than carrying a nonfinite optimizer step
        # into later updates or reporting successful training.
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError("Nonfinite learned instrument parameter after update")
        losses.append(float(loss.detach()))
        sample_count += len(indices) * batch.frames
    final = model_state_sha256(model)
    if final == initial:
        raise ValueError("Instrument training did not update model weights")
    if len(losses) != plan.updates or any(not math.isfinite(loss) for loss in losses):
        raise ValueError("Invalid instrument training loss trace")
    return {
        "initial_model_sha256": initial, "trained_model_sha256": final,
        "optimizer_updates": len(losses), "note_examples_seen": len(schedule) * plan.batch_size,
        "waveform_samples_seen": sample_count, "schedule_sha256": schedule_sha,
        "first_objective": losses[0], "last_objective": losses[-1],
        "mean_training_objective": statistics.fmean(losses),
        "final_eight_objective": statistics.fmean(losses[-8:]),
        "optimizer": "AdamW", "learning_rate": .002, "gradient_norm_limit": 2.,
        "objective": "relative waveform MSE + 0.25 * relative compressed-STFT magnitude error",
    }


def paired_effects(arms):
    if set(arms) != set(ARMS):
        raise ValueError("Incomplete instrument evaluation controls")
    results = {}
    for panel in EVALUATION_PANELS:
        groups = {arm: arms[arm][panel] for arm in ARMS}
        if len({group["notes"] for group in groups.values()}) != 1:
            raise ValueError("Instrument control arms use different note counts")
        effects = {}
        for metric in ERROR_METRICS:
            values = {arm: group["instrument_macro"][metric] for arm, group in groups.items()}
            # Positive means lower error for the trained matched condition.
            effects[metric] = {
                "initial_minus_trained": values["initial"] - values["trained"],
                "sine_minus_trained": values["fixed_sine"] - values["trained"],
                "wrong_minus_matched": values["wrong_instrument"] - values["trained"],
            }
        results[panel] = effects
    return results


def aggregate_effects(rows):
    if not rows:
        raise ValueError("Missing independent instrument study seeds")
    results = {}
    for panel in EVALUATION_PANELS:
        results[panel] = {}
        for metric in ERROR_METRICS:
            results[panel][metric] = {}
            for name in ("initial_minus_trained", "sine_minus_trained", "wrong_minus_matched"):
                values = [row["paired_effects"][panel][metric][name] for row in rows]
                results[panel][metric][name] = {
                    "seed_values": values, "mean": statistics.fmean(values),
                    "minimum": min(values), "maximum": max(values),
                    "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
                    "positive_seeds": sum(x > 0 for x in values),
                }
    return results


def run_study(plan=None):
    plan = StudyPlan() if plan is None else plan
    if not isinstance(plan, StudyPlan):
        raise ValueError("Expected StudyPlan")
    corpus = build_corpus(plan.grid)
    schedule, schedule_sha = training_schedule(corpus.panel("train"), updates=plan.updates,
                                              batch_size=plan.batch_size, seed=plan.sampler_seed)
    fixed = evaluate_corpus(corpus, plan, arm="fixed_sine")
    rows = []
    for seed in plan.model_seeds:
        model = initialize_model(plan.width, seed)
        initial = evaluate_corpus(corpus, plan, model=model, arm="initial")
        training = train_model(model, corpus, plan, schedule)
        trained = evaluate_corpus(corpus, plan, model=model, arm="trained")
        wrong = evaluate_corpus(corpus, plan, model=model, arm="wrong_instrument")
        if training["schedule_sha256"] != schedule_sha:
            raise ValueError("Instrument study training schedules diverged")
        arms = {"fixed_sine": fixed, "initial": initial, "trained": trained,
                "wrong_instrument": wrong}
        rows.append({
            "model_seed": seed, "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "training": training, "evaluation": arms,
            "paired_effects": paired_effects(arms),
        })
    if len({row["training"]["initial_model_sha256"] for row in rows}) != len(rows):
        raise ValueError("Independent instrument seeds produced identical initial states")
    return {
        "protocol": PROTOCOL, "plan": asdict(plan), "source": corpus.report(),
        "training_sample_budget": plan.training_sample_budget,
        "evaluation_sample_budget": plan.evaluation_sample_budget,
        "shared_training_schedule_sha256": schedule_sha,
        "results": rows, "seed_summary": aggregate_effects(rows),
        "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__),
                    "device": "cpu", "torch_threads": torch.get_num_threads()},
        "evaluation_used_for_training": False,
        "wrong_instrument_conditions": len(INSTRUMENTS) - 1,
        "independent_initialization_replicates": len(rows),
        "statistical_significance_claimed": False,
        "checkpoint_written": False, "generated_audio_written": False,
        "model_weights_published": False, "perceptual_quality_established": False,
        "open_vocabulary_text_conditioning": False, "model_generation": None,
    }


def _parse_seeds(value):
    try:
        return tuple(int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seeds", type=_parse_seeds, default=(17, 29, 41))
    parser.add_argument("--sampler-seed", type=int, default=271)
    parser.add_argument("--width", type=int, choices=WIDTHS, default=48)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        report = run_study(StudyPlan(model_seeds=args.model_seeds, sampler_seed=args.sampler_seed,
                                    width=args.width, updates=args.updates, batch_size=args.batch_size,
                                    evaluation_batch_size=args.evaluation_batch_size))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Instrument reconstruction study failed: {exc}\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
