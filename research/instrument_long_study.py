"""Train sustained harmonic notes with bounded windows and full-note evaluation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
import statistics

import torch

from research.instrument_harmonic_model import instrument_loss, WIDTHS
from research.instrument_long_model import initialize_windowed, sine_window, WindowedHarmonicInstrument
from research.instrument_long_data import (
    LongGrid, Corpus, build_corpus, training_schedule, collate,
    INSTRUMENTS, PANELS, REGIONS, region_span, MAX_UPDATES, MAX_BATCH, WINDOW,
)
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-sustained-instrument-study-v1"
ARMS = ("sine", "initial", "trained", "wrong")
EVAL_BLOCK = 4096
MAX_TRAIN = 25_000_000
MAX_EVAL = 150_000_000


@dataclass(frozen=True, slots=True)
class StudyPlan:
    grid: LongGrid = LongGrid()
    seeds: tuple = (17, 29, 41)
    sampler_seed: int = 127
    width: int = 48
    updates: int = 120
    batch_size: int = 8

    def __post_init__(self):
        if not isinstance(self.grid, LongGrid):
            raise ValueError("Expected sustained note grid")
        if (type(self.seeds) is not tuple or not 2 <= len(self.seeds) <= 5
                or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("Invalid sustained note seeds")
        if (type(self.sampler_seed) is not int or not 0 <= self.sampler_seed < 2**32
                or type(self.width) is not int or self.width not in WIDTHS
                or type(self.updates) is not int or not 1 <= self.updates <= MAX_UPDATES
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= MAX_BATCH):
            raise ValueError("Invalid sustained study training configuration")
        if self.updates * self.batch_size * WINDOW * len(self.seeds) > MAX_TRAIN:
            raise ValueError("Sustained training sample budget exceeded")
        samples = sum(n.frames for panel in PANELS[1:] for n in self.grid.specs(panel))
        if samples * (1 + len(self.seeds) * (1 + len(INSTRUMENTS))) > MAX_EVAL:
            raise ValueError("Sustained evaluation sample budget exceeded")


def train(model, corpus, plan, schedule):
    if not isinstance(model, WindowedHarmonicInstrument) or not isinstance(corpus, Corpus) or corpus.grid != plan.grid:
        raise ValueError("Invalid sustained training context")
    rows = corpus.get("train")
    expected, trace = training_schedule(rows, updates=plan.updates, batch_size=plan.batch_size,
                                        seed=plan.sampler_seed)
    if schedule != expected:
        raise ValueError("Sustained window schedule differs from prescribed training")
    before = model_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    losses = []
    counts = dict.fromkeys(REGIONS, 0)
    model.train()
    for batch in schedule:
        labels, pitches, velocities, targets = collate(rows, batch)
        optimizer.zero_grad(set_to_none=True)
        predicted = model.render_window(labels, pitches, velocities, batch.frames, batch.start, batch.count)
        loss = instrument_loss(predicted, targets)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Invalid sustained gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Invalid sustained optimizer update")
        losses.append(float(loss.detach()))
        counts[batch.region] += len(batch.indices) * batch.count
    after = model_state_sha256(model)
    if before == after or len(losses) != plan.updates:
        raise ValueError("Sustained training failed to update weights")
    return {"initial_sha256": before, "trained_sha256": after, "schedule_sha256": trace,
            "updates": len(losses), "window_samples_by_region": counts,
            "first_loss": losses[0], "last_loss": losses[-1],
            "mean_loss": statistics.fmean(losses)}


class _Moments:
    def __init__(self):
        self.n = 0
        self.error = self.reference = self.predicted = self.dot = 0.

    def add(self, predicted, reference):
        a = predicted.to(torch.float64)
        b = reference.to(torch.float64)
        self.n += a.numel()
        self.error += float((a-b).square().sum())
        self.reference += float(b.square().sum())
        self.predicted += float(a.square().sum())
        self.dot += float((a*b).sum())

    def report(self):
        if not self.n or self.reference <= 1e-12:
            raise ValueError("Empty or silent sustained reference region")
        return {"samples": self.n, "relative_wave_mse": self.error/self.reference,
                "rms_ratio": math.sqrt(self.predicted/self.reference),
                "waveform_cosine": self.dot / math.sqrt(self.predicted*self.reference)
                if self.predicted > 1e-12 else 0.}


@torch.no_grad()
def evaluate_note(example, *, model=None, wrong_offset=0):
    if type(wrong_offset) is not int or not 0 <= wrong_offset < len(INSTRUMENTS):
        raise ValueError("Invalid sustained wrong-label control")
    if model is not None and not isinstance(model, WindowedHarmonicInstrument):
        raise ValueError("Expected windowed harmonic model")
    if model is None and wrong_offset:
        raise ValueError("Sine control does not use labels")
    note = example.note
    label = torch.tensor([(INSTRUMENTS.index(note.instrument)+wrong_offset) % len(INSTRUMENTS)], dtype=torch.long)
    pitch = torch.tensor([note.pitch], dtype=torch.long)
    velocity = torch.tensor([note.velocity_milli], dtype=torch.long)
    full = _Moments()
    regions = {}
    for region in REGIONS:
        moments = _Moments()
        lo, hi = region_span(note.frames, region)
        for start in range(lo, hi, EVAL_BLOCK):
            count = min(EVAL_BLOCK, hi-start)
            target = torch.from_numpy(example.samples[start:start+count].copy())
            output = (sine_window(label, pitch, velocity, note.frames, start, count)
                      if model is None else model.render_window(label, pitch, velocity, note.frames, start, count))[0]
            moments.add(output, target)
            full.add(output, target)
        regions[region] = moments.report()
    if full.n != note.frames:
        raise ValueError("Sustained evaluation did not visit every frame")
    return {"full": full.report(), **regions}


def _average(rows):
    return {region: {metric: statistics.fmean(row[region][metric] for row in rows)
                     for metric in ("relative_wave_mse", "rms_ratio", "waveform_cosine")}
            for region in ("full", *REGIONS)}


def evaluate_panel(rows, *, model=None, wrong=False):
    if type(wrong) is not bool or (wrong and model is None):
        raise ValueError("Invalid sustained evaluation control")
    before = model_state_sha256(model) if model is not None else None
    mode = model.training if model is not None else None
    if model is not None:
        model.eval()
    grouped = {name: [] for name in INSTRUMENTS}
    try:
        for row in rows:
            if wrong:
                result = _average([evaluate_note(row, model=model, wrong_offset=i)
                                   for i in range(1,len(INSTRUMENTS))])
            else:
                result = evaluate_note(row, model=model)
            grouped[row.note.instrument].append(result)
    finally:
        if model is not None:
            model.train(mode)
    if model is not None and before != model_state_sha256(model):
        raise ValueError("Sustained evaluation mutated model state")
    if any(not values for values in grouped.values()):
        raise ValueError("Incomplete sustained instrument coverage")
    by_instrument = {name: _average(values) for name, values in grouped.items()}
    return {"notes": len(rows), "frames": sum(row.note.frames for row in rows),
            "by_instrument": by_instrument,
            "macro": _average(list(by_instrument.values())),
            "wrong_conditions_per_note": len(INSTRUMENTS)-1 if wrong else 0}


def evaluate(corpus, *, model=None, wrong=False):
    return {name: evaluate_panel(corpus.get(name), model=model, wrong=wrong) for name in PANELS[1:]}


def paired(arms):
    return {panel: {region: {
        name: arms[control][panel]["macro"][region]["relative_wave_mse"] -
              arms["trained"][panel]["macro"][region]["relative_wave_mse"]
        for name, control in (("initial_minus_trained","initial"),
                              ("sine_minus_trained","sine"),
                              ("wrong_minus_matched","wrong"))}
        for region in ("full", *REGIONS)}
        for panel in PANELS[1:]}


def run_study(plan=None):
    plan = StudyPlan() if plan is None else plan
    if not isinstance(plan, StudyPlan):
        raise ValueError("Expected sustained study plan")
    corpus = build_corpus(plan.grid)
    schedule, trace = training_schedule(corpus.get("train"), updates=plan.updates,
                                        batch_size=plan.batch_size, seed=plan.sampler_seed)
    sine = evaluate(corpus)
    results = []
    for seed in plan.seeds:
        model = initialize_windowed(plan.width, seed)
        initial = evaluate(corpus, model=model)
        training = train(model, corpus, plan, schedule)
        trained = evaluate(corpus, model=model)
        wrong = evaluate(corpus, model=model, wrong=True)
        arms = {"sine": sine, "initial": initial, "trained": trained, "wrong": wrong}
        results.append({"seed": seed, "training": training, "evaluation": arms, "paired": paired(arms)})
    summary = {}
    for panel in PANELS[1:]:
        summary[panel] = {}
        for region in ("full", *REGIONS):
            summary[panel][region] = {}
            for key in ("initial_minus_trained", "sine_minus_trained", "wrong_minus_matched"):
                values = [result["paired"][panel][region][key] for result in results]
                summary[panel][region][key] = {"mean": statistics.fmean(values),
                    "minimum": min(values), "maximum": max(values),
                    "positive_seeds": sum(v > 0 for v in values)}
    return {"protocol": PROTOCOL, "plan": asdict(plan), "source": corpus.receipt(),
            "schedule_sha256": trace, "results": results, "summary": summary,
            "evaluation_used_for_training": False, "full_note_evaluation": True,
            "external_recordings_used": False, "model_weights_published": False,
            "perceptual_quality_established": False, "checkpoint_written": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        report = run_study(StudyPlan(updates=args.updates, batch_size=args.batch_size))
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Sustained note experiment failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
