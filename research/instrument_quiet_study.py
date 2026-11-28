"""Matched sustained-note training with prespecified quiet-region objectives."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import statistics

import torch

from research.instrument_harmonic_model import WIDTHS
from research.instrument_long_data import (
    LongGrid, Corpus, INSTRUMENTS, REGIONS, WINDOW, MAX_BATCH, MAX_UPDATES,
    build_corpus, collate,
)
from research.instrument_edge_model import initialize_edges
from research.instrument_edge_data import build_schedule, describe_schedule, Schedule
from research.instrument_edge_study import evaluate, MEASURE_REGIONS
from research.instrument_quiet_loss import ARMS, POWER_FLOOR, loss_components
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-quiet-region-sustained-study-v1"
PANELS = ("duration","joint")
METRICS = ("relative_wave_mse","absolute_wave_mse","envelope_error","rms_error")
MAX_TRAIN_SAMPLES = 20_000_000
MAX_EVAL_SAMPLES = 95_000_000


@dataclass(frozen=True,slots=True)
class StudyPlan:
    grid: LongGrid = LongGrid()
    seeds: tuple[int,...] = (17,29,41)
    sampler_seed: int = 181
    width: int = 48
    updates: int = 96
    batch_size: int = 6
    panels: tuple[str,...] = PANELS

    def __post_init__(self):
        if not isinstance(self.grid,LongGrid):
            raise ValueError("Expected quiet-region source grid")
        if (type(self.seeds) is not tuple or not 2<=len(self.seeds)<=4
                or any(type(s) is not int or not 0<=s<2**32 for s in self.seeds)
                or len(set(self.seeds))!=len(self.seeds)
                or type(self.sampler_seed) is not int or not 0<=self.sampler_seed<2**32
                or type(self.width) is not int or self.width not in WIDTHS
                or type(self.updates) is not int or not 1<=self.updates<=MAX_UPDATES
                or type(self.batch_size) is not int or not 1<=self.batch_size<=MAX_BATCH
                or type(self.panels) is not tuple or not self.panels
                or self.panels!=tuple(p for p in PANELS if p in self.panels)):
            raise ValueError("Invalid quiet-region study configuration")
        if self.updates*self.batch_size*WINDOW*len(ARMS)*len(self.seeds)>MAX_TRAIN_SAMPLES:
            raise ValueError("Quiet-region training budget exceeded")
        frames=sum(note.frames for panel in self.panels for note in self.grid.specs(panel))
        if frames*(1+len(self.seeds)*(1+len(ARMS)+len(INSTRUMENTS)-1))>MAX_EVAL_SAMPLES:
            raise ValueError("Quiet-region evaluation budget exceeded")


def _mean(rows):
    if not rows:
        return None
    return {key:statistics.fmean(row[key] for row in rows) for key in rows[0]}


def train(model, corpus, plan, schedule, arm):
    if (not isinstance(corpus,Corpus) or corpus.grid!=plan.grid
            or not isinstance(schedule,Schedule) or arm not in ARMS):
        raise ValueError("Invalid quiet-region training context")
    rows=corpus.get("train")
    expected=build_schedule(rows,updates=plan.updates,batch_size=plan.batch_size,seed=plan.sampler_seed)
    if expected!=schedule:
        raise ValueError("Unexpected quiet-region training schedule")
    before=model_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    reports=[]
    regions={region:[] for region in REGIONS}
    model.train()
    for batch in schedule.batches:
        labels,pitches,velocities,target=collate(rows,batch)
        optimizer.zero_grad(set_to_none=True)
        predicted=model.render_window(labels,pitches,velocities,batch.frames,batch.start,batch.count)
        loss,parts=loss_components(predicted,target,arm)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Invalid quiet-region gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Nonfinite quiet-region optimizer state")
        report={key:float(value.detach()) for key,value in parts.items()}
        report.update(loss=float(loss.detach()), gradient_norm=float(norm),
                      gradient_clipped=float(norm>2.))
        reports.append(report)
        regions[batch.region].append(report)
    after=model_state_sha256(model)
    if before==after:
        raise ValueError("Quiet-region training did not update weights")
    return {"initial_sha256":before,"trained_sha256":after,"arm":arm,
            "updates":len(reports),"schedule_sha256":schedule.sha256,
            "mean":_mean(reports),"first":reports[0],"last":reports[-1],
            "by_region":{name:_mean(values) for name,values in regions.items()}}


def _metric(evaluation,panel,region,metric,instrument=None):
    entry=evaluation[panel]
    group=entry["macro"] if instrument is None else entry["by_instrument"][instrument]
    return group["regions"][region][metric]


def _contrasts(replica,panels):
    result={}
    for panel in panels:
        result[panel]={}
        for region in MEASURE_REGIONS:
            result[panel][region]={}
            for metric in METRICS:
                baseline=_metric(replica["trained"]["baseline"],panel,region,metric)
                floor=_metric(replica["trained"]["floor"],panel,region,metric)
                envelope=_metric(replica["trained"]["floor-envelope"],panel,region,metric)
                result[panel][region][metric]={
                    "baseline_minus_floor":baseline-floor,
                    "baseline_minus_floor_envelope":baseline-envelope,
                    "floor_minus_floor_envelope":floor-envelope,
                    "initial_minus_floor_envelope":_metric(replica["initial"],panel,region,metric)-envelope,
                    "wrong_minus_matched":_metric(replica["wrong"],panel,region,metric)-envelope,
                    "per_instrument_baseline_minus_floor_envelope":{
                        instrument:_metric(replica["trained"]["baseline"],panel,region,metric,instrument)
                        -_metric(replica["trained"]["floor-envelope"],panel,region,metric,instrument)
                        for instrument in INSTRUMENTS
                    },
                }
    return result


def _stats(values):
    return {"mean":statistics.fmean(values),"min":min(values),"max":max(values),
            "positive_seeds":sum(value>0 for value in values)}


def _aggregate(replicates,panels):
    result={}
    for panel in panels:
        result[panel]={}
        for region in MEASURE_REGIONS:
            result[panel][region]={}
            for metric in METRICS:
                rows=[r["contrasts"][panel][region][metric] for r in replicates]
                result[panel][region][metric]={
                    key:(_stats([row[key] for row in rows]) if type(rows[0][key]) is not dict
                         else {instrument:_stats([row[key][instrument] for row in rows])
                               for instrument in INSTRUMENTS})
                    for key in rows[0]
                }
    return result


def run_study(plan=None):
    plan=StudyPlan() if plan is None else plan
    if not isinstance(plan,StudyPlan):
        raise ValueError("Expected quiet-region StudyPlan")
    corpus=build_corpus(plan.grid)
    schedule=build_schedule(corpus.get("train"),updates=plan.updates,
                            batch_size=plan.batch_size,seed=plan.sampler_seed)
    sine=evaluate(corpus,plan)
    replicates=[]
    for seed in plan.seeds:
        initial=initialize_edges(plan.width,seed,"fraction")
        state=model_state_sha256(initial)
        replica={"seed":seed,"initial_sha256":state,"initial":evaluate(corpus,plan,initial),
                 "trained":{},"training":{}}
        for arm in ARMS:
            model=initialize_edges(plan.width,seed,"fraction")
            if model_state_sha256(model)!=state:
                raise ValueError("Quiet-region arms do not share initialization")
            replica["training"][arm]=train(model,corpus,plan,schedule,arm)
            replica["trained"][arm]=evaluate(corpus,plan,model)
            if arm=="floor-envelope":
                replica["wrong"]=evaluate(corpus,plan,model,wrong=True)
        replica["contrasts"]=_contrasts(replica,plan.panels)
        replicates.append(replica)
    return {"protocol":PROTOCOL,"plan":asdict(plan),"sources":corpus.receipt(),
            "schedule":describe_schedule(schedule,corpus.get("train")),
            "power_floor":POWER_FLOOR,"sine":sine,"replicates":replicates,
            "summary":_aggregate(replicates,plan.panels),
            "same_architecture":True,"same_initial_weights_per_seed":True,
            "same_training_schedule":True,"evaluation_used_for_training":False,
            "external_recordings_used":False,"weights_published":False,
            "perceptual_quality_established":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    args=parser.parse_args(argv)
    try:
        plan=StudyPlan()
        if args.quick:
            plan=replace(plan,updates=12,panels=("joint",))
        print(json.dumps(run_study(plan),sort_keys=True,allow_nan=False))
    except (ValueError,RuntimeError) as exc:
        parser.exit(2,f"Quiet-region experiment failed: {exc}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
