"""Factorial comparison of sustained-note edge coordinates and objectives.

Four arms use the same parameter count, initial state, source corpus,
window schedule, optimizer and update budget. Only the edge mapping
and the added phase-insensitive RMS-envelope term vary. The existing
v1 sustained research protocol and production renderer are unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass,replace
import json
import statistics

import torch

from research.instrument_harmonic_model import WIDTHS
from research.instrument_long_data import (
    LongGrid, Corpus, INSTRUMENTS, REGIONS, WINDOW, MAX_BATCH, MAX_UPDATES,
    build_corpus,collate,region_span,
)
from research.instrument_long_model import WindowedHarmonicInstrument,render_complete,sine_window
from research.instrument_edge_data import Schedule,build_schedule,describe_schedule
from research.instrument_edge_model import (
    COORDINATES,OBJECTIVES,initialize_edges,edge_training_loss,
    waveform_diagnostics,edge_parameters,
)
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-sustained-edge-factorial-v1"
ARMS=tuple((coordinate,objective) for coordinate in COORDINATES for objective in OBJECTIVES)
PRIMARY="absolute-envelope"
BASELINE="fraction-waveform"
EVALUATION_PANELS=("duration","joint")
MEASURE_REGIONS=("full","attack","sustain","release","first2048","last2048")
METRICS=("relative_wave_mse","absolute_wave_mse","rms_error","rms_ratio","envelope_error","reference_rms")
BLOCK=4096
MAX_TRAIN=25_000_000
MAX_EVAL=110_000_000


def arm_name(coordinate,objective):
    return coordinate+"-"+objective


@dataclass(frozen=True,slots=True)
class StudyPlan:
    grid:LongGrid=LongGrid()
    seeds:tuple[int,...]=(17,29,41)
    sampler_seed:int=181
    width:int=48
    updates:int=96
    batch_size:int=6
    panels:tuple[str,...]=EVALUATION_PANELS

    def __post_init__(self):
        if not isinstance(self.grid,LongGrid):
            raise ValueError("Expected sustained edge source grid")
        if (type(self.seeds) is not tuple or not 2<=len(self.seeds)<=4
                or len(set(self.seeds))!=len(self.seeds)
                or any(type(s) is not int or not 0<=s<2**32 for s in self.seeds)
                or type(self.sampler_seed) is not int or not 0<=self.sampler_seed<2**32
                or type(self.width) is not int or self.width not in WIDTHS
                or type(self.updates) is not int or not 1<=self.updates<=MAX_UPDATES
                or type(self.batch_size) is not int or not 1<=self.batch_size<=MAX_BATCH
                or type(self.panels) is not tuple or not self.panels
                or self.panels!=tuple(p for p in EVALUATION_PANELS if p in self.panels)):
            raise ValueError("Invalid sustained edge study configuration")
        if self.updates*self.batch_size*WINDOW*len(ARMS)*len(self.seeds)>MAX_TRAIN:
            raise ValueError("Sustained edge study training budget exceeded")
        samples=sum(n.frames for p in self.panels for n in self.grid.specs(p))
        # Sine once, two initial coordinate arms, four trained arms and
        # all wrong labels for the primary arm in each seed.
        if samples*(1+len(self.seeds)*(2+4+len(INSTRUMENTS)-1))>MAX_EVAL:
            raise ValueError("Sustained edge study evaluation budget exceeded")


def train(model,corpus,plan,schedule,objective):
    if (not isinstance(model,WindowedHarmonicInstrument)
            or not isinstance(corpus,Corpus) or corpus.grid!=plan.grid
            or not isinstance(schedule,Schedule) or objective not in OBJECTIVES):
        raise ValueError("Invalid factorial edge training context")
    rows=corpus.get("train")
    expected=build_schedule(rows,updates=plan.updates,batch_size=plan.batch_size,seed=plan.sampler_seed)
    if schedule!=expected:
        raise ValueError("Edge training schedule differs from prescribed source")
    initial=model_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    losses=[]
    by_region={region:[] for region in REGIONS}
    model.train()
    for batch in schedule.batches:
        labels,pitches,velocities,target=collate(rows,batch)
        optimizer.zero_grad(set_to_none=True)
        predicted=model.render_window(labels,pitches,velocities,batch.frames,batch.start,batch.count)
        loss=edge_training_loss(predicted,target,objective)
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise ValueError("Invalid factorial edge gradient")
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError("Invalid factorial edge optimizer update")
        row={"loss":float(loss.detach()),"gradient_norm":float(norm)}
        losses.append(row)
        by_region[batch.region].append(row["loss"])
    trained=model_state_sha256(model)
    if initial==trained or len(losses)!=plan.updates:
        raise ValueError("Factorial edge training failed to update model")
    return {
        "objective":objective,"initial_sha256":initial,"trained_sha256":trained,
        "schedule_sha256":schedule.sha256,"updates":len(losses),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "first_loss":losses[0]["loss"],"last_loss":losses[-1]["loss"],
        "mean_loss":statistics.fmean(x["loss"] for x in losses),
        "max_gradient_norm":max(x["gradient_norm"] for x in losses),
        "mean_loss_by_region":{name:statistics.fmean(values) if values else None
                               for name,values in by_region.items()},
    }


def _spans(frames):
    return {"full":(0,frames),**{name:region_span(frames,name) for name in REGIONS},
            "first2048":(0,2048),"last2048":(frames-2048,frames)}


@torch.no_grad()
def evaluate_note(example,model=None,*,wrong_offset=0):
    if type(wrong_offset) is not int or not 0<=wrong_offset<len(INSTRUMENTS):
        raise ValueError("Invalid edge wrong-label offset")
    if (model is None and wrong_offset) or (model is not None and not isinstance(model,WindowedHarmonicInstrument)):
        raise ValueError("Invalid edge evaluation model")
    note=example.note
    labels=torch.tensor([(INSTRUMENTS.index(note.instrument)+wrong_offset)%len(INSTRUMENTS)],dtype=torch.long)
    pitches=torch.tensor([note.pitch],dtype=torch.long)
    velocities=torch.tensor([note.velocity_milli],dtype=torch.long)
    blocks=[]
    cursor=0
    if model is None:
        iterator=((start,sine_window(labels,pitches,velocities,note.frames,start,min(BLOCK,note.frames-start)))
                  for start in range(0,note.frames,BLOCK))
        parameters={"attack_frames":160.,"release_frames":1200.}
    else:
        iterator=render_complete(model,labels,pitches,velocities,note.frames,block=BLOCK)
        parameters={key:float(value[0]) for key,value in edge_parameters(
            model,labels,pitches,velocities,note.frames).items()}
    for start,block in iterator:
        if (start!=cursor or block.shape!=(1,min(BLOCK,note.frames-start))
                or not bool(torch.isfinite(block).all())):
            raise ValueError("Sustained edge evaluation block geometry differs")
        blocks.append(block)
        cursor+=block.shape[1]
    if cursor!=note.frames:
        raise ValueError("Sustained edge evaluation did not cover full note")
    predicted=torch.cat(blocks,dim=1)
    target=torch.from_numpy(example.samples.copy())[None,:]
    result={}
    for region,(lo,hi) in _spans(note.frames).items():
        metrics=waveform_diagnostics(predicted[:,lo:hi],target[:,lo:hi])
        result[region]={key:float(value[0]) for key,value in metrics.items()}
    return {"regions":result,"parameters":parameters}


def _average(rows):
    if not rows:
        raise ValueError("Empty edge evaluation aggregation")
    return {
        "regions":{name:{metric:statistics.fmean(row["regions"][name][metric] for row in rows)
                        for metric in METRICS} for name in MEASURE_REGIONS},
        "parameters":{name:statistics.fmean(row["parameters"][name] for row in rows)
                      for name in ("attack_frames","release_frames")},
    }


def evaluate_panel(rows,model=None,*,wrong=False):
    if type(wrong) is not bool or (wrong and model is None):
        raise ValueError("Invalid edge evaluation condition")
    state=model_state_sha256(model) if model is not None else None
    mode=model.training if model is not None else None
    if model is not None:
        model.eval()
    groups={name:[] for name in INSTRUMENTS}
    try:
        for row in rows:
            if wrong:
                metrics=_average([evaluate_note(row,model,wrong_offset=i) for i in range(1,len(INSTRUMENTS))])
            else:
                metrics=evaluate_note(row,model)
            groups[row.note.instrument].append(metrics)
    finally:
        if model is not None:
            model.train(mode)
    if model is not None and model_state_sha256(model)!=state:
        raise ValueError("Edge evaluation mutated model weights")
    by_instrument={name:_average(values) for name,values in groups.items()}
    return {"notes":len(rows),"frames":sum(row.note.frames for row in rows),
            "by_instrument":by_instrument,"macro":_average(list(by_instrument.values())),
            "wrong_labels_per_note":len(INSTRUMENTS)-1 if wrong else 0}


def evaluate(corpus,plan,model=None,*,wrong=False):
    return {name:evaluate_panel(corpus.get(name),model,wrong=wrong) for name in plan.panels}


def _metric(arm,panel,region,metric):
    return arm[panel]["macro"]["regions"][region][metric]


def _contrasts(initial,trained,sine,wrong,panels):
    result={}
    for panel in panels:
        result[panel]={}
        for region in MEASURE_REGIONS:
            result[panel][region]={}
            for metric in ("relative_wave_mse","envelope_error","rms_error"):
                a=_metric(trained["fraction-waveform"],panel,region,metric)
                b=_metric(trained["absolute-waveform"],panel,region,metric)
                c=_metric(trained["fraction-envelope"],panel,region,metric)
                d=_metric(trained[PRIMARY],panel,region,metric)
                result[panel][region][metric]={
                    "baseline_minus_primary":a-d,
                    "fraction_minus_absolute_waveform":a-b,
                    "fraction_minus_absolute_envelope":c-d,
                    "waveform_minus_envelope_fraction":a-c,
                    "waveform_minus_envelope_absolute":b-d,
                    "primary_initial_minus_trained":_metric(initial["absolute"],panel,region,metric)-d,
                    "sine_minus_primary":_metric(sine,panel,region,metric)-d,
                    "wrong_minus_matched_primary":_metric(wrong,panel,region,metric)-d,
                }
    return result


def _aggregate(replicates,panels):
    summary={}
    for panel in panels:
        summary[panel]={}
        for region in MEASURE_REGIONS:
            summary[panel][region]={}
            for metric in ("relative_wave_mse","envelope_error","rms_error"):
                first=replicates[0]["contrasts"][panel][region][metric]
                summary[panel][region][metric]={}
                for key in first:
                    values=[r["contrasts"][panel][region][metric][key] for r in replicates]
                    summary[panel][region][metric][key]={
                        "mean":statistics.fmean(values),"min":min(values),"max":max(values),
                        "positive_seeds":sum(v>0 for v in values),
                    }
    return summary


def run_study(plan=None):
    plan=StudyPlan() if plan is None else plan
    if not isinstance(plan,StudyPlan):
        raise ValueError("Expected sustained edge StudyPlan")
    corpus=build_corpus(plan.grid)
    schedule=build_schedule(corpus.get("train"),updates=plan.updates,batch_size=plan.batch_size,seed=plan.sampler_seed)
    sine=evaluate(corpus,plan)
    replicates=[]
    for seed in plan.seeds:
        initial={}
        trained={}
        training={}
        primary_model=None
        initial_hash=None
        parameters=None
        for coordinate,objective in ARMS:
            name=arm_name(coordinate,objective)
            model=initialize_edges(plan.width,seed,coordinate)
            state=model_state_sha256(model)
            count=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if initial_hash is None:
                initial_hash=state
                parameters=count
            elif state!=initial_hash or parameters!=count:
                raise ValueError("Factorial arms do not share initialization or parameter count")
            if coordinate not in initial:
                initial[coordinate]=evaluate(corpus,plan,model)
            training[name]=train(model,corpus,plan,schedule,objective)
            trained[name]=evaluate(corpus,plan,model)
            if name==PRIMARY:
                primary_model=model
        if primary_model is None:
            raise ValueError("Primary edge model missing")
        wrong=evaluate(corpus,plan,primary_model,wrong=True)
        contrasts=_contrasts(initial,trained,sine,wrong,plan.panels)
        replicates.append({"seed":seed,"initial_sha256":initial_hash,"trainable_parameters":parameters,
                           "initial":initial,"trained":trained,"training":training,
                           "wrong_primary":wrong,"contrasts":contrasts})
    return {"protocol":PROTOCOL,"plan":asdict(plan),"sources":corpus.receipt(),
            "schedule":describe_schedule(schedule,corpus.get("train")),
            "sine":sine,"replicates":replicates,"summary":_aggregate(replicates,plan.panels),
            "same_initial_weights_per_seed":True,"same_training_schedule":True,
            "evaluation_used_for_training":False,"external_recordings_used":False,
            "weights_published":False,"perceptual_quality_established":False,
            "generated_audio_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--updates",type=int,default=96)
    args=parser.parse_args(argv)
    try:
        plan=StudyPlan(updates=args.updates)
        if args.quick:
            plan=replace(plan,updates=min(plan.updates,12),panels=("joint",))
        print(json.dumps(run_study(plan),sort_keys=True,allow_nan=False))
    except (ValueError,RuntimeError) as exc:
        parser.exit(2,f"Edge factorial experiment failed: {exc}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
