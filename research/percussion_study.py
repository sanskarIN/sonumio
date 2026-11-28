"""Replicated percussion reconstruction of feature distributions, not source phase."""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import json
import statistics

import torch

from research.percussion_data import (
    DRUMS,PANELS,DrumGrid,DrumCorpus,build_corpus,collate,schedule,
    evaluation_batches,MAX_UPDATES,MAX_BATCH,
)
from research.percussion_model import (
    PercussionModel,initialize,broadband_control,objective,diagnostics,WIDTHS,FEATURE_NAMES,
)
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-percussion-feature-study-v1"
METRICS=(*FEATURE_NAMES,"objective","peak","rms_ratio")
ARMS=("broadband","initial","trained","wrong")
MAX_TRAIN_SAMPLES=30_000_000
MAX_EVAL_SAMPLES=80_000_000


@dataclass(frozen=True,slots=True)
class StudyPlan:
    grid:DrumGrid=DrumGrid()
    seeds:tuple=(17,29,41)
    sampler_seed:int=157
    width:int=48
    updates:int=128
    batch_size:int=8
    evaluation_batch_size:int=8
    excitation_replicates:int=2

    def __post_init__(self):
        if not isinstance(self.grid,DrumGrid):
            raise ValueError("Expected DrumGrid")
        if (type(self.seeds) is not tuple or not 2<=len(self.seeds)<=5
                or any(type(s) is not int or not 0<=s<2**32 for s in self.seeds)
                or len(set(self.seeds))!=len(self.seeds)):
            raise ValueError("Invalid percussion model seeds")
        for value,lo,hi in ((self.sampler_seed,0,2**32-1),(self.updates,1,MAX_UPDATES),
                            (self.batch_size,1,MAX_BATCH),(self.evaluation_batch_size,1,MAX_BATCH),
                            (self.excitation_replicates,1,4)):
            if type(value) is not int or not lo<=value<=hi:
                raise ValueError("Invalid percussion study configuration")
        if type(self.width) is not int or self.width not in WIDTHS:
            raise ValueError("Invalid percussion model width")
        if self.updates*self.batch_size*max(self.grid.training_frames)*len(self.seeds)>MAX_TRAIN_SAMPLES:
            raise ValueError("Percussion training sample budget exceeded")
        n=sum(spec.frames for panel in PANELS[1:] for spec in self.grid.specs(panel))
        if n*self.excitation_replicates*(1+len(self.seeds)*(1+len(DRUMS)))>MAX_EVAL_SAMPLES:
            raise ValueError("Percussion evaluation sample budget exceeded")


def train(model,corpus,plan,batches):
    if not isinstance(model,PercussionModel) or not isinstance(corpus,DrumCorpus) or corpus.grid!=plan.grid:
        raise ValueError("Invalid percussion training context")
    rows=corpus.get("train")
    expected,trace=schedule(rows,updates=plan.updates,batch_size=plan.batch_size,seed=plan.sampler_seed)
    if batches!=expected:
        raise ValueError("Percussion schedule differs from prescribed samples")
    initial=model_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train()
    losses=[]
    samples=0
    for step,indices in enumerate(batches):
        batch=collate(tuple(rows[i] for i in indices),domain="train",replica=step)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(batch.labels,batch.velocities,batch.frames,batch.excitation)
        loss=objective(prediction,batch.targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite percussion objective")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Invalid percussion gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Invalid percussion update")
        losses.append(float(loss.detach()))
        samples+=len(indices)*batch.frames
    final=model_state_sha256(model)
    if initial==final or len(losses)!=plan.updates:
        raise ValueError("Percussion training did not update model")
    return {"initial_sha256":initial,"trained_sha256":final,"schedule_sha256":trace,
            "updates":len(losses),"samples_seen":samples,"first_loss":losses[0],
            "last_loss":losses[-1],"mean_loss":statistics.fmean(losses)}


@torch.no_grad()
def evaluate_panel(rows,plan,*,model=None,wrong=False):
    if type(wrong) is not bool or (wrong and model is None):
        raise ValueError("Invalid percussion evaluation control")
    if model is not None and not isinstance(model,PercussionModel):
        raise ValueError("Expected percussion model")
    before=model_state_sha256(model) if model is not None else None
    mode=model.training if model is not None else None
    if model is not None:
        model.eval()
    groups={name:{key:[] for key in METRICS} for name in DRUMS}
    counts=dict.fromkeys(DRUMS,0)
    try:
        for rows_batch in evaluation_batches(rows,plan.evaluation_batch_size):
            for replica in range(plan.excitation_replicates):
                batch=collate(rows_batch,domain="evaluation",replica=replica)
                offsets=range(1,len(DRUMS)) if wrong else (0,)
                values=[]
                for offset in offsets:
                    labels=(batch.labels+offset)%len(DRUMS)
                    renderer=broadband_control if model is None else model
                    predicted=renderer(labels,batch.velocities,batch.frames,batch.excitation)
                    values.append(diagnostics(predicted,batch.targets))
                for i,label in enumerate(batch.labels.tolist()):
                    drum=DRUMS[label]
                    counts[drum]+=1
                    for key in METRICS:
                        groups[drum][key].append(statistics.fmean(float(v[key][i]) for v in values))
    finally:
        if model is not None:
            model.train(mode)
    if before is not None and before!=model_state_sha256(model):
        raise ValueError("Percussion evaluation mutated model state")
    if sum(counts.values())!=len(rows)*plan.excitation_replicates or any(n==0 for n in counts.values()):
        raise ValueError("Incomplete percussion evaluation")
    result={name:{key:statistics.fmean(items) for key,items in metrics.items()}
            for name,metrics in groups.items()}
    return {"notes":len(rows),"excitation_replicates":plan.excitation_replicates,
            "conditions_per_note":len(DRUMS)-1 if wrong else 1,
            "by_drum":result,"macro":{key:statistics.fmean(row[key] for row in result.values()) for key in METRICS}}


def evaluate(corpus,plan,*,model=None,wrong=False):
    return {name:evaluate_panel(corpus.get(name),plan,model=model,wrong=wrong) for name in PANELS[1:]}


def paired(arms):
    if set(arms)!=set(ARMS):
        raise ValueError("Incomplete percussion comparison")
    return {panel:{metric:{name:arms[control][panel]["macro"][metric]-arms["trained"][panel]["macro"][metric]
                           for name,control in (("initial_minus_trained","initial"),
                                                ("broadband_minus_trained","broadband"),
                                                ("wrong_minus_matched","wrong"))}
                   for metric in (*FEATURE_NAMES,"objective")} for panel in PANELS[1:]}


def run_study(plan=None):
    plan=StudyPlan() if plan is None else plan
    if not isinstance(plan,StudyPlan):
        raise ValueError("Expected percussion study plan")
    corpus=build_corpus(plan.grid)
    batches,trace=schedule(corpus.get("train"),updates=plan.updates,batch_size=plan.batch_size,seed=plan.sampler_seed)
    broadband=evaluate(corpus,plan)
    results=[]
    for seed in plan.seeds:
        model=initialize(plan.width,seed)
        initial=evaluate(corpus,plan,model=model)
        training=train(model,corpus,plan,batches)
        trained=evaluate(corpus,plan,model=model)
        wrong=evaluate(corpus,plan,model=model,wrong=True)
        arms={"broadband":broadband,"initial":initial,"trained":trained,"wrong":wrong}
        results.append({"seed":seed,"training":training,"evaluation":arms,"paired":paired(arms)})
    summary={}
    for panel in PANELS[1:]:
        summary[panel]={}
        for metric in (*FEATURE_NAMES,"objective"):
            summary[panel][metric]={}
            for key in ("initial_minus_trained","broadband_minus_trained","wrong_minus_matched"):
                values=[r["paired"][panel][metric][key] for r in results]
                summary[panel][metric][key]={"mean":statistics.fmean(values),"min":min(values),
                                           "max":max(values),"positive_seeds":sum(v>0 for v in values)}
    return {"protocol":PROTOCOL,"plan":asdict(plan),"source":corpus.report(),
            "schedule_sha256":trace,"results":results,"summary":summary,
            "source_phase_matching":False,"independent_excitation":True,
            "evaluation_used_for_training":False,"external_recordings_used":False,
            "perceptual_quality_established":False,"model_weights_published":False,
            "checkpoint_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=8)
    args=parser.parse_args(argv)
    try:
        result=run_study(StudyPlan(updates=args.updates,batch_size=args.batch_size))
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    except (ValueError,RuntimeError) as exc:
        parser.exit(2,f"Percussion feature study failed: {exc}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
