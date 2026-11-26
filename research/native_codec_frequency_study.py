"""Train-only frequency basis and regularized fine-tuning experiment.

Fixed projections compare analytic and fitted frequency allocations.
The two learned arms start from the exact same fitted rank-16 model,
receive identical batches, and differ only in explicit regularization.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import statistics

import torch

from research.native_codec import MAX_BATCH
from research.native_codec_transform import NativeBlockCodec, projection_diagnostics
from research.native_codec_frequency_basis import (
    BasisPlan, fit_frequency_basis, initialized_codec, anchored_penalty,
)
from research.native_codec_convergence import score, objective
from research.native_codec_study import dataset, batch_plan, _seeds, _parse_seeds, lowband_control
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-native24k-frequency-basis-anchoring-v1"
ARMS=("unregularized","anchored")
FIXED=("analytic_dct","fitted_4high","fitted_8high")
WEIGHTS={"anchor_distance":.25,"orthogonality_error":.1,"analysis_synthesis_disagreement":.1}
MAX_UPDATES=256
MAX_EXAMPLES=8192


@dataclass(frozen=True,slots=True)
class FrequencyStudyPlan:
    order_seeds:tuple[int,...]=(17,29,41)
    training_seeds:tuple[int,...]=(0,1,2)
    development_seeds:tuple[int,...]=(301,302)
    updates:int=128
    batch_size:int=4
    checkpoints:tuple[int,...]=(32,64,128)

    def __post_init__(self):
        _seeds(self.order_seeds,2,4)
        _seeds(self.training_seeds,1,6)
        _seeds(self.development_seeds,1,4)
        if set(self.training_seeds)&set(self.development_seeds):
            raise ValueError("Frequency study source seeds overlap")
        if (type(self.updates) is not int or not 1<=self.updates<=MAX_UPDATES
                or type(self.batch_size) is not int or not 1<=self.batch_size<=MAX_BATCH
                or self.train_examples>MAX_EXAMPLES):
            raise ValueError("Frequency study exceeds training budget")
        if (type(self.checkpoints) is not tuple or not 1<=len(self.checkpoints)<=8
                or any(type(step) is not int or not 1<=step<=self.updates for step in self.checkpoints)
                or tuple(sorted(set(self.checkpoints)))!=self.checkpoints
                or self.checkpoints[-1]!=self.updates):
            raise ValueError("Invalid frequency study checkpoints")

    @property
    def train_examples(self):
        return len(self.order_seeds)*len(ARMS)*self.updates*self.batch_size


def _penalty_report(model,anchor):
    with torch.no_grad():
        return {key:float(value) for key,value in anchored_penalty(model,anchor).items()}


def train_pair(plan,seed,initial,anchor,training,development,labels):
    identity=model_state_sha256(initial)
    batches,trace=batch_plan(plan,seed,len(training))
    arms={}
    for arm in ARMS:
        model=copy.deepcopy(initial)
        if model_state_sha256(model)!=identity:
            raise ValueError("Frequency arms initialization mismatch")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=False)
        records=[]
        losses=[]
        for step,ids in enumerate(batches,1):
            source=training[ids]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            reconstruction=objective(source,model(source),"balanced_phase")
            terms=anchored_penalty(model,anchor)
            penalty=sum(WEIGHTS[key]*value for key,value in terms.items())
            loss=reconstruction+penalty if arm=="anchored" else reconstruction
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite frequency study loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(reconstruction.detach()))
            if step in plan.checkpoints:
                records.append({"step":step,"evaluation":score(development,labels,model),
                                "mean_reconstruction_loss":statistics.fmean(losses),
                                "last_preclip_gradient_norm":float(norm),
                                "regularizer_terms":_penalty_report(model,anchor),
                                "projection":projection_diagnostics(model)})
                losses=[]
        if records[-1]["evaluation"]["model_sha256"]==identity:
            raise ValueError("Frequency study model did not update")
        arms[arm]=records
    if model_state_sha256(initial)!=identity:
        raise ValueError("Shared initial model changed")
    return {"order_seed":seed,"initial_model_sha256":identity,
            "batch_trace_sha256":trace,"arms":arms}


def passes_gate(panel,silence):
    return (panel["mean"]["snr_db"]>0
            and panel["silence_wins"]*3>=panel["clips"]*2
            and panel["mean"]["complex_highband_error"]<silence["mean"]["complex_highband_error"])


def summarize(runs,plan,silence,initial):
    if len(runs)!=len(plan.order_seeds):
        raise ValueError("Incomplete frequency study panel")
    for run,seed in zip(runs,plan.order_seeds,strict=True):
        if (run["order_seed"]!=seed or tuple(run["arms"])!=ARMS
                or run["initial_model_sha256"]!=initial["model_sha256"]):
            raise ValueError("Frequency study identity mismatch")
        for records in run["arms"].values():
            if tuple(p["step"] for p in records)!=plan.checkpoints:
                raise ValueError("Frequency checkpoint mismatch")
            if any(p["evaluation"]["clips"]!=silence["clips"] for p in records):
                raise ValueError("Frequency evaluation count mismatch")
    result={}
    for arm in ARMS:
        final=[run["arms"][arm][-1] for run in runs]
        panels=[r["evaluation"] for r in final]
        result[arm]={
            "final_mean_snr_db":statistics.fmean(p["mean"]["snr_db"] for p in panels),
            "final_mean_highband_error":statistics.fmean(p["mean"]["complex_highband_error"] for p in panels),
            "mean_anchor_distance":statistics.fmean(r["regularizer_terms"]["anchor_distance"] for r in final),
            "order_seed_gate_passed":[passes_gate(p,silence) for p in panels],
            "all_order_seeds_gate_passed":all(passes_gate(p,silence) for p in panels),
            "all_order_seeds_improve_initial_snr":all(p["mean"]["snr_db"]>initial["mean"]["snr_db"] for p in panels),
        }
    deltas=[]
    for run in runs:
        a,b=(run["arms"][arm][-1] for arm in ARMS)
        deltas.append({"order_seed":run["order_seed"],
            "snr_delta_db":b["evaluation"]["mean"]["snr_db"]-a["evaluation"]["mean"]["snr_db"],
            "highband_error_delta":b["evaluation"]["mean"]["complex_highband_error"]-a["evaluation"]["mean"]["complex_highband_error"],
            "anchor_distance_delta":b["regularizer_terms"]["anchor_distance"]-a["regularizer_terms"]["anchor_distance"]})
    return {"arms":result,"anchored_minus_unregularized":deltas}


def run_study(plan=None):
    plan=FrequencyStudyPlan() if plan is None else plan
    if not isinstance(plan,FrequencyStudyPlan):
        raise ValueError("Expected FrequencyStudyPlan")
    training,_,a,train_hash=dataset(plan.training_seeds)
    development,labels,b,dev_hash=dataset(plan.development_seeds)
    if set(a)&set(b):
        raise ValueError("Exact PCM overlap in frequency study")
    # Fit on training only, before any development scoring.
    basis4,fit4=fit_frequency_basis(training,BasisPlan(4))
    basis8,fit8=fit_frequency_basis(training,BasisPlan(8))
    initial=initialized_codec(basis4)
    fixed={"analytic_dct":score(development,labels,NativeBlockCodec()),
           "fitted_4high":score(development,labels,initial),
           "fitted_8high":score(development,labels,initialized_codec(basis8))}
    silence=score(development,labels,control=torch.zeros_like)
    lowband=score(development,labels,control=lowband_control)
    runs=[train_pair(plan,seed,initial,basis4,training,development,labels) for seed in plan.order_seeds]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"train_examples":plan.train_examples,
            "training_sha256":train_hash,"development_sha256":dev_hash,
            "training_clips":len(training),"development_clips":len(development),
            "basis_fit":{"fitted_4high":fit4,"fitted_8high":fit8},
            "fixed":fixed,"fixed_gate_passed":{k:passes_gate(v,silence) for k,v in fixed.items()},
            "silence_control":silence,"lowband_control":lowband,"runs":runs,
            "summary":summarize(runs,plan,silence,fixed["fitted_4high"]),
            "regularizer_weights":WEIGHTS.copy(),"shared_initialization_and_batches":True,
            "order_seeds_are_not_independent_initializations":True,
            "development_used_for_basis_fit":False,"independent_test_scored":False,
            "perceptual_quality_established":False,"files_written":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=FrequencyStudyPlan()
    for name in ("order_seeds","training_seeds","development_seeds","checkpoints"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(default,name))))
    parser.add_argument("--updates",type=int,default=default.updates)
    parser.add_argument("--batch-size",type=int,default=default.batch_size)
    args=parser.parse_args(argv)
    try:
        plan=FrequencyStudyPlan(_parse_seeds(args.order_seeds),_parse_seeds(args.training_seeds),
                               _parse_seeds(args.development_seeds),args.updates,args.batch_size,
                               _parse_seeds(args.checkpoints))
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Native frequency study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
