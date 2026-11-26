"""Compare native block-transform initialization and convolutional baseline.

The two block arms have identical parameter counts and optimizer settings,
differing only in initialization. The convolutional arm is a reference
architecture, not a compute- or parameter-matched comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import statistics

import torch

from research.native_codec import Native24kCodec
from research.native_codec_transform import NativeBlockCodec, projection_diagnostics, DCT_BINS
from research.native_codec_convergence import ConvergencePlan, objective, score
from research.native_codec_study import dataset, batch_plan, _parse_seeds, lowband_control
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-native-transform-initialization-v1"
ARMS=("convolutional","random_orthogonal","frequency_stratified")


def _model(arm,seed):
    if arm not in ARMS:
        raise ValueError("Unknown native architecture arm")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            return Native24kCodec() if arm=="convolutional" else NativeBlockCodec(arm,seed)


def _diagnostics(model):
    return projection_diagnostics(model) if isinstance(model,NativeBlockCodec) else None


def train_seed(plan,seed,train,development,labels):
    batches,trace=batch_plan(plan,seed,len(train))
    arms={}
    for arm in ARMS:
        model=_model(arm,seed)
        before=model_state_sha256(model)
        initial=score(development,labels,model)
        diagnostics=_diagnostics(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=False)
        snapshots=[]
        losses=[]
        for step,ids in enumerate(batches,1):
            source=train[ids]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=objective(source,model(source),"balanced_phase")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite architecture objective")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step in plan.checkpoints:
                snapshots.append({"step":step,"evaluation":score(development,labels,model),
                                  "training_loss":statistics.fmean(losses),
                                  "preclip_gradient_norm":float(norm),
                                  "projection":_diagnostics(model)})
                losses=[]
        if snapshots[-1]["evaluation"]["model_sha256"]==before:
            raise ValueError("Architecture model did not update")
        arms[arm]={"parameters":sum(p.numel() for p in model.parameters()),
                   "initial_model_sha256":before,"initial":initial,
                   "initial_projection":diagnostics,"snapshots":snapshots}
    return {"seed":seed,"batch_trace_sha256":trace,"arms":arms}


def summarize(runs,plan,silence):
    if len(runs)!=len(plan.model_seeds):
        raise ValueError("Incomplete architecture seed panel")
    summary={}
    for arm in ARMS:
        panels=[]
        initial=[]
        for seed,run in zip(plan.model_seeds,runs,strict=True):
            if run["seed"]!=seed or tuple(run["arms"])!=ARMS:
                raise ValueError("Architecture seed panel mismatch")
            entry=run["arms"][arm]
            if tuple(p["step"] for p in entry["snapshots"])!=plan.checkpoints:
                raise ValueError("Architecture step panel mismatch")
            panels.append(entry["snapshots"][-1]["evaluation"])
            initial.append(entry["initial"])
        snr=[p["mean"]["snr_db"] for p in panels]
        high=[p["mean"]["complex_highband_error"] for p in panels]
        wins=[p["silence_wins"] for p in panels]
        base_high=silence["mean"]["complex_highband_error"]
        if any(p["clips"]!=silence["clips"] for p in panels):
            raise ValueError("Architecture evaluation count mismatch")
        summary[arm]={
            "initial_mean_snr_db":statistics.fmean(p["mean"]["snr_db"] for p in initial),
            "final_mean_snr_db":statistics.fmean(snr),"seed_snr_db":snr,
            "seed_silence_wins":wins,"seed_highband_error":high,
            "final_mean_highband_error":statistics.fmean(high),
            "gate_passed":all(v>0 for v in snr)
            and all(w*3>=p["clips"]*2 for w,p in zip(wins,panels,strict=True))
            and all(v<base_high for v in high),
        }
    return summary


def run_study(plan=None):
    plan=ConvergencePlan() if plan is None else plan
    if not isinstance(plan,ConvergencePlan):
        raise ValueError("Expected ConvergencePlan")
    train,_,a,train_hash=dataset(plan.training_seeds)
    development,labels,b,dev_hash=dataset(plan.validation_seeds)
    if set(a)&set(b):
        raise ValueError("Exact PCM overlap in architecture source")
    silence=score(development,labels,control=torch.zeros_like)
    lowband=score(development,labels,control=lowband_control)
    runs=[train_seed(plan,seed,train,development,labels) for seed in plan.model_seeds]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"dct_bins":list(DCT_BINS),
            "training_sha256":train_hash,"development_sha256":dev_hash,
            "training_clips":len(train),"development_clips":len(development),
            "train_examples":plan.train_examples,"silence_control":silence,
            "lowband_control":lowband,"runs":runs,"summary":summarize(runs,plan,silence),
            "block_arms_share_architecture":True,"all_arms_share_batches_and_objective":True,
            "all_arms_parameter_matched":False,"frequency_initialization_deterministic":True,
            "independent_test_scored":False,"perceptual_quality_established":False,
            "files_written":0,"model_publication_authorized":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=ConvergencePlan()
    for name in ("model_seeds","training_seeds","validation_seeds","checkpoints"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(default,name))))
    parser.add_argument("--updates",type=int,default=default.updates)
    parser.add_argument("--batch-size",type=int,default=default.batch_size)
    args=parser.parse_args(argv)
    try:
        plan=ConvergencePlan(_parse_seeds(args.model_seeds),_parse_seeds(args.training_seeds),
                             _parse_seeds(args.validation_seeds),args.updates,args.batch_size,
                             _parse_seeds(args.checkpoints))
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Native transform study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
