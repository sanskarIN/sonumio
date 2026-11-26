"""Matched original symbolic melody training and free-running evaluation.

The held-out panel is synthetic development data, not independent music.
Evaluate unmasked teacher-forced likelihood against simple baselines;
separately sample without true pitch feedback and measure mask effects.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import statistics

import numpy as np
import torch

from research.music_melody_data import (
    split_corpus,collate,valid_tokens,replace_melody,fingerprint,
    MelodySequence,PITCH_COUNT,
)
from research.music_melody_model import MelodyGRU,token_loss,generate_tokens,WIDTHS
from research.music_melody_baselines import MelodyCounts,evaluate_counts
from research.weight_ema import model_state_sha256
from sonumio.music_analysis import analyze_score
from sonumio.music_mix import generate_arrangement_audio

PROTOCOL="sonumio-original-melody-gru-study-v1"
BASELINES=("uniform","unigram","conditional_bigram")
MAX_UPDATES=512
MAX_TRAIN_EXAMPLES=16_384


def _seeds(values,minimum,maximum):
    if (type(values) is not tuple or not minimum<=len(values)<=maximum
            or any(type(x) is not int or not 0<=x<2**32 for x in values)
            or len(set(values))!=len(values)):
        raise ValueError("Invalid melody study seed list")


@dataclass(frozen=True,slots=True)
class MelodyPlan:
    model_seeds:tuple[int,...]=(17,29,41)
    training_seeds:tuple[int,...]=tuple(range(12))
    development_seeds:tuple[int,...]=(101,102,103)
    updates:int=128
    batch_size:int=8
    width:int=48
    bars:int=8
    sample_seeds:tuple[int,...]=(701,709)

    def __post_init__(self):
        _seeds(self.model_seeds,2,4)
        _seeds(self.training_seeds,1,24)
        _seeds(self.development_seeds,1,8)
        _seeds(self.sample_seeds,1,4)
        if set(self.training_seeds)&set(self.development_seeds):
            raise ValueError("Melody study source seeds overlap")
        if (type(self.updates) is not int or not 1<=self.updates<=MAX_UPDATES
                or type(self.batch_size) is not int or not 1<=self.batch_size<=32
                or type(self.width) is not int or self.width not in WIDTHS
                or type(self.bars) is not int or not 4<=self.bars<=16
                or self.train_examples>MAX_TRAIN_EXAMPLES):
            raise ValueError("Invalid bounded melody training budget")

    @property
    def train_examples(self):
        return len(self.model_seeds)*self.updates*self.batch_size


def batch_plan(plan,seed,count):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    order=[]
    remaining=[]
    for _ in range(plan.updates):
        while len(remaining)<plan.batch_size:
            remaining.extend(torch.randperm(count,generator=generator).tolist())
        order.append(remaining[:plan.batch_size])
        del remaining[:plan.batch_size]
    digest=hashlib.sha256(np.asarray(order,dtype="<i8").tobytes()).hexdigest()
    return order,digest


@torch.no_grad()
def neural_probabilities(model,sequences):
    before=model_state_sha256(model)
    mode=model.training
    model.eval()
    rows=[]
    try:
        for seq in sequences:
            prev,ctx,_=collate([seq])
            logits,_=model(prev,ctx)
            rows.append(torch.softmax(logits[0].double(),dim=-1).numpy())
        if model_state_sha256(model)!=before:
            raise ValueError("Melody evaluation changed weights")
    finally:
        model.train(mode)
    return tuple(rows)


def likelihood_metrics(sequences,probabilities):
    if len(sequences)!=len(probabilities) or not len(sequences):
        raise ValueError("Incomplete melody likelihood panel")
    nll=[]
    sequence_means=[]
    accuracy=[]
    valid_mass=[]
    for seq,p in zip(sequences,probabilities,strict=True):
        if (not isinstance(p,np.ndarray) or p.shape!=(len(seq.pitches),PITCH_COUNT)
                or not np.isfinite(p).all() or (p<0).any()
                or not np.allclose(p.sum(1),1,rtol=0,atol=1e-7)):
            raise ValueError("Invalid melody probability panel")
        selected=p[np.arange(len(seq.pitches)),np.asarray(seq.pitches)]
        loss=-np.log(np.maximum(selected,1e-300))
        nll.extend(loss.tolist())
        sequence_means.append(float(loss.mean()))
        accuracy.extend((p.argmax(1)==np.asarray(seq.pitches)).tolist())
        for i,context in enumerate(seq.contexts):
            valid_mass.append(float(p[i,list(valid_tokens(context,final=i==len(seq.pitches)-1))].sum()))
    return {"tokens":len(nll),"sequences":len(sequences),"mean_nll":statistics.fmean(nll),
            "perplexity":math.exp(min(700.,statistics.fmean(nll))),
            "mean_sequence_nll":statistics.fmean(sequence_means),
            "top1_accuracy":statistics.fmean(accuracy),
            "unmasked_valid_probability_mass":statistics.fmean(valid_mass)}


def free_running_metrics(sequence,pitches):
    if len(pitches)!=len(sequence.pitches):
        raise ValueError("Invalid generated melody length")
    validity=[p in valid_tokens(c,final=i==len(pitches)-1)
              for i,(p,c) in enumerate(zip(pitches,sequence.contexts,strict=True))]
    intervals=[abs(b-a) for a,b in zip(pitches,pitches[1:])]
    return {"valid_token_fraction":statistics.fmean(validity),
            "exact_target_match":pitches==sequence.pitches,
            "distinct_pitch_tokens":len(set(pitches)),
            "mean_absolute_interval":statistics.fmean(intervals) if intervals else 0.,
            "maximum_absolute_interval":max(intervals,default=0),
            "final_tonic":pitches[-1]%12==0}


@torch.no_grad()
def generate_panel(model,development,training,plan):
    before=model_state_sha256(model)
    training_hashes={fingerprint(s,pitches_only=True) for _,s in training}
    seen=set()
    rows=[]
    # Prespecified one example per style/mode, same panel for all models.
    selected={}
    for score,sequence in development:
        selected.setdefault((score.plan.style,score.plan.mode),(score,sequence))
    rendered=None
    for (style,mode),(score,seq) in selected.items():
        for seed in plan.sample_seeds:
            constrained=generate_tokens(model,seq,seed,constrained=True)
            raw=generate_tokens(model,seq,seed,constrained=False)
            generated=MelodySequence(constrained,seq.contexts)
            digest=fingerprint(generated,pitches_only=True)
            seen.add(digest)
            metrics=free_running_metrics(seq,constrained)
            if metrics["valid_token_fraction"]!=1:
                raise ValueError("Constrained melody violated mask")
            changed=replace_melody(score,constrained)
            structural=analyze_score(changed)
            rows.append({"style":style,"mode":mode,"sample_seed":seed,
                         "constrained":metrics,"unconstrained":free_running_metrics(seq,raw),
                         "exact_training_melody_match":digest in training_hashes,
                         "score_lead_chord_hits":structural["strong_beat_chord_hits"],
                         "score_lead_strong_beats":structural["strong_beat_notes"]})
            if rendered is None:
                # Exercise actual original stereo synthesis without writing audio.
                wav,report=generate_arrangement_audio(changed)
                rendered={"wav_bytes":len(wav),"sample_rate":report["sample_rate"],
                          "frames":report["frames"],"channels":report["channels"],
                          "metrics":report["metrics"]}
    if model_state_sha256(model)!=before:
        raise ValueError("Sampling mutated melody weights")
    return {"samples":len(rows),"distinct_generated_pitch_sequences":len(seen),
            "exact_training_pitch_matches":sum(r["exact_training_melody_match"] for r in rows),
            "mean_unmasked_valid_token_fraction":statistics.fmean(r["unconstrained"]["valid_token_fraction"] for r in rows),
            "mean_masked_valid_token_fraction":statistics.fmean(r["constrained"]["valid_token_fraction"] for r in rows),
            "mean_masked_distinct_pitches":statistics.fmean(r["constrained"]["distinct_pitch_tokens"] for r in rows),
            "rows":rows,"first_generated_score_render":rendered}


def train_seed(plan,seed,training,development):
    train=[s for _,s in training]
    dev=[s for _,s in development]
    batches,trace=batch_plan(plan,seed,len(train))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            model=MelodyGRU(plan.width)
        initial_hash=model_state_sha256(model)
        initial=likelihood_metrics(dev,neural_probabilities(model,dev))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,foreach=False,fused=False)
        losses=[]
        model.train()
        for ids in batches:
            prev,ctx,targets=collate([train[i] for i in ids])
            optimizer.zero_grad(set_to_none=True)
            logits,_=model(prev,ctx)
            loss=token_loss(logits,targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        final_hash=model_state_sha256(model)
        if final_hash==initial_hash:
            raise ValueError("Melody model failed to update")
        final=likelihood_metrics(dev,neural_probabilities(model,dev))
        return {"seed":seed,"initial_model_sha256":initial_hash,"final_model_sha256":final_hash,
                "batch_trace_sha256":trace,"parameters":sum(p.numel() for p in model.parameters()),
                "first_training_loss":losses[0],"last_training_loss":losses[-1],
                "initial":initial,"trained":final,"generation":generate_panel(model,development,training,plan)}


def run_study(plan=None):
    plan=MelodyPlan() if plan is None else plan
    if not isinstance(plan,MelodyPlan):
        raise ValueError("Expected MelodyPlan")
    training,development,receipt=split_corpus(plan.training_seeds,plan.development_seeds,bars=plan.bars)
    train=[s for _,s in training]
    dev=[s for _,s in development]
    counts=MelodyCounts(train)
    baselines={kind:likelihood_metrics(dev,evaluate_counts(counts,dev,kind)) for kind in BASELINES}
    runs=[train_seed(plan,seed,training,development) for seed in plan.model_seeds]
    reference=baselines["conditional_bigram"]["mean_nll"]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"data":receipt,"baselines":baselines,
            "runs":runs,"train_examples":plan.train_examples,
            "mean_trained_nll":statistics.fmean(r["trained"]["mean_nll"] for r in runs),
            "all_seeds_beat_initial_nll":all(r["trained"]["mean_nll"]<r["initial"]["mean_nll"] for r in runs),
            "all_seeds_beat_conditional_bigram_nll":all(r["trained"]["mean_nll"]<reference for r in runs),
            "unmasked_likelihood_evaluation":True,"mask_constraints_are_not_learned":True,
            "backing_and_rhythm_supplied":True,"synthetic_only":True,
            "independent_test_scored":False,"perceptual_quality_established":False,
            "audio_or_weights_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=MelodyPlan()
    for name in ("model_seeds","training_seeds","development_seeds","sample_seeds"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(default,name))))
    for name in ("updates","batch_size","width","bars"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(default,name))
    args=parser.parse_args(argv)
    def seeds(x):
        return tuple(int(v) for v in x.split(","))
    try:
        plan=MelodyPlan(seeds(args.model_seeds),seeds(args.training_seeds),seeds(args.development_seeds),
                        args.updates,args.batch_size,args.width,args.bars,seeds(args.sample_seeds))
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Melody study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
