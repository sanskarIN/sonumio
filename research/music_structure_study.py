"""Train and evaluate bar-level chord/rhythm generation on original scores.

Keep exact-overlap and novel-development results separate. A finite
procedural source may produce no novel token sequences, which is an
explicitly reported limitation rather than evidence of generalization.
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

from research.music_structure_data import (
    split_corpus,fingerprint,reconstruct,allowed_chords,allowed_rhythms,
    CHORD_COUNT,RHYTHM_COUNT,
)
from research.music_structure_model import StructureGRU,collate,joint_loss,sample_structure,WIDTHS
from research.music_structure_baselines import StructureCounts,evaluate_counts,BASELINES
from research.weight_ema import model_state_sha256
from sonumio.music_mix import generate_arrangement_audio
from sonumio.music_analysis import analyze_score

PROTOCOL="sonumio-structure-gru-study-v1"


def _seeds(values,minimum,maximum):
    if (type(values) is not tuple or not minimum<=len(values)<=maximum
            or any(type(v) is not int or not 0<=v<2**32 for v in values)
            or len(set(values))!=len(values)):
        raise ValueError("Invalid structure study seeds")


@dataclass(frozen=True,slots=True)
class StructurePlan:
    model_seeds:tuple[int,...]=(17,29,41)
    training_seeds:tuple[int,...]=tuple(range(12))
    development_seeds:tuple[int,...]=(101,102,103)
    bar_counts:tuple[int,...]=(6,8,12)
    updates:int=128
    batch_size:int=8
    width:int=48
    sample_seeds:tuple[int,...]=(701,709)

    def __post_init__(self):
        _seeds(self.model_seeds,2,4)
        _seeds(self.training_seeds,1,24)
        _seeds(self.development_seeds,1,8)
        _seeds(self.sample_seeds,1,4)
        if set(self.training_seeds)&set(self.development_seeds):
            raise ValueError("Structure source seed overlap")
        if (type(self.bar_counts) is not tuple or not 1<=len(self.bar_counts)<=4
                or len(set(self.bar_counts))!=len(self.bar_counts)
                or any(type(v) is not int or not 4<=v<=16 for v in self.bar_counts)
                or len(self.training_seeds)*len(self.bar_counts)*6>384):
            raise ValueError("Invalid structure corpus budget")
        if (type(self.updates) is not int or not 1<=self.updates<=512
                or type(self.batch_size) is not int or not 1<=self.batch_size<=32
                or type(self.width) is not int or self.width not in WIDTHS
                or self.examples>16384):
            raise ValueError("Invalid structure training budget")

    @property
    def examples(self):
        return len(self.model_seeds)*self.updates*self.batch_size


def batch_plan(plan,seed,n):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    pending=[]
    result=[]
    for _ in range(plan.updates):
        while len(pending)<plan.batch_size:
            pending.extend(torch.randperm(n,generator=generator).tolist())
        result.append(pending[:plan.batch_size])
        del pending[:plan.batch_size]
    return result,hashlib.sha256(np.asarray(result,dtype="<i8").tobytes()).hexdigest()


@torch.no_grad()
def model_probabilities(model,sequences):
    before=model_state_sha256(model)
    mode=model.training
    model.eval()
    rows=[]
    try:
        for seq in sequences:
            pc,pr,ctx,tc,tr=collate([seq])
            c,r,_=model(pc,pr,ctx,tc)
            rows.append((torch.softmax(c[0].double(),-1).numpy(),
                         torch.softmax(r[0].double(),-1).numpy()))
        if model_state_sha256(model)!=before:
            raise ValueError("Structure evaluation modified model")
    finally:
        model.train(mode)
    return tuple(rows)


def _probability(value,length,count):
    if (not isinstance(value,np.ndarray) or value.shape!=(length,count)
            or not np.isfinite(value).all() or (value<0).any()
            or not np.allclose(value.sum(-1),1.,atol=1e-7,rtol=0)):
        raise ValueError("Invalid structure probabilities")


def likelihood(sequences,probabilities):
    if len(sequences)!=len(probabilities):
        raise ValueError("Structure evaluation panel mismatch")
    if not sequences:
        return None
    chord_losses=[]
    rhythm_losses=[]
    chord_hits=[]
    rhythm_hits=[]
    cadence_mass=[]
    for seq,(c,r) in zip(sequences,probabilities,strict=True):
        n=len(seq.chords)
        _probability(c,n,CHORD_COUNT)
        _probability(r,n,RHYTHM_COUNT)
        chord=np.asarray(seq.chords)
        rhythm=np.asarray(seq.rhythms)
        chord_losses.extend((-np.log(np.maximum(c[np.arange(n),chord],1e-300))).tolist())
        rhythm_losses.extend((-np.log(np.maximum(r[np.arange(n),rhythm],1e-300))).tolist())
        chord_hits.extend((c.argmax(1)==chord).tolist())
        rhythm_hits.extend((r.argmax(1)==rhythm).tolist())
        cadence_mass.extend((float(c[-2,4]),float(c[-1,0])))
    cl=statistics.fmean(chord_losses)
    rl=statistics.fmean(rhythm_losses)
    return {"sequences":len(sequences),"bars":len(chord_losses),
            "chord_nll":cl,"rhythm_nll":rl,"joint_nll":cl+rl,
            "chord_top1":statistics.fmean(chord_hits),
            "rhythm_top1":statistics.fmean(rhythm_hits),
            "unmasked_cadence_probability":statistics.fmean(cadence_mass)}


def _paired_metrics(model,full,novel):
    return {"all_development":likelihood(full,model_probabilities(model,full)),
            "novel_token_development":likelihood(novel,model_probabilities(model,novel))}


def generated_metrics(sequence,seen):
    n=len(sequence.chords)
    valid=[c in allowed_chords(i,n) and r in allowed_rhythms(ctx)
           for i,(c,r,ctx) in enumerate(zip(sequence.chords,sequence.rhythms,sequence.contexts,strict=True))]
    return {"all_rules_valid":all(valid),"valid_bar_fraction":statistics.fmean(valid),
            "final_cadence":sequence.chords[-2:]==(4,0),
            "distinct_chords":len(set(sequence.chords)),
            "distinct_rhythms":len(set(sequence.rhythms)),
            "exact_training_token_match":fingerprint(sequence,tokens_only=True) in seen,
            "silent_lead_bars":sequence.rhythms.count(0)}


@torch.no_grad()
def generation_panel(model,training,development,plan):
    seen={fingerprint(s,tokens_only=True) for _,s in training}
    selected={}
    for p,s in development:
        selected.setdefault((p.style,p.mode),(p,s))
    before=model_state_sha256(model)
    records=[]
    rendered=None
    distinct=set()
    for key,(p,seq) in selected.items():
        for seed in plan.sample_seeds:
            raw=sample_structure(model,seq.contexts,seed,constrained=False)
            masked=sample_structure(model,seq.contexts,seed,constrained=True)
            m=generated_metrics(masked,seen)
            if not m["all_rules_valid"]:
                raise ValueError("Masked structure violated constraints")
            score=reconstruct(p,masked)
            structural=analyze_score(score)
            distinct.add(fingerprint(masked,tokens_only=True))
            records.append({"style":key[0],"mode":key[1],"seed":seed,
                            "unmasked":generated_metrics(raw,seen),"masked":m,
                            "renderable_score_events":structural["events"],
                            "final_procedural_lead_tonic":structural["final_lead_tonic"]})
            if rendered is None:
                wav,report=generate_arrangement_audio(score)
                rendered={"frames":report["frames"],"sample_rate":report["sample_rate"],
                          "channels":report["channels"],"bytes":len(wav),"metrics":report["metrics"]}
    if model_state_sha256(model)!=before:
        raise ValueError("Structure generation mutated model")
    return {"samples":len(records),"distinct_token_sequences":len(distinct),
            "masked_exact_training_matches":sum(r["masked"]["exact_training_token_match"] for r in records),
            "unmasked_mean_valid_fraction":statistics.fmean(r["unmasked"]["valid_bar_fraction"] for r in records),
            "unmasked_cadences":sum(r["unmasked"]["final_cadence"] for r in records),
            "rows":records,"render":rendered}


def train_seed(plan,seed,training,development,novel):
    train=[s for _,s in training]
    dev=[s for _,s in development]
    novel_sequences=[s for _,s in novel]
    order,trace=batch_plan(plan,seed,len(train))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            model=StructureGRU(plan.width)
        first=model_state_sha256(model)
        initial=_paired_metrics(model,dev,novel_sequences)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,foreach=False,fused=False)
        history=[]
        for ids in order:
            model.train()
            pc,pr,ctx,tc,tr=collate([train[i] for i in ids])
            optimizer.zero_grad(set_to_none=True)
            c,r,_=model(pc,pr,ctx,tc.clamp_min(0))
            total,cl,rl=joint_loss(c,r,tc,tr)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            history.append((float(cl.detach()),float(rl.detach())))
        last=model_state_sha256(model)
        if last==first:
            raise ValueError("Structure training did not update")
        return {"seed":seed,"initial_sha256":first,"trained_sha256":last,
                "batch_trace_sha256":trace,"parameters":sum(p.numel() for p in model.parameters()),
                "first_training_losses":history[0],"last_training_losses":history[-1],
                "initial":initial,"trained":_paired_metrics(model,dev,novel_sequences),
                "generation":generation_panel(model,training,development,plan)}


def run_study(plan=None):
    plan=StructurePlan() if plan is None else plan
    if not isinstance(plan,StructurePlan):
        raise ValueError("Expected StructurePlan")
    training,development,novel,receipt=split_corpus(plan.training_seeds,plan.development_seeds,plan.bar_counts)
    full=[s for _,s in development]
    unique=[s for _,s in novel]
    counts=StructureCounts([s for _,s in training])
    baselines={kind:{"all_development":likelihood(full,evaluate_counts(counts,full,kind)),
                     "novel_token_development":likelihood(unique,evaluate_counts(counts,unique,kind))}
               for kind in BASELINES}
    runs=[train_seed(plan,seed,training,development,novel) for seed in plan.model_seeds]
    reference=baselines["conditional"]["all_development"]["joint_nll"]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"data":receipt,"baselines":baselines,
            "runs":runs,"training_examples":plan.examples,
            "mean_trained_joint_nll":statistics.fmean(r["trained"]["all_development"]["joint_nll"] for r in runs),
            "all_seeds_beat_conditional_counts":all(r["trained"]["all_development"]["joint_nll"]<reference for r in runs),
            "novel_subset_available":bool(novel),"unmasked_likelihood":True,
            "rhythm_and_chords_learned_within_fixed_vocabulary":True,
            "lead_pitch_policy_procedural":True,"synthetic_only":True,
            "perceptual_quality_established":False,"independent_test_scored":False,
            "model_or_audio_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=StructurePlan()
    for field in ("model_seeds","training_seeds","development_seeds","bar_counts","sample_seeds"):
        parser.add_argument("--"+field.replace("_","-"),default=",".join(map(str,getattr(defaults,field))))
    for field in ("updates","batch_size","width"):
        parser.add_argument("--"+field.replace("_","-"),type=int,default=getattr(defaults,field))
    a=parser.parse_args(argv)
    def numbers(s):
        return tuple(int(x) for x in s.split(","))
    try:
        plan=StructurePlan(numbers(a.model_seeds),numbers(a.training_seeds),
                           numbers(a.development_seeds),numbers(a.bar_counts),
                           a.updates,a.batch_size,a.width,numbers(a.sample_seeds))
        result=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Structure study: {exc}\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
