"""Joint original-music generation through separately trained structure and melody.

The pipeline learns bar chords, lead onset patterns and note pitches on
synthetic arrangements. The section plan, durations, accompaniment
rules and timbres remain procedural. This is not a joint end-to-end model.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
import math
import statistics
import argparse

import numpy as np
import torch

from research.music_structure_data import (
    schedule,extract as extract_structure,reconstruct,
    fingerprint as structure_fingerprint,
)
from research.music_structure_model import StructureGRU,collate as structure_batch,joint_loss,sample_structure
from research.music_structure_baselines import StructureCounts,evaluate_counts
from research.music_structure_study import likelihood as structure_likelihood,model_probabilities
from research.music_melody_data import (
    extract as extract_melody,fingerprint as melody_fingerprint,replace_melody,
    collate as melody_batch,
)
from research.music_melody_model import MelodyGRU,token_loss,generate_tokens
from research.music_melody_baselines import MelodyCounts,evaluate_counts as melody_counts
from research.music_melody_study import likelihood_metrics,neural_probabilities
from research.weight_ema import model_state_sha256
from sonumio.arrangement import ArrangementPlan,compose_arrangement,arrangement_digest
from sonumio.music_mix import generate_arrangement_audio
from sonumio.music_analysis import analyze_score
from sonumio.music import STYLES,MODES

PROTOCOL="sonumio-original-combined-structure-melody-v1"
ARMS=("procedural","learned_structure","learned_melody","learned_both")


def _seeds(values,low,high):
    if (type(values) is not tuple or not low<=len(values)<=high
            or any(type(v) is not int or not 0<=v<2**32 for v in values)
            or len(set(values))!=len(values)):
        raise ValueError("Invalid combined music seeds")


@dataclass(frozen=True,slots=True)
class CombinedPlan:
    model_seeds:tuple[int,...]=(17,29,41)
    training_seeds:tuple[int,...]=tuple(range(12))
    development_seeds:tuple[int,...]=(201,202,203)
    sampling_seeds:tuple[int,...]=(701,709)
    bars:int=8
    width:int=48
    updates:int=128
    batch_size:int=8

    def __post_init__(self):
        _seeds(self.model_seeds,2,4)
        _seeds(self.training_seeds,1,24)
        _seeds(self.development_seeds,1,8)
        _seeds(self.sampling_seeds,1,4)
        if set(self.training_seeds)&set(self.development_seeds):
            raise ValueError("Combined music source seed overlap")
        if (type(self.bars) is not int or not 4<=self.bars<=16
                or type(self.width) is not int or self.width not in (32,48,64)
                or type(self.updates) is not int or not 1<=self.updates<=256
                or type(self.batch_size) is not int or not 1<=self.batch_size<=32
                or self.training_examples>16384):
            raise ValueError("Invalid combined music training budget")

    @property
    def training_examples(self):
        return 2*len(self.model_seeds)*self.updates*self.batch_size


def source_panel(seeds,bars):
    rows=[]
    trace=hashlib.sha256(PROTOCOL.encode()+b"\0")
    for seed in seeds:
        for style in STYLES:
            for mode in MODES:
                plan=ArrangementPlan(style=style,mode=mode,bars=bars,bpm=120,seed=seed)
                score=compose_arrangement(plan)
                structure=extract_structure(score)
                melody=extract_melody(score)
                rows.append((plan,structure,melody))
                trace.update(bytes.fromhex(arrangement_digest(score)))
    return tuple(rows),trace.hexdigest()


def training_panel(plan):
    if not isinstance(plan,CombinedPlan):
        raise ValueError("Expected CombinedPlan")
    train,th=source_panel(plan.training_seeds,plan.bars)
    dev,dh=source_panel(plan.development_seeds,plan.bars)
    # Keep paired arrangement rows, but report structural and melodic
    # overlap independently. Only the melody-novel subset is used for
    # melody likelihood in the combined study.
    structures={structure_fingerprint(s,tokens_only=True) for _,s,_ in train}
    melodies={melody_fingerprint(m,pitches_only=True) for _,_,m in train}
    novel_structure=tuple(row for row in dev if structure_fingerprint(row[1],tokens_only=True) not in structures)
    novel_melody=tuple(row for row in dev if melody_fingerprint(row[2],pitches_only=True) not in melodies)
    if not novel_melody:
        raise ValueError("No novel combined-development melodies")
    return train,dev,novel_structure,novel_melody,{
        "training_sha256":th,"development_sha256":dh,"training_rows":len(train),
        "development_rows":len(dev),"novel_structure_rows":len(novel_structure),
        "novel_melody_rows":len(novel_melody),
        "development_structure_exact_overlap_rows":len(dev)-len(novel_structure),
        "development_melody_exact_overlap_rows":len(dev)-len(novel_melody)}


def _batch_indices(plan,seed,count):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    out=[]
    pending=[]
    for _ in range(plan.updates):
        while len(pending)<plan.batch_size:
            pending.extend(torch.randperm(count,generator=generator).tolist())
        out.append(pending[:plan.batch_size])
        del pending[:plan.batch_size]
    return out,hashlib.sha256(np.asarray(out,dtype="<i8").tobytes()).hexdigest()


def _evaluate(structure,melody,dev,novel_structure,novel_melody):
    all_structures=[s for _,s,_ in dev]
    unique_structures=[s for _,s,_ in novel_structure]
    unique_melodies=[m for _,_,m in novel_melody]
    return {
        "structure_all":structure_likelihood(all_structures,model_probabilities(structure,all_structures)),
        "structure_novel":structure_likelihood(unique_structures,model_probabilities(structure,unique_structures)),
        "melody_novel":likelihood_metrics(unique_melodies,neural_probabilities(melody,unique_melodies)),
    }


def build_variant(plan,structure,melody,seed,arm):
    """No held-out reference notes or harmony are read by either sampler."""
    if not isinstance(plan,ArrangementPlan) or arm not in ARMS:
        raise ValueError("Invalid combined generation request")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid combined sampling seed")
    if arm in ("learned_structure","learned_both"):
        tokens=sample_structure(structure,schedule(plan),seed,constrained=True)
        score=reconstruct(plan,tokens)
    else:
        score=compose_arrangement(plan)
    if arm in ("learned_melody","learned_both"):
        # The existing schedule comes from the generated structure, not a
        # held-out target. generate_tokens reads contexts only.
        seq=extract_melody(score)
        pitches=generate_tokens(melody,seq,(seed+0x9e3779b9)%(2**32),constrained=True)
        score=replace_melody(score,pitches)
    return score


def _identity(score):
    s=extract_structure(score)
    m=extract_melody(score)
    return (structure_fingerprint(s,tokens_only=True),melody_fingerprint(m,pitches_only=True))


@torch.no_grad()
def generate_panel(structure,melody,train,dev,plan):
    before=(model_state_sha256(structure),model_state_sha256(melody))
    known_structure={structure_fingerprint(s,tokens_only=True) for _,s,_ in train}
    known_melody={melody_fingerprint(m,pitches_only=True) for _,_,m in train}
    chosen={}
    for p,_,_ in dev:
        chosen.setdefault((p.style,p.mode),p)
    result={}
    for arm in ARMS:
        rows=[]
        hashes=set()
        identities=set()
        audio=None
        seeds=plan.sampling_seeds if arm!="procedural" else plan.sampling_seeds[:1]
        for (style,mode),p in chosen.items():
            for seed in seeds:
                score=build_variant(p,structure,melody,seed,arm)
                sh,mh=_identity(score)
                summary=analyze_score(score)
                if not summary["final_lead_tonic"] or summary["strong_beat_notes"]!=summary["strong_beat_chord_hits"]:
                    raise ValueError("Combined generated score violated musical constraints")
                hashes.add(arrangement_digest(score))
                identities.add((sh,mh))
                rows.append({"style":style,"mode":mode,"sample_seed":seed,
                             "structure_exact_training_match":sh in known_structure,
                             "melody_exact_training_match":mh in known_melody,
                             "events":summary["events"],
                             "mean_absolute_lead_interval":summary["mean_absolute_lead_interval"],
                             "distinct_interval_phrases":summary["distinct_interval_phrases"],
                             "final_tonic":summary["final_lead_tonic"],
                             "strong_chord_hits":summary["strong_beat_chord_hits"],
                             "strong_notes":summary["strong_beat_notes"]})
                if audio is None:
                    wav,report=generate_arrangement_audio(score)
                    audio={"frames":report["frames"],"channels":report["channels"],
                           "sample_rate":report["sample_rate"],"wav_bytes":len(wav),
                           "metrics":report["metrics"]}
        result[arm]={"rows":rows,"sample_count":len(rows),
                     "distinct_score_digests":len(hashes),
                     "distinct_structure_melody_pairs":len(identities),
                     "exact_training_structure_matches":sum(r["structure_exact_training_match"] for r in rows),
                     "exact_training_melody_matches":sum(r["melody_exact_training_match"] for r in rows),
                     "mean_note_interval":statistics.fmean(r["mean_absolute_lead_interval"] for r in rows
                                                           if r["mean_absolute_lead_interval"] is not None),
                     "first_audio_render":audio}
    if before!=(model_state_sha256(structure),model_state_sha256(melody)):
        raise ValueError("Combined music evaluation modified model weights")
    return result


def train_seed(plan,seed,train,dev,novel_structure,novel_melody):
    batches,trace=_batch_indices(plan,seed,len(train))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            structure=StructureGRU(plan.width)
            melody=MelodyGRU(plan.width)
        initial=(model_state_sha256(structure),model_state_sha256(melody))
        first=_evaluate(structure,melody,dev,novel_structure,novel_melody)
        so=torch.optim.AdamW(structure.parameters(),lr=.002,foreach=False,fused=False)
        mo=torch.optim.AdamW(melody.parameters(),lr=.002,foreach=False,fused=False)
        losses=[]
        for ids in batches:
            structure.train()
            melody.train()
            pc,pr,ctx,tc,tr=structure_batch([train[i][1] for i in ids])
            so.zero_grad(set_to_none=True)
            c,r,_=structure(pc,pr,ctx,tc.clamp_min(0))
            loss,_,_=joint_loss(c,r,tc,tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(structure.parameters(),1.,error_if_nonfinite=True)
            so.step()
            prev,context,target=melody_batch([train[i][2] for i in ids])
            mo.zero_grad(set_to_none=True)
            logits,_=melody(prev,context)
            ml=token_loss(logits,target)
            ml.backward()
            torch.nn.utils.clip_grad_norm_(melody.parameters(),1.,error_if_nonfinite=True)
            mo.step()
            losses.append((float(loss.detach()),float(ml.detach())))
        trained=(model_state_sha256(structure),model_state_sha256(melody))
        if any(a==b for a,b in zip(initial,trained,strict=True)):
            raise ValueError("Combined music training failed to update a model")
        return {"seed":seed,"initial_model_sha256":list(initial),"trained_model_sha256":list(trained),
                "batch_trace_sha256":trace,"structure_parameters":sum(p.numel() for p in structure.parameters()),
                "melody_parameters":sum(p.numel() for p in melody.parameters()),
                "first_training_loss":losses[0],"last_training_loss":losses[-1],
                "initial":first,"trained":_evaluate(structure,melody,dev,novel_structure,novel_melody),
                "generation":generate_panel(structure,melody,train,dev,plan)}


def run_study(plan=None):
    plan=CombinedPlan() if plan is None else plan
    train,dev,ns,nm,receipt=training_panel(plan)
    structures=StructureCounts([s for _,s,_ in train])
    melodies=MelodyCounts([m for _,_,m in train])
    full=[s for _,s,_ in dev]
    new_s=[s for _,s,_ in ns]
    new_m=[m for _,_,m in nm]
    baseline={"structure_all":structure_likelihood(full,evaluate_counts(structures,full,"conditional")),
              "structure_novel":structure_likelihood(new_s,evaluate_counts(structures,new_s,"conditional")),
              "melody_novel":likelihood_metrics(new_m,melody_counts(melodies,new_m,"conditional_bigram"))}
    runs=[train_seed(plan,seed,train,dev,ns,nm) for seed in plan.model_seeds]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"data":receipt,"baseline":baseline,
            "runs":runs,"training_examples":plan.training_examples,
            "all_models_beat_structure_all_baseline":all(
                r["trained"]["structure_all"]["joint_nll"]<baseline["structure_all"]["joint_nll"] for r in runs),
            "all_models_beat_melody_novel_baseline":all(
                r["trained"]["melody_novel"]["mean_nll"]<baseline["melody_novel"]["mean_nll"] for r in runs),
            "models_trained_separately":True,"generation_conditions_on_sampled_structure":True,
            "section_timbre_and_backing_rules_procedural":True,"constraints_not_learned":True,
            "perceptual_quality_established":False,"independent_test_scored":False,
            "weights_or_audio_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=CombinedPlan()
    for name in ("model_seeds","training_seeds","development_seeds","sampling_seeds"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(default,name))))
    for name in ("bars","width","updates","batch_size"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(default,name))
    args=parser.parse_args(argv)
    def seeds(s):
        return tuple(int(x) for x in s.split(","))
    try:
        plan=CombinedPlan(seeds(args.model_seeds),seeds(args.training_seeds),seeds(args.development_seeds),
                          seeds(args.sampling_seeds),args.bars,args.width,args.updates,args.batch_size)
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Combined music study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
