"""Paired original sound effect diagnostics, controls and seed diversity.

Compare common categories under the old and new procedural engines.
The comparison does not hold loudness, algorithm complexity or subjective
realism constant. It is a diagnostic study, not a listening benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass,replace
import hashlib
import json
import math
import statistics

import numpy as np

from sonumio.engines.procedural import generate
from sonumio.nature_synthesis import NatureVoice,render_voice,SAMPLE_RATE,KINDS
from sonumio.nature_analysis import describe,compare

PROTOCOL="sonumio-nature-controls-v1"
COMMON={"rain":"rain","wind":"wind","ocean":"ocean","thunder":"thunder","birds":"forest"}
MAX_RENDER_FRAMES=24_000_000


@dataclass(frozen=True,slots=True)
class NatureStudyPlan:
    seeds:tuple[int,...]=(17,29,41)
    seconds:float=4.
    kinds:tuple[str,...]=KINDS

    def __post_init__(self):
        if (type(self.seeds) is not tuple or not 2<=len(self.seeds)<=8
                or any(type(v) is not int or not 0<=v<2**32 for v in self.seeds)
                or len(set(self.seeds))!=len(self.seeds)):
            raise ValueError("Invalid nature study seeds")
        if type(self.seconds) not in (int,float) or not 2<=self.seconds<=10:
            raise ValueError("Invalid nature study duration")
        if (type(self.kinds) is not tuple or not 1<=len(self.kinds)<=len(KINDS)
                or any(type(v) is not str or v not in KINDS for v in self.kinds)
                or len(set(self.kinds))!=len(self.kinds)):
            raise ValueError("Invalid nature study categories")
        if self.render_frames>MAX_RENDER_FRAMES:
            raise ValueError("Nature study render budget exceeded")

    @property
    def frames(self):
        return round(self.seconds*SAMPLE_RATE)

    @property
    def render_frames(self):
        # Seven control renders, baseline and replay, plus optional old baseline.
        return self.frames*len(self.seeds)*sum(10 if k in COMMON else 9 for k in self.kinds)


def _digest(samples):
    return hashlib.sha256(np.asarray(samples,dtype="<f4").tobytes()).hexdigest()


def study_seed(kind,seed,plan):
    voice=NatureVoice(kind,.65,.5,.5,seed)
    base=render_voice(voice,plan.frames)
    replay=render_voice(voice,plan.frames)
    if not np.array_equal(base,replay):
        raise ValueError("Nature renderer failed exact seeded replay")
    controls={
        "zero_intensity":replace(voice,intensity=0),
        "low_intensity":replace(voice,intensity=.25),
        "high_intensity":replace(voice,intensity=.9),
        "low_texture":replace(voice,texture=.1),
        "high_texture":replace(voice,texture=.9),
        "low_motion":replace(voice,motion=.1),
        "high_motion":replace(voice,motion=.9),
    }
    samples={name:render_voice(spec,plan.frames) for name,spec in controls.items()}
    if np.count_nonzero(samples["zero_intensity"]):
        raise ValueError("Zero intensity did not produce exact silence")
    summary=describe(base)
    intensity=compare(samples["low_intensity"],samples["high_intensity"])
    texture=compare(samples["low_texture"],samples["high_texture"])
    motion=compare(samples["low_motion"],samples["high_motion"])
    old=None
    if kind in COMMON:
        result=generate(COMMON[kind],plan.seconds,seed)
        if result.sample_rate!=SAMPLE_RATE or result.samples.shape!=(plan.frames,):
            raise ValueError("Old baseline sample clock changed")
        old=compare(result.samples,base)
    for signal in (base,*samples.values()):
        if not np.isfinite(signal).all() or np.max(np.abs(signal))>.800001:
            raise ValueError("Nature voice exceeded bounded signal contract")
    return {"seed":seed,"sha256":_digest(base),"baseline":summary,
            "replay_exact":True,"silence_exact":True,
            "intensity":intensity,"texture":texture,"motion":motion,
            "old_to_new":old},base


def _average(values):
    finite=[x for x in values if x is not None]
    return statistics.fmean(finite) if finite else None


def study_kind(kind,plan):
    records=[]
    signals=[]
    for seed in plan.seeds:
        row,signal=study_seed(kind,seed,plan)
        records.append(row)
        signals.append(signal)
    pairwise=[]
    for i in range(len(signals)):
        for j in range(i+1,len(signals)):
            x=compare(signals[i],signals[j])
            pairwise.append({"seed_a":plan.seeds[i],"seed_b":plan.seeds[j],
                             "waveform_mae":x["waveform_mae"],
                             "band_distribution_l1":x["band_distribution_l1"],
                             "centered_waveform_correlation":x["centered_waveform_correlation"]})
    return {"kind":kind,"runs":records,"pairwise_seed_comparisons":pairwise,
            "distinct_baseline_sha256":len({r["sha256"] for r in records}),
            "mean_rms":_average([r["baseline"]["rms"] for r in records]),
            "mean_centroid_hz":_average([r["baseline"]["centroid_hz"] for r in records]),
            "intensity_rms_increase_seeds":sum(r["intensity"]["rms_delta"]>0 for r in records),
            "mean_texture_centroid_delta_hz":_average([r["texture"]["centroid_delta_hz"] for r in records]),
            "mean_motion_envelope_activity_delta":_average([r["motion"]["envelope_activity_delta"] for r in records]),
            "mean_old_to_new_centroid_delta_hz":_average([r["old_to_new"]["centroid_delta_hz"] for r in records if r["old_to_new"] is not None])}


def run_study(plan=None):
    plan=NatureStudyPlan() if plan is None else plan
    if not isinstance(plan,NatureStudyPlan):
        raise ValueError("Expected NatureStudyPlan")
    rows=[study_kind(kind,plan) for kind in plan.kinds]
    return {"protocol":PROTOCOL,"plan":asdict(plan),"sample_rate":SAMPLE_RATE,
            "render_frame_budget":plan.render_frames,"categories":rows,
            "all_seeded_replays_exact":all(r["replay_exact"] for row in rows for r in row["runs"]),
            "all_zero_intensity_silent":all(r["silence_exact"] for row in rows for r in row["runs"]),
            "all_categories_vary_by_seed":all(row["distinct_baseline_sha256"]==len(plan.seeds) for row in rows),
            "baseline_comparison_loudness_matched":False,
            "source_is_procedural":True,"perceptual_quality_established":False,
            "files_written":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    default=NatureStudyPlan()
    parser.add_argument("--seeds",default=",".join(map(str,default.seeds)))
    parser.add_argument("--seconds",type=float,default=default.seconds)
    parser.add_argument("--kinds",default=",".join(default.kinds))
    args=parser.parse_args(argv)
    try:
        plan=NatureStudyPlan(tuple(int(x) for x in args.seeds.split(",")),args.seconds,
                             tuple(args.kinds.split(",")))
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Nature study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
