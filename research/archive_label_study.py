"""Multi-seed controlled comparison of true versus random category supervision.

This is a local, CPU-only diagnostic using the same private reviewed corpus.
No checkpoint, waveform, audio content, or private recording identifier is
written. It is not a text-to-audio benchmark or a model release criterion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import torch

from research.archive_conditioning_assay import (
    AssayConfig, evaluate_conditioning, compare_initialization,
    PROTOCOL as ASSAY_PROTOCOL, LEVELS,
)
from research.archive_readiness import audit_archive
from research.archive_sampler import CounterGroupSampler
from research.archive_study_plan import (
    StudyPlan, ARMS, PROTOCOL, corruption_batch, update_trace, parse_seeds,
)
from research.archive_training_contracts import make_archive_optimizer, runtime_identity
from research.archive_validation import model_fingerprint
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, require_sha256
from research.wave_diffusion import CATEGORIES, TinyWaveDenoiser, DiffusionSchedule

METRICS=("matched_mse","mean_wrong_minus_matched","hardest_wrong_minus_matched",
         "null_minus_matched","mean_rank","strict_win_fraction")


def _finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ArchiveError("Study metric must be a finite scalar")
    return float(value)


def describe(values):
    if len(values)<2:
        raise ArchiveError("Study summary needs independent seeds")
    numbers=[_finite(x) for x in values]
    return {"seeds":len(numbers),"mean":statistics.fmean(numbers),
            "median":statistics.median(numbers),"sample_stddev":statistics.stdev(numbers),
            "min":min(numbers),"max":max(numbers),"positive":sum(x>0 for x in numbers),
            "zero":sum(x==0 for x in numbers),"negative":sum(x<0 for x in numbers)}


def _panel_pin(report):
    if report.get("protocol")!=ASSAY_PROTOCOL:
        raise ArchiveError("Unexpected study assay protocol")
    keys=("archive_sha256","eligible_validation_sha256","seed","repeats","batch_size",
          "diffusion_profile","diffusion_steps","timesteps","eligible_validation_clips",
          "prediction_rows","noise_trace_sha256","condition_columns","truth_rows",
          "tie_absolute_tolerance")
    return tuple(report[key] for key in keys)


def _sections(report):
    yield "clip_weighted",report["clip_weighted"]
    yield "group_macro",report["group_macro"]
    for level in LEVELS:
        for kind in ("clip_weighted","group_macro"):
            yield f"{level}/{kind}",report["by_noise_level"][level][kind]
    for category in CATEGORIES:
        yield f"category/{category}",report["by_category"][category]


def _paired_deltas(initial,matched,control):
    if _panel_pin(initial)!=_panel_pin(matched) or _panel_pin(initial)!=_panel_pin(control):
        raise ArchiveError("Study comparison corruption or coverage changed")
    result={}
    for (name,a),(name_b,b),(name_c,c) in zip(_sections(initial),_sections(matched),_sections(control),strict=True):
        if name!=name_b or name!=name_c:
            raise ArchiveError("Study comparison section changed")
        for count in ("cases","recording_groups"):
            if a.get(count)!=b.get(count) or a.get(count)!=c.get(count):
                raise ArchiveError("Study comparison coverage changed")
        result[name]={}
        for metric in METRICS:
            values=[_finite(row[metric]) for row in (a,b,c)]
            result[name][metric]={
                "matched_minus_initial":values[1]-values[0],
                "random_label_minus_initial":values[2]-values[0],
                "matched_minus_random_label":values[1]-values[2],
            }
    return result


def summarize_runs(runs,plan:StudyPlan):
    if not isinstance(plan,StudyPlan) or len(runs)!=len(plan.seeds):
        raise ArchiveError("Study run count differs from plan")
    expected_panel=None
    sections={}
    for seed,run in zip(plan.seeds,runs,strict=True):
        if type(run["seed"]) is not int or run["seed"]!=seed:
            raise ArchiveError("Study run seed order changed")
        reports=[run["initialization"],run["arms"]["matched"]["assay"],
                 run["arms"]["random_label"]["assay"]]
        current=_panel_pin(reports[0])
        if expected_panel is None:
            expected_panel=current
        if expected_panel!=current:
            raise ArchiveError("Study seeds use different validation panels")
        for report in reports:
            if (report["seed"]!=plan.assay_seed or report["repeats"]!=plan.assay_repeats
                    or report["batch_size"]!=plan.assay_batch_size
                    or report["diffusion_profile"]!=plan.diffusion_profile
                    or report["diffusion_steps"]!=plan.diffusion_steps):
                raise ArchiveError("Study assay settings differ from plan")
        for arm in ARMS:
            row=run["arms"][arm]
            if row["initial_model_sha256"]!=reports[0]["model_sha256"]:
                raise ArchiveError("Study arms do not share initialization")
            if row["trained_model_sha256"]!=row["assay"]["model_sha256"]:
                raise ArchiveError("Study assay model does not match trained state")
            if row["completed_steps"]!=plan.steps or row["training_examples"]!=plan.steps*plan.batch_size:
                raise ArchiveError("Study optimizer workload changed")
        deltas=_paired_deltas(*reports)
        for section,metrics in deltas.items():
            for metric,comparisons in metrics.items():
                for comparison,value in comparisons.items():
                    sections.setdefault(section,{}).setdefault(metric,{}).setdefault(comparison,[]).append(value)
    return {
        section:{metric:{name:describe(values) for name,values in comparisons.items()}
                 for metric,comparisons in metrics.items()}
        for section,metrics in sections.items()
    }


def _single_run(reader,ready,plan,seed):
    sampler=CounterGroupSampler(ready,seed=seed)
    eligible={item.row.index:item for item in ready.eligible["train"]}
    assay_config=AssayConfig(plan.assay_seed,plan.assay_repeats,plan.assay_batch_size)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            initial=TinyWaveDenoiser(plan.channels)
            schedule=DiffusionSchedule(plan.diffusion_steps,profile=plan.diffusion_profile)
        initial_hash=model_fingerprint(initial)
        initialization=evaluate_conditioning(reader,ready,initial,schedule,config=assay_config)
        models={name:copy.deepcopy(initial) for name in ARMS}
        del initial
        optimizers={name:make_archive_optimizer(model) for name,model in models.items()}
        totals=dict.fromkeys(ARMS,0.)
        first={}
        last={}
        null_count=matches=non_null=0
        true_counts=np.zeros(len(CATEGORIES),dtype=np.int64)
        random_counts=np.zeros_like(true_counts)
        trace=hashlib.sha256(b"sonumio-paired-archive-label-study-trace-v1\0")
        for step in range(plan.steps):
            indices=sampler.draw(plan.batch_size).tolist()
            batch=reader.read_train_batch(indices)
            for index,row,clip in zip(indices,batch.rows,batch.clips,strict=True):
                item=eligible.get(index)
                if item is None or item.row!=row or item.pcm_sha256!=_cropped_clip_digest(clip[0]):
                    raise ArchiveError("Study training input differs from audited row")
            labels=torch.from_numpy(batch.labels)
            corruption=corruption_batch(plan,seed,step,labels)
            update_trace(trace,indices,corruption)
            keep=~corruption.dropped
            null_count+=int(corruption.dropped.sum())
            non_null+=int(keep.sum())
            matches+=int((corruption.random_label[keep]==labels[keep]).sum())
            true_counts+=np.bincount(labels[keep].numpy(),minlength=len(CATEGORIES))
            random_counts+=np.bincount(corruption.random_label[keep].numpy(),minlength=len(CATEGORIES))
            noisy=schedule.corrupt(torch.from_numpy(batch.clips),corruption.steps,corruption.noise)
            for name,condition in (("matched",corruption.matched),("random_label",corruption.random_label)):
                model=models[name]
                optimizer=optimizers[name]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction=model(noisy,corruption.steps,condition)
                loss=(prediction-corruption.noise).square().mean()
                if not torch.isfinite(loss):
                    raise ArchiveError("Nonfinite paired study loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                value=float(loss.detach())
                first.setdefault(name,value)
                last[name]=value
                totals[name]+=value
        arms={}
        for name,model in models.items():
            fingerprint=model_fingerprint(model)
            if fingerprint==initial_hash:
                raise ArchiveError("Study optimizer did not change model state")
            assay=evaluate_conditioning(reader,ready,model,schedule,config=assay_config)
            arms[name]={
                "initial_model_sha256":initial_hash,"trained_model_sha256":fingerprint,
                "completed_steps":plan.steps,"training_examples":plan.steps*plan.batch_size,
                "first_loss":first[name],"last_loss":last[name],"mean_loss":totals[name]/plan.steps,
                "assay":assay,
            }
        _paired_deltas(initialization,arms["matched"]["assay"],arms["random_label"]["assay"])
        return {
            "seed":seed,"initialization":initialization,"arms":arms,
            "training_trace_sha256":trace.hexdigest(),"sampler":sampler.describe(),
            "shared_null_examples":null_count,"non_null_examples":non_null,
            "random_label_accidental_matches":matches,
            "random_label_accidental_match_fraction":matches/non_null if non_null else None,
            "true_non_null_label_counts":dict(zip(CATEGORIES,true_counts.tolist(),strict=True)),
            "random_non_null_label_counts":dict(zip(CATEGORIES,random_counts.tolist(),strict=True)),
            "paired_training_corruption":True,
        }


def run_study(archive,*,expected_sha256,plan:StudyPlan|None=None):
    require_sha256(expected_sha256)
    plan=StudyPlan() if plan is None else plan
    if not isinstance(plan,StudyPlan):
        raise ArchiveError("Expected StudyPlan")
    runtime=runtime_identity()
    started=perf_counter()
    with CorpusArchiveReader(archive,expected_sha256=expected_sha256) as reader:
        ready=audit_archive(reader)
        if ready.report["readiness_gate_passed"] is not True:
            raise ArchiveError("Archive failed paired study readiness")
        validation_work=plan.assay_predictions(len(ready.eligible["validation"]))
        fingerprint=plan.fingerprint(ready.archive_sha256,ready.eligible_fingerprints["train"],
                                     ready.eligible_fingerprints["validation"])
        runs=[_single_run(reader,ready,plan,seed) for seed in plan.seeds]
        summary=summarize_runs(runs,plan)
        reader.report()
    return {
        "protocol":PROTOCOL,"study_plan":plan.as_dict(),"study_plan_sha256":fingerprint,
        "archive_sha256":ready.archive_sha256,"eligible_split_fingerprints":ready.eligible_fingerprints,
        "runtime":runtime,"arms":list(ARMS),"runs":runs,"paired_seed_summaries":summary,
        "training_examples_across_arms":plan.train_examples,"validation_prediction_rows":validation_work,
        "elapsed_seconds":round(perf_counter()-started,3),
        "summary_unit":"independent_model_initialization_seed",
        "uncertainty":"descriptive_seed_spread_not_confidence_interval",
        "random_label_control":"uniform_independent_with_replacement_shared_null_dropout",
        "statistical_significance_claimed":False,"test_split_scored":False,
        "generated_audio_semantics_established":False,"model_publication_authorized":False,
        "weights_or_audio_files_written":False,"original_sonumio_1":False,
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--expected-sha256",required=True)
    defaults=StudyPlan()
    parser.add_argument("--seeds",default=",".join(map(str,defaults.seeds)))
    for field in ("steps","batch_size","channels","diffusion_steps","assay_seed","assay_repeats","assay_batch_size"):
        parser.add_argument("--"+field.replace("_","-"),type=int,default=getattr(defaults,field))
    parser.add_argument("--diffusion-profile",choices=DiffusionSchedule.PROFILES,default=defaults.diffusion_profile)
    parser.add_argument("--drop-condition",type=float,default=defaults.drop_condition)
    args=parser.parse_args(argv)
    try:
        settings={key:getattr(args,key) for key in StudyPlan.__dataclass_fields__ if key!="seeds"}
        plan=StudyPlan(seeds=parse_seeds(args.seeds),**settings)
        result=run_study(args.archive,expected_sha256=args.expected_sha256,plan=plan)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Paired archive study: {exc}\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
