"""Paired generation-panel comparisons for independent training-seed studies.

No selection of favorable samples, significance estimates, or automatic
model promotion. Unscorable cases remain in denominators; unavailable
paired continuous metrics are represented explicitly rather than imputed.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import statistics

from research.archive_generated_assay import (
    GenerationConfig, PROTOCOL as GENERATION_PROTOCOL, reverse_indices,
)
from research.archive_study_plan import StudyPlan, ARMS
from research.corpus_archive_format import ArchiveError, require_sha256
from research.wave_diffusion import CATEGORIES

PROTOCOL="sonumio-paired-generated-signal-study-v1"
MAX_STUDY_GENERATION_FORWARDS=100_000
SIGNAL_METRICS=("rms","centered_rms","peak","absolute_dc","overrange_fraction",
                "near_clip_fraction","crest_factor")
REFERENCE_METRICS=("target_rank","nearest_wrong_minus_target","target_distance")


def preflight_generation(plan:StudyPlan,config:GenerationConfig):
    if not isinstance(plan,StudyPlan) or not isinstance(config,GenerationConfig):
        raise ArchiveError("Expected bounded study and generation plans")
    reverse_indices(plan.diffusion_steps,config.reverse_steps)
    total=len(plan.seeds)*3*config.forward_calls
    if total>MAX_STUDY_GENERATION_FORWARDS:
        raise ArchiveError("Paired generated-signal study exceeds forward-call budget")
    return total


def _number(value,*,optional=False):
    if optional and value is None:
        return None
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ArchiveError("Nonfinite generated study metric")
    return float(value)


def _identity(report):
    if report["protocol"]!=GENERATION_PROTOCOL:
        raise ArchiveError("Unexpected generated-signal protocol")
    return tuple(report[key] for key in (
        "reference_sha256","reference_validation","diffusion_profile","diffusion_steps",
        "reverse_indices","seeds","guidances","initial_noise_trace_sha256",
        "forward_calls","terminal_alpha_bar","sampling_protocol",
    ))


def _case_map(report,plan,config):
    if (report["seeds"]!=list(config.seeds)
            or report["guidances"]!=[float(x) for x in config.guidances]
            or report["diffusion_steps"]!=plan.diffusion_steps
            or report["diffusion_profile"]!=plan.diffusion_profile
            or report["reverse_indices"]!=list(reverse_indices(plan.diffusion_steps,config.reverse_steps))
            or report["forward_calls"]!=config.forward_calls
            or report["model_state_unchanged"] is not True
            or report["test_split_scored"] is not False
            or report["files_written"]!=0):
        raise ArchiveError("Generated report differs from declared plan")
    for name in ("model_sha256","reference_sha256","initial_noise_trace_sha256"):
        require_sha256(report[name])
    expected=[(seed,category,float(g)) for seed in config.seeds for category in CATEGORIES for g in config.guidances]
    if len(report["cases"])!=len(expected):
        raise ArchiveError("Generated report case count changed")
    result={}
    null_hashes={}
    for key,row in zip(expected,report["cases"],strict=True):
        observed=(row["seed"],row["category"],row["guidance"])
        if observed!=key or type(row["seed"]) is not int:
            raise ArchiveError("Generated report case order changed")
        require_sha256(row["waveform_sha256"])
        if row["guidance"]==0:
            if null_hashes.setdefault(row["seed"],row["waveform_sha256"])!=row["waveform_sha256"]:
                raise ArchiveError("Null output depends on requested category")
        ref=row["reference"]
        if type(ref["strict_win"]) is not bool or type(ref["tied_best"]) is not bool:
            raise ArchiveError("Invalid generated reference outcome")
        if type(row["signal"]["silent_or_constant"]) is not bool:
            raise ArchiveError("Invalid generated signal outcome")
        for metric in REFERENCE_METRICS:
            _number(ref[metric],optional=True)
        for metric in SIGNAL_METRICS:
            value=_number(row["signal"][metric],optional=metric=="crest_factor")
            if value is not None and value<0:
                raise ArchiveError("Negative generated amplitude statistic")
        if row["signal"]["overrange_fraction"]>1 or row["signal"]["near_clip_fraction"]>1:
            raise ArchiveError("Invalid generated amplitude fraction")
        result[key]=row
    if [null_hashes[seed] for seed in config.seeds]!=report["null_output_sha256_by_seed"]:
        raise ArchiveError("Generated null trace does not match case panel")
    return result


def _mean_pairs(first,second,keys,metric):
    differences=[]
    for key in keys:
        a,b=first[key],second[key]
        if metric=="strict_target_win":
            x,y=float(a["reference"]["strict_win"]),float(b["reference"]["strict_win"])
        elif metric=="silent_or_constant":
            x,y=float(a["signal"]["silent_or_constant"]),float(b["signal"]["silent_or_constant"])
        elif metric in REFERENCE_METRICS:
            x,y=a["reference"][metric],b["reference"][metric]
        else:
            x,y=a["signal"][metric],b["signal"][metric]
        if x is not None and y is not None:
            differences.append(y-x)
    return {"paired_cases":len(keys),"defined_pairs":len(differences),
            "mean_delta":statistics.fmean(differences) if differences else None}


def _summarize_seed_values(values):
    defined=[value for value in values if value is not None]
    return {
        "declared_training_seeds":len(values),"defined_training_seeds":len(defined),
        "missing_training_seeds":len(values)-len(defined),
        "mean":statistics.fmean(defined) if defined else None,
        "median":statistics.median(defined) if defined else None,
        "sample_stddev":statistics.stdev(defined) if len(defined)>1 else None,
        "min":min(defined) if defined else None,"max":max(defined) if defined else None,
        "positive":sum(value>0 for value in defined),
        "zero":sum(value==0 for value in defined),"negative":sum(value<0 for value in defined),
        # A complete panel is required for a complete-seed interpretation.
        "all_seed_values_defined":len(defined)==len(values),
    }


def summarize_generated_study(runs,plan,config):
    forwards=preflight_generation(plan,config)
    if len(runs)!=len(plan.seeds):
        raise ArchiveError("Generated study seed count differs from plan")
    shared_pin=None
    summaries={}
    per_seed=[]
    metrics=("strict_target_win","silent_or_constant",*REFERENCE_METRICS,*SIGNAL_METRICS)
    for seed,run in zip(plan.seeds,runs,strict=True):
        if type(run["seed"]) is not int or run["seed"]!=seed:
            raise ArchiveError("Generated study seed order changed")
        reports={
            "initial":run["initial_generation"],
            **{name:run["arms"][name]["generation"] for name in ARMS},
        }
        hashes={"initial":run["initialization"]["model_sha256"],
                **{name:run["arms"][name]["trained_model_sha256"] for name in ARMS}}
        cases={}
        for name,report in reports.items():
            if report["model_sha256"]!=hashes[name]:
                raise ArchiveError("Generated assay model fingerprint changed")
            pin=_identity(report)
            if shared_pin is None:
                shared_pin=pin
            if pin!=shared_pin:
                raise ArchiveError("Generated study reference or sampling panel changed")
            cases[name]=_case_map(report,plan,config)
        comparisons={}
        for left,right in (("initial","matched"),("initial","random_label"),("random_label","matched")):
            name=f"{right}_minus_{left}"
            comparisons[name]={}
            for guidance in config.guidances:
                for category in ("all",*CATEGORIES):
                    group=f"{float(guidance)}/{category}"
                    keys=[key for key in cases[left] if key[2]==guidance and (category=="all" or key[1]==category)]
                    comparisons[name][group]={}
                    for metric in metrics:
                        paired=_mean_pairs(cases[left],cases[right],keys,metric)
                        comparisons[name][group][metric]=paired
                        summaries.setdefault(name,{}).setdefault(group,{}).setdefault(metric,[]).append(paired["mean_delta"])
        per_seed.append({"seed":seed,"comparisons":comparisons})
    return {
        "protocol":PROTOCOL,"generation_plan":asdict(config),
        "total_forward_calls":forwards,"per_seed":per_seed,
        "paired_seed_summaries":{
            comparison:{group:{metric:_summarize_seed_values(values) for metric,values in items.items()}
                        for group,items in groups.items()}
            for comparison,groups in summaries.items()
        },
        "reference_panel_and_noise_trace_shared":True,
        "unscorable_cases_excluded_from_continuous_pair_means":True,
        "strict_win_denominators_include_unscorable_cases":True,
        "statistical_significance_claimed":False,"generated_audio_semantics_established":False,
    }
