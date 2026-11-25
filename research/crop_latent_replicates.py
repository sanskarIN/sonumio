"""Independent-seed precision ablation of the two-stage crop latent generator.

The representation changes across arms, so raw standardized latent velocity
MSE is never subtracted between arms. All cross-arm quality diagnostics are
sample-aligned waveform errors, or numerical generated-waveform statistics.
This is descriptive first-party synthetic research, not perceptual evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass,replace
import hashlib
import json
import math
import statistics

from research.crop_codec_packet import CropCodecError,BITS
from research.crop_latent_study import LatentStudyPlan,run_study,PROTOCOL as SINGLE_PROTOCOL
from research.wave_diffusion import CATEGORIES

PROTOCOL="sonumio-crop-latent-precision-replicates-v1"
MAX_TOTAL_TRAIN_EXAMPLES=100_000
MAX_TOTAL_GENERATION_FORWARDS=100_000
WAVEFORM_METRICS=("waveform_l1","waveform_mse","preemphasis_l1","log_magnitude_l1",
                  "spectral_convergence","envelope_l1","snr_db")
GENERATED_METRICS=("rms","centered_rms","peak","absolute_dc","near_clip_fraction",
                   "pre_tanh_peak","pre_tanh_overrange_fraction")
PROJECTION_METRICS=("native_peak_before_projection","native_outside_bound_fraction",
                    "clamp_mse","requantization_mse")


@dataclass(frozen=True,slots=True)
class ReplicatePlan:
    model_seeds:tuple[int,...]=(17,29,41)
    precision_bits:tuple[int,...]=(4,8)
    base:LatentStudyPlan=LatentStudyPlan()

    def __post_init__(self):
        if (type(self.model_seeds) is not tuple or not 2<=len(self.model_seeds)<=8
                or any(type(x) is not int or not 0<=x<2**32 for x in self.model_seeds)
                or len(set(self.model_seeds))!=len(self.model_seeds)):
            raise CropCodecError("Use 2..8 independent model seeds")
        if (type(self.precision_bits) is not tuple or not 1<=len(self.precision_bits)<=len(BITS)
                or any(type(x) is not int or x not in BITS for x in self.precision_bits)
                or len(set(self.precision_bits))!=len(self.precision_bits)):
            raise CropCodecError("Invalid replicate precision arms")
        if not isinstance(self.base,LatentStudyPlan):
            raise CropCodecError("Expected validated latent study settings")
        if self.train_examples>MAX_TOTAL_TRAIN_EXAMPLES or self.generation_forwards>MAX_TOTAL_GENERATION_FORWARDS:
            raise CropCodecError("Latent precision replicate workload exceeds budget")

    @property
    def arms(self):
        return ("float",*(f"q{x}" for x in self.precision_bits))

    @property
    def train_examples(self):
        return len(self.model_seeds)*len(self.arms)*(self.base.codec_steps+self.base.diffusion_updates)*self.base.batch_size

    @property
    def generation_forwards(self):
        # Guidance 0/1 use one forward; guidance 2 uses two.
        return len(self.model_seeds)*len(self.arms)*len(self.base.generation_seeds)*len(CATEGORIES)*self.base.reverse_steps*4

    def as_dict(self):
        return {
            "model_seeds":list(self.model_seeds),"precision_bits":list(self.precision_bits),
            "shared_configuration":{k:v for k,v in asdict(self.base).items() if k not in ("seed","bits")},
        }

    def fingerprint(self):
        raw=json.dumps({"protocol":PROTOCOL,"plan":self.as_dict()},sort_keys=True,
                       separators=(",",":"),allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()


def _finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise CropCodecError("Invalid finite replicate metric")
    return float(value)


def _digest(value):
    if type(value) is not str or len(value)!=64 or any(x not in "0123456789abcdef" for x in value):
        raise CropCodecError("Invalid replicate SHA256")
    return value


def _summary(values):
    if len(values)<2:
        raise CropCodecError("Replicate summary requires independent model seeds")
    values=[_finite(x) for x in values]
    return {"seeds":len(values),"mean":statistics.fmean(values),"median":statistics.median(values),
            "sample_stddev":statistics.stdev(values),"min":min(values),"max":max(values),
            "positive":sum(x>0 for x in values),"negative":sum(x<0 for x in values),
            "zero":sum(x==0 for x in values)}


def _pin(report,plan,seed,bits):
    if report["protocol"]!=SINGLE_PROTOCOL:
        raise CropCodecError("Unexpected latent replicate protocol")
    configuration=asdict(replace(plan.base,seed=seed,bits=bits))
    # JSON representation makes in-memory tuples and reloaded arrays equivalent.
    if json.dumps(configuration,sort_keys=True)!=json.dumps(report["configuration"],sort_keys=True):
        raise CropCodecError("Replicate configuration differs from prespecified arm")
    if (report["exact_cross_split_crop_overlap"] is not False
            or report["no_model_weights_or_audio_written"] is not True
            or report["codec"]["frozen_during_diffusion"] is not True
            or report["generation"]["null_control_verified"] is not True):
        raise CropCodecError("Replicate experiment integrity flags changed")
    for key in ("training_data_sha256","validation_data_sha256"):
        _digest(report[key])
    for section in ("codec","denoiser"):
        if _digest(report[section]["initial_sha256"])==_digest(report[section]["trained_sha256"]):
            raise CropCodecError("Replicate model did not train")
    if report["training_clips"]!=len(CATEGORIES)*len(plan.base.training_seeds):
        raise CropCodecError("Replicate training clip count changed")
    if report["validation_clips"]!=len(CATEGORIES)*len(plan.base.validation_seeds):
        raise CropCodecError("Replicate validation clip count changed")
    if report["codec"]["calibration"]["training_clips"]!=report["training_clips"]:
        raise CropCodecError("Replicate calibration count differs from training")
    return (
        report["training_data_sha256"],report["validation_data_sha256"],
        report["training_clips"],report["validation_clips"],
        _digest(report["validation"]["trace_sha256"]),
        _digest(report["generation"]["noise_trace_sha256"]),
    )


def _paired_train_pin(report):
    return (
        report["codec"]["initial_sha256"],report["denoiser"]["initial_sha256"],
        _digest(report["codec"]["training"]["training_index_trace_sha256"]),
        _digest(report["denoiser"]["training"]["training_trace_sha256"]),
        report["denoiser"]["training"]["null_examples"],
    )


def _validation_rows(report,plan):
    levels=(0,(plan.base.schedule_steps-1)//2,plan.base.schedule_steps-1)
    expected=[(category,index,t) for category in CATEGORIES
              for index in range(len(plan.base.validation_seeds)) for t in levels]
    rows=report["validation"]["rows"]
    if len(rows)!=len(expected):
        raise CropCodecError("Replicate validation panel count changed")
    result={}
    for (category,index,t),row in zip(expected,rows,strict=True):
        if row["category"]!=category or row["timestep"]!=t:
            raise CropCodecError("Replicate validation panel order changed")
        for section in ("codec_reconstruction","decoded_x0"):
            if set(row[section])!=set(WAVEFORM_METRICS):
                raise CropCodecError("Replicate waveform metric schema changed")
            for value in row[section].values():
                _finite(value)
        for metric in PROJECTION_METRICS:
            _finite(row["decoder_projection"][metric])
        result[(category,index,t)]=row
    return result


def _generation_rows(report,plan):
    expected=[(seed,category,g) for seed in plan.base.generation_seeds
              for category in CATEGORIES for g in (0.,1.,2.)]
    rows=report["generation"]["rows"]
    if len(rows)!=len(expected):
        raise CropCodecError("Replicate generation panel count changed")
    result={}
    null={}
    for key,row in zip(expected,rows,strict=True):
        if (row["seed"],row["category"],row["guidance"])!=key:
            raise CropCodecError("Replicate generation panel order changed")
        pin=_digest(row["waveform_sha256"])
        if key[2]==0:
            if null.setdefault(key[0],pin)!=pin:
                raise CropCodecError("Replicate null control changed with category")
        if set(row["signal"])!=set(GENERATED_METRICS):
            raise CropCodecError("Replicate generated signal schema changed")
        for value in row["signal"].values():
            _finite(value)
        for metric in PROJECTION_METRICS:
            _finite(row["decoder_projection"][metric])
        result[key]=row
    return result


def _pair_mean(a,b,keys,section,metrics):
    if not keys:
        raise CropCodecError("Empty prespecified paired replicate stratum")
    return {metric:statistics.fmean(_finite(b[key][section][metric])-_finite(a[key][section][metric])
                                   for key in keys) for metric in metrics}


def _compare_rows(base,arm,plan):
    a,b=_validation_rows(base,plan),_validation_rows(arm,plan)
    c,d=_generation_rows(base,plan),_generation_rows(arm,plan)
    if a.keys()!=b.keys() or c.keys()!=d.keys():
        raise CropCodecError("Replicate paired case identities changed")
    result={}
    for name in ("codec_reconstruction","decoded_x0"):
        result[name]={}
        for category in ("all",*CATEGORIES):
            keys=[key for key in a if category=="all" or key[0]==category]
            result[name][category]=_pair_mean(a,b,keys,name,WAVEFORM_METRICS)
    for t in (0,(plan.base.schedule_steps-1)//2,plan.base.schedule_steps-1):
        keys=[key for key in a if key[2]==t]
        result.setdefault("decoded_x0_by_timestep",{})[str(t)]=_pair_mean(a,b,keys,"decoded_x0",WAVEFORM_METRICS)
        result.setdefault("validation_projection",{})[str(t)]=_pair_mean(a,b,keys,"decoder_projection",PROJECTION_METRICS)
    for guidance in (0.,1.,2.):
        keys=[key for key in c if key[2]==guidance]
        result.setdefault("generated_signal",{})[str(guidance)]=_pair_mean(c,d,keys,"signal",GENERATED_METRICS)
        result.setdefault("generated_projection",{})[str(guidance)]=_pair_mean(c,d,keys,"decoder_projection",PROJECTION_METRICS)
    return result


def _append_tree(target,tree):
    for key,value in tree.items():
        if isinstance(value,dict):
            _append_tree(target.setdefault(key,{}),value)
        else:
            target.setdefault(key,[]).append(_finite(value))


def _summarize_tree(tree):
    return {key:_summarize_tree(value) if isinstance(value,dict) else _summary(value)
            for key,value in tree.items()}


def summarize_replicates(runs,plan):
    if not isinstance(plan,ReplicatePlan) or len(runs)!=len(plan.model_seeds):
        raise CropCodecError("Replicate run count changed")
    shared=None
    seed_states=[]
    per_seed=[]
    series={}
    for seed,run in zip(plan.model_seeds,runs,strict=True):
        if type(run["seed"]) is not int or run["seed"]!=seed or set(run["arms"])!=set(plan.arms):
            raise CropCodecError("Replicate seed or arm panel changed")
        base=run["arms"]["float"]
        pin=_paired_train_pin(base)
        seed_states.append(pin[:2])
        deltas={}
        for name,bits in zip(plan.arms,(None,*plan.precision_bits),strict=True):
            report=run["arms"][name]
            candidate=_pin(report,plan,seed,bits)
            if shared is None:
                shared=candidate
            if candidate!=shared:
                raise CropCodecError("Replicate validation or generated-noise panel differs")
            if _paired_train_pin(report)!=pin:
                raise CropCodecError("Within-seed training initialization or corruption changed")
            if name!="float":
                deltas[name]=_compare_rows(base,report,plan)
                _append_tree(series.setdefault(name,{}),deltas[name])
        per_seed.append({"seed":seed,"waveform_domain_deltas_from_float":deltas})
    if len(set(seed_states))!=len(seed_states):
        raise CropCodecError("Replicate initializations are not distinct")
    return {
        "per_seed":per_seed,"waveform_domain_seed_summaries":_summarize_tree(series),
        "shared_data_and_evaluation_noise_verified":True,
        "shared_within_seed_training_traces_verified":True,
        "distinct_initializations_across_seeds_verified":True,
        "latent_velocity_mse_compared_across_precision_arms":False,
        "statistical_significance_claimed":False,
    }


def run_replicates(plan=None):
    plan=ReplicatePlan() if plan is None else plan
    if not isinstance(plan,ReplicatePlan):
        raise CropCodecError("Expected ReplicatePlan")
    runs=[]
    for seed in plan.model_seeds:
        arms={}
        for name,bits in zip(plan.arms,(None,*plan.precision_bits),strict=True):
            arms[name]=run_study(replace(plan.base,seed=seed,bits=bits))
        runs.append({"seed":seed,"arms":arms})
    return {
        "protocol":PROTOCOL,"plan":plan.as_dict(),"plan_sha256":plan.fingerprint(),
        "training_examples":plan.train_examples,"generation_forward_calls":plan.generation_forwards,
        "runs":runs,"paired_comparison":summarize_replicates(runs,plan),
        "no_audio_or_weights_written":True,"model_publication_authorized":False,
        "perceptual_quality_established":False,"original_sonumio_1":False,
    }


def _list(value):
    if type(value) is not str or len(value)>100:
        raise CropCodecError("Invalid replicate integer list")
    parts=value.split(",")
    if any(not x.isascii() or not x.isdecimal() or len(x)>10 or str(int(x))!=x for x in parts):
        raise CropCodecError("Invalid canonical integer list")
    return tuple(map(int,parts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=ReplicatePlan()
    parser.add_argument("--model-seeds",default=",".join(map(str,defaults.model_seeds)))
    parser.add_argument("--precision-bits",default=",".join(map(str,defaults.precision_bits)))
    for name in ("codec_steps","diffusion_updates","batch_size","width","schedule_steps","reverse_steps"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(defaults.base,name))
    for name in ("training_seeds","validation_seeds","generation_seeds"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(defaults.base,name))))
    args=parser.parse_args(argv)
    try:
        config={key:getattr(args,key) for key in ("codec_steps","diffusion_updates","batch_size","width",
                                                 "schedule_steps","reverse_steps")}
        for name in ("training_seeds","validation_seeds","generation_seeds"):
            config[name]=_list(getattr(args,name))
        plan=ReplicatePlan(_list(args.model_seeds),_list(args.precision_bits),LatentStudyPlan(**config))
        report=run_replicates(plan)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Crop latent replicates: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
