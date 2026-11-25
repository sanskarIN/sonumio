"""Paired crop codec rate-distortion experiment on first-party synthetic sounds.

One seed creates a common model initialization, then all precision arms
receive exactly the same batches and optimizer configuration. Test samples
come from disjoint DSP seeds. No actual audio or neural state is written.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import statistics

import numpy as np
import torch

from research.crop_codec import CropAudioCodec,codec_loss,reconstruction_metrics,quantize_latent
from research.crop_codec_packet import CropCodecError,BITS,payload_rate_report
from research.wave_diffusion import dataset,CATEGORIES,CLIP_FRAMES
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-crop-codec-rate-distortion-v1"
MAX_UPDATES=500
MAX_TRAIN_EXAMPLES=50_000


@dataclass(frozen=True,slots=True)
class RateDistortionPlan:
    model_seeds:tuple[int,...]=(17,29,41)
    training_seeds:tuple[int,...]=(0,1,2)
    validation_seeds:tuple[int,...]=(101,102)
    precision_bits:tuple[int,...]=(4,8)
    updates:int=32
    batch_size:int=4

    def __post_init__(self):
        for name,seeds,minimum,maximum in (
            ("model",self.model_seeds,2,8),("training",self.training_seeds,1,8),
            ("validation",self.validation_seeds,1,8),
        ):
            if (type(seeds) is not tuple or not minimum<=len(seeds)<=maximum
                    or any(type(seed) is not int or not 0<=seed<2**32 for seed in seeds)
                    or len(set(seeds))!=len(seeds)):
                raise CropCodecError("Invalid "+name+" seed list")
        if set(self.training_seeds)&set(self.validation_seeds):
            raise CropCodecError("Training and held-out DSP seeds overlap")
        if (type(self.precision_bits) is not tuple or not 1<=len(self.precision_bits)<=len(BITS)
                or any(type(bits) is not int or bits not in BITS for bits in self.precision_bits)
                or len(set(self.precision_bits))!=len(self.precision_bits)):
            raise CropCodecError("Invalid precision arm list")
        if type(self.updates) is not int or not 1<=self.updates<=MAX_UPDATES:
            raise CropCodecError("Invalid codec study update count")
        if type(self.batch_size) is not int or not 1<=self.batch_size<=16:
            raise CropCodecError("Invalid codec study batch size")
        if self.train_examples>MAX_TRAIN_EXAMPLES:
            raise CropCodecError("Codec study exceeds training workload budget")

    @property
    def arms(self):
        return ("float",*(f"q{bits}" for bits in self.precision_bits))

    @property
    def train_examples(self):
        return len(self.model_seeds)*len(self.arms)*self.updates*self.batch_size


def _clip_hashes(wave):
    return [hashlib.sha256(np.asarray(row.numpy(),dtype="<f4").tobytes()).hexdigest() for row in wave]


def _split_data(plan):
    train,train_labels,train_hash=dataset(plan.training_seeds)
    validation,val_labels,val_hash=dataset(plan.validation_seeds)
    a,b=_clip_hashes(train),_clip_hashes(validation)
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)&set(b):
        raise CropCodecError("Exact synthetic source crop duplication")
    return train,train_labels,validation,val_labels,{
        "training_source_sha256":train_hash,"validation_source_sha256":val_hash,
        "training_clips":len(train),"validation_clips":len(validation),
        "exact_cross_split_clip_overlap":False,"source_kind":"first_party_synthetic_dsp",
    }


def _batch_plan(plan,seed,count):
    generator=torch.Generator(device="cpu").manual_seed(seed)
    batches=[]
    pending=[]
    for _ in range(plan.updates):
        while len(pending)<plan.batch_size:
            pending.extend(torch.randperm(count,generator=generator).tolist())
        batches.append(pending[:plan.batch_size])
        del pending[:plan.batch_size]
    trace=hashlib.sha256(np.asarray(batches,dtype="<i8").tobytes()).hexdigest()
    return batches,trace


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=False)


def _mean(rows):
    if not rows:
        raise CropCodecError("Empty codec metric panel")
    keys=tuple(rows[0])
    if any(tuple(row)!=keys for row in rows):
        raise CropCodecError("Codec metric schema changed")
    return {key:statistics.fmean(row[key] for row in rows) for key in keys}


@torch.no_grad()
def _evaluate(model,wave,labels,bits):
    before=model_state_sha256(model)
    mode=model.training
    model.eval()
    rows=[]
    packet_hash=hashlib.sha256(b"sonumio-crop-codec-packets-v1\0")
    try:
        for source,label in zip(wave,labels,strict=True):
            source=source[None]
            latent=model.encode(source)
            if bits is None:
                pre,restored=model.decode_with_preactivation(latent)
            else:
                packets=model.encode_packets(source,bits)
                pre,restored=model.decode_packets(packets)
                exact=model(source,bits)
                if not torch.equal(exact,restored):
                    raise CropCodecError("Packet reconstruction differs from hard quantization")
                packet_hash.update(packets[0])
            metrics={name:float(value[0]) for name,value in reconstruction_metrics(
                source,restored,frames=CLIP_FRAMES).items()}
            metrics.update({
                "latent_abs_mean":float(latent.abs().mean()),
                "latent_near_bound_fraction":float((latent.abs()>=.999).float().mean()),
                "decoder_pre_tanh_peak":float(pre.abs().max()),
                "decoder_pre_tanh_overrange_fraction":float((pre.abs()>1).float().mean()),
            })
            if not all(math.isfinite(v) for v in metrics.values()):
                raise CropCodecError("Invalid finite codec diagnostics")
            rows.append({"category":CATEGORIES[int(label)],"metrics":metrics})
        if model_state_sha256(model)!=before:
            raise CropCodecError("Codec changed during evaluation")
    finally:
        model.train(mode)
    return {
        "model_sha256":before,"clips":len(rows),"rows":rows,
        "clip_mean":_mean([row["metrics"] for row in rows]),
        "by_category":{category:_mean([row["metrics"] for row in rows if row["category"]==category])
                       for category in CATEGORIES},
        "quantized_packet_trace_sha256":packet_hash.hexdigest() if bits is not None else None,
        "packet_roundtrip_verified":bits is not None,
        "rate":payload_rate_report(CLIP_FRAMES,bits) if bits is not None else {
            "waveform_frames":CLIP_FRAMES,"latent_scalars":CLIP_FRAMES//4,
            "scalar_compression_ratio":4.,"float32_latent_bytes":CLIP_FRAMES,
            "packet_bytes":None,"production_bitstream_claimed":False,
        },
    }


def _single_run(plan,seed,train,validation,val_labels):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            initial=CropAudioCodec(CLIP_FRAMES)
        initial_hash=model_state_sha256(initial)
        batches,trace=_batch_plan(plan,seed,len(train))
        arms={}
        for name,bits in zip(plan.arms,(None,*plan.precision_bits),strict=True):
            model=copy.deepcopy(initial)
            if model_state_sha256(model)!=initial_hash:
                raise CropCodecError("Codec arms do not share initialization")
            before=_evaluate(model,validation,val_labels,bits)
            optimizer=_optimizer(model)
            losses=[]
            model.train()
            for indices in batches:
                source=train[indices]
                optimizer.zero_grad(set_to_none=True)
                restored=model(source,bits)
                loss=codec_loss(source,restored,frames=CLIP_FRAMES)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            trained=_evaluate(model,validation,val_labels,bits)
            if trained["model_sha256"]==initial_hash:
                raise CropCodecError("Codec optimizer did not change parameters")
            arms[name]={"precision_bits":bits,"initialization":before,"trained":trained,
                        "first_loss":losses[0],"last_loss":losses[-1],"mean_loss":statistics.fmean(losses)}
        return {"seed":seed,"initial_model_sha256":initial_hash,"batch_trace_sha256":trace,"arms":arms}


def _summary(values):
    if len(values)<2 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):
        raise CropCodecError("Invalid independent seed summary")
    return {"seeds":len(values),"mean":statistics.fmean(values),"median":statistics.median(values),
            "sample_stddev":statistics.stdev(values),"min":min(values),"max":max(values),
            "positive":sum(v>0 for v in values),"negative":sum(v<0 for v in values),
            "zero":sum(v==0 for v in values)}


def summarize(runs,plan):
    if len(runs)!=len(plan.model_seeds):
        raise CropCodecError("Codec seed count mismatch")
    expected_categories=[category for category in CATEGORIES for _ in plan.validation_seeds]
    metrics={}
    for seed,run in zip(plan.model_seeds,runs,strict=True):
        if run["seed"]!=seed or tuple(run["arms"])!=plan.arms:
            raise CropCodecError("Codec study arm or seed drift")
        for name,arm in run["arms"].items():
            initial,trained=arm["initialization"],arm["trained"]
            if initial["model_sha256"]!=run["initial_model_sha256"]:
                raise CropCodecError("Codec initialization fingerprint mismatch")
            if [row["category"] for row in trained["rows"]]!=expected_categories:
                raise CropCodecError("Codec validation panel changed")
            if [row["category"] for row in initial["rows"]]!=expected_categories:
                raise CropCodecError("Codec initialization validation panel changed")
            for key,value in trained["clip_mean"].items():
                base=run["arms"]["float"]["trained"]["clip_mean"][key]
                series=metrics.setdefault(name,{}).setdefault(key,{"trained":[],"delta_from_initial":[],"delta_from_float":[]})
                series["trained"].append(value)
                series["delta_from_initial"].append(value-initial["clip_mean"][key])
                series["delta_from_float"].append(value-base)
    return {name:{key:{variant:_summary(series) for variant,series in measures.items()}
                  for key,measures in items.items()} for name,items in metrics.items()}


def run_study(plan=None):
    plan=RateDistortionPlan() if plan is None else plan
    if not isinstance(plan,RateDistortionPlan):
        raise CropCodecError("Expected RateDistortionPlan")
    train,_,validation,val_labels,data=_split_data(plan)
    runs=[_single_run(plan,seed,train,validation,val_labels) for seed in plan.model_seeds]
    return {
        "protocol":PROTOCOL,"plan":asdict(plan),"data":data,
        "runs":runs,"seed_summaries":summarize(runs,plan),
        "training_examples_across_arms":plan.train_examples,
        "shared_initialization_and_batches_within_seed":True,
        "fixed_width_packets_not_entropy_coded":True,
        "statistical_significance_claimed":False,"perceptual_quality_established":False,
        "audio_or_weights_written":False,"model_publication_authorized":False,
    }


def _list(raw):
    if type(raw) is not str or len(raw)>100:
        raise CropCodecError("Invalid integer list")
    fields=raw.split(",")
    if any(not x.isascii() or not x.isdecimal() or len(x)>10 or str(int(x))!=x for x in fields):
        raise CropCodecError("Expected canonical comma-separated integers")
    return tuple(int(x) for x in fields)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=RateDistortionPlan()
    for name in ("model_seeds","training_seeds","validation_seeds","precision_bits"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(defaults,name))))
    parser.add_argument("--updates",type=int,default=defaults.updates)
    parser.add_argument("--batch-size",type=int,default=defaults.batch_size)
    args=parser.parse_args(argv)
    try:
        plan=RateDistortionPlan(_list(args.model_seeds),_list(args.training_seeds),
                                _list(args.validation_seeds),_list(args.precision_bits),
                                args.updates,args.batch_size)
        report=run_study(plan)
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Crop codec study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
