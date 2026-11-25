"""Two-stage original crop latent generation on first-party synthetic examples.

Codec training, frozen train-only calibration, class-conditioned velocity
training, paired held-out diagnostics and deterministic latent sampling.
No local private corpus is needed and no model/audio is saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import statistics
import struct

import numpy as np
import torch

from research.crop_codec import CropAudioCodec,codec_loss,reconstruction_metrics
from research.crop_codec_packet import CropCodecError,BITS,code_shape,payload_rate_report
from research.crop_latent import (
    CropLatentDenoiser,encode_native,fit_calibration,restore_for_decoder,
    velocity_loss,sample_latent,reverse_indices,
)
from research.wave_diffusion import dataset,CATEGORIES,CLIP_FRAMES,DiffusionSchedule
from research.velocity_prediction import velocity_target,decode_velocity
from research.weight_ema import model_state_sha256

PROTOCOL="sonumio-crop-velocity-latent-study-v1"


@dataclass(frozen=True,slots=True)
class LatentStudyPlan:
    seed:int=17
    codec_steps:int=32
    diffusion_updates:int=64
    batch_size:int=4
    bits:int|None=8
    width:int=32
    schedule_steps:int=32
    reverse_steps:int=8
    training_seeds:tuple[int,...]=(0,1,2)
    validation_seeds:tuple[int,...]=(101,102)
    generation_seeds:tuple[int,...]=(701,709)

    def __post_init__(self):
        for value,lo,hi,name in (
            (self.seed,0,2**32-1,"seed"),(self.codec_steps,1,500,"codec steps"),
            (self.diffusion_updates,1,2000,"diffusion updates"),(self.batch_size,1,16,"batch"),
            (self.schedule_steps,8,256,"schedule"),
        ):
            if type(value) is not int or not lo<=value<=hi:
                raise CropCodecError("Invalid latent study "+name)
        if self.bits is not None and (type(self.bits) is not int or self.bits not in BITS):
            raise CropCodecError("Invalid latent study precision")
        if type(self.width) is not int or self.width not in (16,32,48,64):
            raise CropCodecError("Invalid latent study width")
        reverse_indices(self.schedule_steps,self.reverse_steps)
        for seeds,minimum,name in ((self.training_seeds,1,"training"),(self.validation_seeds,1,"validation"),
                                   (self.generation_seeds,2,"generation")):
            if (type(seeds) is not tuple or not minimum<=len(seeds)<=8
                    or any(type(x) is not int or not 0<=x<2**32 for x in seeds)
                    or len(set(seeds))!=len(seeds)):
                raise CropCodecError("Invalid latent study "+name+" seeds")
        if set(self.training_seeds)&set(self.validation_seeds):
            raise CropCodecError("Latent study source seeds overlap")
        if (self.codec_steps+self.diffusion_updates)*self.batch_size>40_000:
            raise CropCodecError("Latent study training budget exceeded")


def _generator(seed,domain,counter=0):
    key=hashlib.sha256(b"sonumio-crop-latent-study-v1\0"+domain+struct.pack("<II",seed,counter)).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(key[:8],"little")&((1<<63)-1))


def _hash_values(trace,tensor,dtype):
    data=np.asarray(tensor.detach().numpy(),dtype=dtype,order="C").tobytes()
    trace.update(len(data).to_bytes(8,"little"))
    trace.update(data)


def _batches(count,size,steps,generator):
    pool=[]
    for _ in range(steps):
        while len(pool)<size:
            pool.extend(torch.randperm(count,generator=generator).tolist())
        yield torch.tensor(pool[:size],dtype=torch.long)
        del pool[:size]


def _chunks(values,size):
    for start in range(0,len(values),size):
        yield values[start:start+size]


def _mean(rows):
    if not rows:
        raise CropCodecError("Empty latent study metric panel")
    return {key:statistics.fmean(row[key] for row in rows) for key in rows[0]}


def _train_codec(plan,codec,train):
    optimizer=torch.optim.AdamW(codec.parameters(),lr=.001,foreach=False,fused=False)
    trace=hashlib.sha256(b"sonumio-crop-codec-training-indices-v1\0")
    losses=[]
    for indices in _batches(len(train),plan.batch_size,plan.codec_steps,_generator(plan.seed,b"codec")):
        _hash_values(trace,indices,"<i8")
        wave=train[indices]
        optimizer.zero_grad(set_to_none=True)
        loss=codec_loss(wave,codec(wave,plan.bits),frames=CLIP_FRAMES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    codec.eval().requires_grad_(False)
    return {"steps":plan.codec_steps,"first_loss":losses[0],"last_loss":losses[-1],
            "mean_loss":statistics.fmean(losses),"training_index_trace_sha256":trace.hexdigest()}


def _train_denoiser(plan,model,schedule,train_latents,labels):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,fused=False)
    trace=hashlib.sha256(b"sonumio-crop-velocity-training-trace-v1\0")
    losses=[]
    dropped_total=0
    for index,indices in enumerate(_batches(len(train_latents),plan.batch_size,plan.diffusion_updates,
                                            _generator(plan.seed,b"denoiser-indices"))):
        clean=train_latents[indices]
        steps=torch.randint(schedule.steps,(len(indices),),generator=_generator(plan.seed,b"timestep",index))
        noise=torch.randn(clean.shape,generator=_generator(plan.seed,b"noise",index),dtype=torch.float32)
        dropped=torch.rand(len(indices),generator=_generator(plan.seed,b"drop",index))<.1
        condition=torch.where(dropped,model.classes,labels[indices])
        dropped_total+=int(dropped.sum())
        for values,dtype in ((indices,"<i8"),(steps,"<i8"),(noise,"<f4"),(condition,"<i8")):
            _hash_values(trace,values,dtype)
        optimizer.zero_grad(set_to_none=True)
        loss=velocity_loss(model,schedule,clean,steps,noise,condition)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"steps":plan.diffusion_updates,"first_loss":losses[0],"last_loss":losses[-1],
            "mean_loss":statistics.fmean(losses),"training_trace_sha256":trace.hexdigest(),
            "null_examples":dropped_total}


@torch.no_grad()
def _validation(plan,codec,model,schedule,calibration,waves,labels):
    mode=model.training
    model.eval()
    rows=[]
    trace=hashlib.sha256(b"sonumio-crop-latent-validation-trace-v1\0")
    panel=(0,(schedule.steps-1)//2,schedule.steps-1)
    try:
        for index,(wave,label) in enumerate(zip(waves,labels,strict=True)):
            source=wave[None]
            native=encode_native(codec,source,plan.bits)
            clean=calibration.normalize(native)
            reconstructed=codec.decode(native)
            codec_metrics={key:float(value[0]) for key,value in reconstruction_metrics(
                source,reconstructed,frames=CLIP_FRAMES).items()}
            for level,t in enumerate(panel):
                timestep=torch.tensor([t],dtype=torch.long)
                noise=torch.randn(clean.shape,generator=_generator(301,b"validation",index*3+level),
                                  dtype=torch.float32)
                _hash_values(trace,timestep,"<i8")
                _hash_values(trace,noise,"<f4")
                target=velocity_target(clean,noise,timestep,schedule)
                noisy=schedule.corrupt(clean,timestep,noise)
                prediction=model(noisy,timestep,label[None])
                null=model(noisy,timestep,torch.tensor([model.classes]))
                wrong=model(noisy,timestep,((label+1)%model.classes)[None])
                estimated,_=decode_velocity(noisy,prediction,timestep,schedule)
                projection=restore_for_decoder(estimated,calibration,bits=plan.bits)
                waveform=codec.decode(projection.latent)
                decoded={key:float(value[0]) for key,value in reconstruction_metrics(
                    source,waveform,frames=CLIP_FRAMES).items()}
                rows.append({"category":CATEGORIES[int(label)],"timestep":t,
                             "velocity":{
                                 "matched_mse":float((prediction-target).double().square().mean()),
                                 "null_mse":float((null-target).double().square().mean()),
                                 "rotated_wrong_mse":float((wrong-target).double().square().mean()),
                                 "zero_predictor_mse":float(target.double().square().mean()),
                             },
                             "codec_reconstruction":codec_metrics,"decoded_x0":decoded,
                             "decoder_projection":projection.diagnostics})
    finally:
        model.train(mode)
    return {
        "rows":rows,"timesteps":list(panel),"trace_sha256":trace.hexdigest(),
        "clip_timestep_cases":len(rows),"heldout_clips":len(waves),
        "velocity_mean":_mean([r["velocity"] for r in rows]),
        "by_timestep":{
            str(t):{
                "velocity":_mean([r["velocity"] for r in rows if r["timestep"]==t]),
                "decoded_x0":_mean([r["decoded_x0"] for r in rows if r["timestep"]==t]),
                "codec_reconstruction":_mean([r["codec_reconstruction"] for r in rows if r["timestep"]==t]),
            } for t in panel
        },
        "teacher_forced_not_generation":True,
    }


@torch.no_grad()
def _generated(plan,codec,model,schedule,calibration):
    trace=hashlib.sha256(b"sonumio-crop-latent-generation-noise-v1\0")
    rows=[]
    for seed in plan.generation_seeds:
        noise=torch.randn((1,*code_shape(CLIP_FRAMES)),generator=_generator(seed,b"generation"),
                          dtype=torch.float32)
        _hash_values(trace,noise,"<f4")
        null_hash=None
        for label,category in enumerate(CATEGORIES):
            for guidance in (0.,1.,2.):
                standardized=sample_latent(model,schedule,noise,label,guidance=guidance,budget=plan.reverse_steps)
                projection=restore_for_decoder(standardized,calibration,bits=plan.bits)
                pre,wave=codec.decode_with_preactivation(projection.latent)
                waveform_hash=hashlib.sha256(np.asarray(wave.numpy(),dtype="<f4").tobytes()).hexdigest()
                if guidance==0:
                    if null_hash is None:
                        null_hash=waveform_hash
                    elif null_hash!=waveform_hash:
                        raise CropCodecError("Null latent generation changed with category")
                rows.append({
                    "seed":seed,"category":category,"guidance":guidance,"waveform_sha256":waveform_hash,
                    "decoder_projection":projection.diagnostics,
                    "signal":{
                        "rms":float(wave.double().square().mean().sqrt()),
                        "centered_rms":float((wave.double()-wave.double().mean()).square().mean().sqrt()),
                        "peak":float(wave.abs().max()),
                        "absolute_dc":abs(float(wave.double().mean())),
                        "near_clip_fraction":float((wave.abs()>=.999).float().mean()),
                        "pre_tanh_peak":float(pre.abs().max()),
                        "pre_tanh_overrange_fraction":float((pre.abs()>1).float().mean()),
                    },
                })
    return {"rows":rows,"noise_trace_sha256":trace.hexdigest(),"null_control_verified":True,
            "reverse_indices":list(reverse_indices(schedule.steps,plan.reverse_steps)),
            "by_guidance":{str(g):_mean([r["signal"] for r in rows if r["guidance"]==g])
                           for g in (0.,1.,2.)},
            "generated_audio_semantics_established":False}


def run_study(plan=None):
    plan=LatentStudyPlan() if plan is None else plan
    if not isinstance(plan,LatentStudyPlan):
        raise CropCodecError("Expected LatentStudyPlan")
    train,labels,train_hash=dataset(plan.training_seeds)
    validation,val_labels,val_hash=dataset(plan.validation_seeds)
    train_hashes={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in train}
    val_hashes={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in validation}
    if len(train_hashes)!=len(train) or len(val_hashes)!=len(validation) or train_hashes&val_hashes:
        raise CropCodecError("Exact synthetic crop overlap")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(plan.seed)
        with torch.device("cpu"):
            codec=CropAudioCodec(CLIP_FRAMES)
        codec_initial=model_state_sha256(codec)
        codec_training=_train_codec(plan,codec,train)
        codec_final=model_state_sha256(codec)
        if codec_initial==codec_final:
            raise CropCodecError("Crop codec did not train")
        calibration=fit_calibration(codec,_chunks(train,plan.batch_size),bits=plan.bits)
        train_latents=torch.cat([calibration.normalize(encode_native(codec,batch,plan.bits))
                                for batch in _chunks(train,plan.batch_size)])
        torch.random.default_generator.manual_seed(plan.seed)
        with torch.device("cpu"):
            model=CropLatentDenoiser(CLIP_FRAMES,plan.width)
            schedule=DiffusionSchedule(plan.schedule_steps,profile="cosine_zero_terminal_v1")
        initial=model_state_sha256(model)
        training=_train_denoiser(plan,model,schedule,train_latents,labels)
        trained=model_state_sha256(model)
        if initial==trained or model_state_sha256(codec)!=codec_final:
            raise CropCodecError("Crop latent training state integrity failure")
        validation_report=_validation(plan,codec,model,schedule,calibration,validation,val_labels)
        generation_report=_generated(plan,codec,model,schedule,calibration)
        if model_state_sha256(codec)!=codec_final or model_state_sha256(model)!=trained:
            raise CropCodecError("Crop latent evaluation mutated model state")
    return {
        "protocol":PROTOCOL,"configuration":asdict(plan),
        "training_data_sha256":train_hash,"validation_data_sha256":val_hash,
        "training_clips":len(train),"validation_clips":len(validation),
        "exact_cross_split_crop_overlap":False,
        "codec":{"initial_sha256":codec_initial,"trained_sha256":codec_final,"training":codec_training,
                 "frozen_during_diffusion":True,"calibration":calibration.report(),
                 "rate":payload_rate_report(CLIP_FRAMES,plan.bits) if plan.bits is not None else None},
        "denoiser":{"initial_sha256":initial,"trained_sha256":trained,"training":training,
                    "parameter_count":sum(p.numel() for p in model.parameters()),
                    "objective":"velocity","schedule":"cosine_zero_terminal_v1",
                    "terminal_alpha_bar":float(schedule.alpha_bars[-1])},
        "validation":validation_report,"generation":generation_report,
        "no_model_weights_or_audio_written":True,"model_publication_authorized":False,
        "perceptual_quality_established":False,"original_sonumio_1":False,
    }


def _seeds(raw):
    if type(raw) is not str or len(raw)>100:
        raise CropCodecError("Invalid seed list")
    fields=raw.split(",")
    if any(not x.isascii() or not x.isdecimal() or len(x)>10 or str(int(x))!=x for x in fields):
        raise CropCodecError("Invalid canonical seed list")
    return tuple(int(x) for x in fields)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    defaults=LatentStudyPlan()
    for name in ("seed","codec_steps","diffusion_updates","batch_size","width","schedule_steps","reverse_steps"):
        parser.add_argument("--"+name.replace("_","-"),type=int,default=getattr(defaults,name))
    parser.add_argument("--bits",choices=("float","2","4","6","8"),default="8")
    for name in ("training_seeds","validation_seeds","generation_seeds"):
        parser.add_argument("--"+name.replace("_","-"),default=",".join(map(str,getattr(defaults,name))))
    args=parser.parse_args(argv)
    try:
        values=vars(args).copy()
        values["bits"]=None if args.bits=="float" else int(args.bits)
        for name in ("training_seeds","validation_seeds","generation_seeds"):
            values[name]=_seeds(values[name])
        report=run_study(LatentStudyPlan(**values))
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Crop latent study: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
