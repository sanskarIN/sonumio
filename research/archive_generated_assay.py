"""Deterministic paired DDIM signal diagnostics for private waveform checkpoints.

Reverse samples are examined BEFORE clipping. A simple train-fitted spectral
probe is a numerical reference only, never a semantic audio quality oracle.
No generated audio, model weights or individual recording metadata are saved.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

from research.archive_checkpoint import restore_training_checkpoint
from research.archive_readiness import audit_archive
from research.archive_signal_reference import (
    fit_reference,validation_reference_report,spectral_features,signal_statistics,
    classify,MAX_SIGNAL_MAGNITUDE,
)
from research.archive_training_contracts import ArchiveTrainingConfig,make_archive_optimizer
from research.archive_validation import model_fingerprint
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError,require_sha256
from research.wave_diffusion import TinyWaveDenoiser,DiffusionSchedule,CATEGORIES,CLIP_FRAMES

PROTOCOL="sonumio-archive-generated-signal-assay-v1"
MAX_FORWARD_CALLS=50_000
GUIDANCES=(0.,1.,2.)


@dataclass(frozen=True,slots=True)
class GenerationConfig:
    seeds:tuple[int,...]=(701,709,727)
    reverse_steps:int=16
    guidances:tuple[float,...]=GUIDANCES

    def __post_init__(self):
        if (type(self.seeds) is not tuple or not 2<=len(self.seeds)<=8
                or any(type(seed) is not int or not 0<=seed<2**32 for seed in self.seeds)
                or len(set(self.seeds))!=len(self.seeds)):
            raise ArchiveError("Use 2..8 distinct generation seeds")
        if type(self.reverse_steps) is not int or not 2<=self.reverse_steps<=64:
            raise ArchiveError("Use 2..64 reverse sampling steps")
        if (type(self.guidances) is not tuple or not 1<=len(self.guidances)<=4
                or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=4
                       for x in self.guidances) or len(set(self.guidances))!=len(self.guidances)
                or 0 not in self.guidances):
            raise ArchiveError("Guidance panel must include a null-guidance control")
        if self.forward_calls>MAX_FORWARD_CALLS:
            raise ArchiveError("Generated waveform assay exceeds compute budget")

    @property
    def forward_calls(self):
        return len(self.seeds)*len(CATEGORIES)*self.reverse_steps*sum(
            1 if g in (0,1) else 2 for g in self.guidances
        )


def reverse_indices(schedule_steps,reverse_steps):
    if (type(schedule_steps) is not int or not 8<=schedule_steps<=1000
            or type(reverse_steps) is not int or not 2<=reverse_steps<=min(64,schedule_steps)):
        raise ArchiveError("Reverse step count exceeds diffusion schedule")
    # Integer arithmetic ensures unique descending indices, endpoints included.
    return tuple((schedule_steps-1)*(reverse_steps-1-i)//(reverse_steps-1)
                 for i in range(reverse_steps))


def initial_noise(seed):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ArchiveError("Invalid generation noise seed")
    raw=hashlib.sha256(b"sonumio-archive-ddim-initial-v1\0"+struct.pack("<I",seed)).digest()
    generator=torch.Generator(device="cpu").manual_seed(int.from_bytes(raw[:8],"little")&((1<<63)-1))
    return torch.randn((1,1,CLIP_FRAMES),generator=generator,dtype=torch.float32)


@torch.no_grad()
def sample_unclipped(model,schedule,noise,category,*,guidance,reverse_steps):
    """Deterministic eta=0 epsilon-DDIM with explicit final raw waveform.

    All conditions start from the same noise. No dynamic thresholding, clamp
    or fresh randomness is applied. The bounded diagnostic refuses unstable
    intermediates instead of silently clipping them into plausible samples.
    """
    if (not isinstance(model,TinyWaveDenoiser) or model.classes!=len(CATEGORIES)
            or not isinstance(schedule,DiffusionSchedule)
            or any(p.device.type!="cpu" or p.dtype!=torch.float32 for p in model.parameters())
            or schedule.alpha_bars.device.type!="cpu"):
        raise ArchiveError("Expected CPU epsilon waveform model")
    if (type(category) is not int or not 0<=category<len(CATEGORIES)
            or type(guidance) not in (int,float) or not math.isfinite(guidance) or not 0<=guidance<=4):
        raise ArchiveError("Invalid generated waveform condition")
    if (not isinstance(noise,torch.Tensor) or noise.device.type!="cpu"
            or noise.dtype!=torch.float32 or noise.shape!=(1,1,CLIP_FRAMES)
            or not torch.isfinite(noise).all() or noise.abs().max()>MAX_SIGNAL_MAGNITUDE):
        raise ArchiveError("Invalid initial generation noise")
    indices=reverse_indices(schedule.steps,reverse_steps)
    x=noise.clone()
    mode=model.training
    model.eval()
    try:
        for i,index in enumerate(indices):
            t=torch.tensor([index],dtype=torch.long)
            if guidance==0:
                eps=model(x,t,torch.tensor([model.classes]))
            elif guidance==1:
                eps=model(x,t,torch.tensor([category]))
            else:
                unconditional=model(x,t,torch.tensor([model.classes]))
                conditional=model(x,t,torch.tensor([category]))
                eps=unconditional+guidance*(conditional-unconditional)
            if eps.shape!=x.shape or not torch.isfinite(eps).all():
                raise ArchiveError("Nonfinite generated waveform prediction")
            alpha=schedule.alpha_bars[index]
            clean=(x-(1-alpha).sqrt()*eps)/alpha.sqrt()
            if i+1<len(indices):
                previous=schedule.alpha_bars[indices[i+1]]
                x=previous.sqrt()*clean+(1-previous).sqrt()*eps
            else:
                x=clean
            if not torch.isfinite(x).all() or x.abs().max()>MAX_SIGNAL_MAGNITUDE:
                raise ArchiveError("Unstable unclipped generated waveform trajectory")
    finally:
        model.train(mode)
    return x[0,0].numpy().copy()


def _aggregate(rows):
    if not rows:
        raise ArchiveError("Empty generated signal panel")
    count=len(rows)
    stats={}
    for name in ("rms","centered_rms","peak","absolute_dc","overrange_fraction","near_clip_fraction","crest_factor"):
        values=[row["signal"][name] for row in rows if row["signal"][name] is not None]
        stats[name]={"defined":len(values),"mean":float(np.mean(values)) if values else None,
                     "maximum":max(values) if values else None}
    scored=[row["reference"] for row in rows if row["reference"]["target_rank"] is not None]
    return {
        "samples":count,"scorable_samples":len(scored),
        "strict_target_win_fraction":sum(row["reference"]["strict_win"] for row in rows)/count,
        "mean_target_rank":float(np.mean([row["target_rank"] for row in scored])) if scored else None,
        "mean_wrong_minus_target_distance":float(np.mean([row["nearest_wrong_minus_target"] for row in scored])) if scored else None,
        "silent_or_constant_samples":sum(row["signal"]["silent_or_constant"] for row in rows),
        "signal":stats,
    }


def evaluate_generated(model,schedule,reference,reference_validation,*,config=None):
    config=GenerationConfig() if config is None else config
    if not isinstance(config,GenerationConfig):
        raise ArchiveError("Expected GenerationConfig")
    indices=reverse_indices(schedule.steps,config.reverse_steps)
    before=model_fingerprint(model)
    trace=hashlib.sha256(b"sonumio-generated-signal-initial-noise-v1\0")
    rows=[]
    null_outputs=[]
    for seed in config.seeds:
        noise=initial_noise(seed)
        trace.update(struct.pack("<I",seed))
        trace.update(np.asarray(noise.numpy(),dtype="<f4").tobytes())
        null_hash=None
        for label in range(len(CATEGORIES)):
            for guidance in config.guidances:
                waveform=sample_unclipped(model,schedule,noise,label,guidance=guidance,
                                          reverse_steps=config.reverse_steps)
                raw_hash=hashlib.sha256(np.asarray(waveform,dtype="<f4").tobytes()).hexdigest()
                if guidance==0:
                    if null_hash is None:
                        null_hash=raw_hash
                    elif null_hash!=raw_hash:
                        raise ArchiveError("Null guidance unexpectedly depends on category")
                rows.append({"seed":seed,"category":CATEGORIES[label],"guidance":float(guidance),
                             "waveform_sha256":raw_hash,"signal":signal_statistics(waveform),
                             "reference":classify(reference,spectral_features(waveform),label)})
        null_outputs.append(null_hash)
    if model_fingerprint(model)!=before:
        raise ArchiveError("Model changed during generated waveform assay")
    # Paired per-seed comparison to null on the same requested category.
    controls={(row["seed"],row["category"]):row for row in rows if row["guidance"]==0}
    by_guidance={}
    for guidance in config.guidances:
        subset=[row for row in rows if row["guidance"]==guidance]
        differences=[]
        for row in subset:
            current=row["reference"]["nearest_wrong_minus_target"]
            null=controls[(row["seed"],row["category"])]["reference"]["nearest_wrong_minus_target"]
            if current is not None and null is not None:
                differences.append(current-null)
        by_guidance[str(float(guidance))]={
            **_aggregate(subset),
            "paired_margin_delta_vs_null":{"defined":len(differences),
                "mean":float(np.mean(differences)) if differences else None},
            "by_category":{category:_aggregate([row for row in subset if row["category"]==category])
                           for category in CATEGORIES},
        }
    return {
        "protocol":PROTOCOL,"model_sha256":before,"reference_sha256":reference.fingerprint,
        "reference_validation":reference_validation,
        "diffusion_profile":schedule.profile,"diffusion_steps":schedule.steps,
        "reverse_indices":list(indices),"seeds":list(config.seeds),
        "guidances":[float(g) for g in config.guidances],
        "initial_noise_trace_sha256":trace.hexdigest(),"null_output_sha256_by_seed":null_outputs,
        "forward_calls":config.forward_calls,"cases":rows,"by_guidance":by_guidance,
        "terminal_alpha_bar":float(schedule.alpha_bars[-1]),
        "sampling_protocol":"epsilon_ddim_eta_zero_unclipped",
        "initial_gaussian_matches_training_terminal_distribution_claimed":False,
        "model_state_unchanged":True,"test_split_scored":False,
        "generated_audio_semantics_established":False,"perceptual_quality_established":False,
        "model_publication_authorized":False,"files_written":0,
    }


def assay_checkpoint(archive,*,archive_sha256,checkpoint,checkpoint_sha256,
                     training:ArchiveTrainingConfig,config:GenerationConfig|None=None):
    require_sha256(archive_sha256)
    require_sha256(checkpoint_sha256)
    if not isinstance(training,ArchiveTrainingConfig):
        raise ArchiveError("Expected ArchiveTrainingConfig")
    with CorpusArchiveReader(archive,expected_sha256=archive_sha256) as reader:
        ready=audit_archive(reader)
        reference=fit_reference(reader,ready)
        validation=validation_reference_report(reader,ready,reference)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(training.seed)
            with torch.device("cpu"):
                model=TinyWaveDenoiser(training.channels)
                schedule=DiffusionSchedule(training.diffusion_steps,profile=training.diffusion_profile)
            optimizer=make_archive_optimizer(model)
            restored=restore_training_checkpoint(
                checkpoint,expected_sha256=checkpoint_sha256,config=training,readiness=ready,
                initial_model_sha256=model_fingerprint(model),model=model,optimizer=optimizer)
            report=evaluate_generated(model,schedule,reference,validation,config=config)
        reader.report()
    return {**report,"archive_sha256":archive_sha256,"checkpoint_sha256":restored.sha256,
            "completed_training_steps":restored.progress.completed_steps,"original_sonumio_1":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--archive-sha256",required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    defaults=ArchiveTrainingConfig()
    for field in ("seed","steps","batch_size","channels","diffusion_steps","validation_seed",
                  "validation_repeats","validation_batch_size"):
        parser.add_argument("--"+field.replace("_","-"),type=int,default=getattr(defaults,field))
    parser.add_argument("--diffusion-profile",choices=DiffusionSchedule.PROFILES,default=defaults.diffusion_profile)
    parser.add_argument("--reverse-steps",type=int,default=8)
    parser.add_argument("--generation-seeds",default="701,709,727")
    args=parser.parse_args(argv)
    try:
        from research.archive_study_plan import parse_seeds
        config=GenerationConfig(seeds=parse_seeds(args.generation_seeds),reverse_steps=args.reverse_steps)
        training=ArchiveTrainingConfig(**{key:getattr(args,key) for key in ArchiveTrainingConfig.__dataclass_fields__})
        result=assay_checkpoint(args.archive,archive_sha256=args.archive_sha256,
                               checkpoint=args.checkpoint,checkpoint_sha256=args.checkpoint_sha256,
                               training=training,config=config)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Generated signal assay: {exc}\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
