"""Train-only crop latent calibration and original class-conditioned v-denoiser.

The codec and denoiser are independent stages. Latent values are standardized
using frozen training codec outputs only. The decoder's bounded input is
explicitly restored, clamped and optionally requantized with diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn

from research.crop_codec import CropAudioCodec,quantize_latent
from research.crop_codec_packet import CropCodecError,check_frames,check_bits,code_shape,LATENT_CHANNELS
from research.wave_diffusion import ConditionedResidual,CATEGORIES,DiffusionSchedule
from research.velocity_prediction import velocity_target,decode_velocity

MAX_BATCH=16
MAX_CALIBRATION_CLIPS=4096
MIN_STD=.01
MAX_STANDARDIZED=10_000.
PROTOCOL="sonumio-crop-latent-calibration-v1"


def checked_latent(value,frames,*,native=False):
    if (not isinstance(value,torch.Tensor) or value.device.type!="cpu"
            or value.dtype!=torch.float32 or value.ndim!=3
            or value.shape[1:]!=code_shape(frames) or not 1<=len(value)<=MAX_BATCH
            or not torch.isfinite(value).all()):
        raise CropCodecError("Invalid finite CPU crop latent batch")
    if value.abs().max()>(1 if native else MAX_STANDARDIZED):
        raise CropCodecError("Crop latent exceeds numerical range")


@dataclass(frozen=True,slots=True)
class CropLatentCalibration:
    frames:int
    mean:tuple[float,...]
    std:tuple[float,...]
    clips:int

    def __post_init__(self):
        check_frames(self.frames)
        if (type(self.clips) is not int or not 2<=self.clips<=MAX_CALIBRATION_CLIPS
                or type(self.mean) is not tuple or type(self.std) is not tuple
                or len(self.mean)!=LATENT_CHANNELS or len(self.std)!=LATENT_CHANNELS):
            raise CropCodecError("Invalid crop calibration geometry")
        if (any(type(x) not in (float,int) or not math.isfinite(x) or abs(x)>1 for x in self.mean)
                or any(type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1.000001 for x in self.std)):
            raise CropCodecError("Invalid bounded native latent moments")

    @property
    def scale(self):
        return tuple(max(MIN_STD,float(x)) for x in self.std)

    def fingerprint(self):
        hasher=hashlib.sha256(b"sonumio-crop-latent-calibration-v1\0")
        hasher.update(self.frames.to_bytes(4,"little"))
        hasher.update(self.clips.to_bytes(4,"little"))
        for row in (self.mean,self.std,self.scale):
            hasher.update(np.asarray(row,dtype="<f8").tobytes())
        return hasher.hexdigest()

    def _parameters(self):
        return (torch.tensor(self.mean,dtype=torch.float32).view(1,-1,1),
                torch.tensor(self.scale,dtype=torch.float32).view(1,-1,1))

    def normalize(self,native):
        checked_latent(native,self.frames,native=True)
        mean,scale=self._parameters()
        result=(native-mean)/scale
        checked_latent(result,self.frames)
        return result

    def restore(self,standardized):
        checked_latent(standardized,self.frames)
        mean,scale=self._parameters()
        result=standardized*scale+mean
        if not torch.isfinite(result).all():
            raise CropCodecError("Nonfinite restored crop latent")
        return result

    def report(self):
        return {
            "protocol":PROTOCOL,"frames":self.frames,"training_clips":self.clips,
            "channel_mean":list(self.mean),"channel_std":list(self.std),
            "channel_scale":list(self.scale),"std_floor":MIN_STD,
            "floored_channels":sum(x<MIN_STD for x in self.std),
            "calibration_sha256":self.fingerprint(),
        }


class LatentMomentAccumulator:
    """Bounded per-channel moments with stable, mergeable double reductions."""
    def __init__(self,frames):
        self.frames=check_frames(frames)
        self.clips=0
        self.count=0
        self.mean=np.zeros(LATENT_CHANNELS,dtype=np.float64)
        self.m2=np.zeros_like(self.mean)

    def add(self,latent):
        checked_latent(latent,self.frames,native=True)
        if latent.requires_grad:
            raise CropCodecError("Calibration requires detached frozen latent values")
        if self.clips+len(latent)>MAX_CALIBRATION_CLIPS:
            raise CropCodecError("Crop calibration exceeds clip budget")
        values=latent.numpy().astype(np.float64).transpose(1,0,2).reshape(LATENT_CHANNELS,-1)
        n=values.shape[1]
        mean=values.mean(axis=1)
        m2=((values-mean[:,None])**2).sum(axis=1)
        total=self.count+n
        delta=mean-self.mean
        self.m2+=m2+delta**2*(self.count*n/total)
        self.mean+=delta*(n/total)
        self.count=total
        self.clips+=len(latent)

    def finish(self):
        if self.clips<2:
            raise CropCodecError("Calibration requires at least two clips")
        std=np.sqrt(np.maximum(self.m2/self.count,0))
        return CropLatentCalibration(self.frames,tuple(self.mean.tolist()),tuple(std.tolist()),self.clips)


@torch.no_grad()
def encode_native(codec,wave,bits):
    if not isinstance(codec,CropAudioCodec):
        raise CropCodecError("Expected crop codec")
    latent=codec.encode(wave)
    if bits is not None:
        latent,_=quantize_latent(latent,bits)
    return latent.detach()


@torch.no_grad()
def fit_calibration(codec,batches,*,bits=None):
    if bits is not None:
        check_bits(bits)
    if not isinstance(codec,CropAudioCodec) or codec.training or any(p.requires_grad for p in codec.parameters()):
        raise CropCodecError("Calibration requires frozen evaluation-mode crop codec")
    moments=LatentMomentAccumulator(codec.frames)
    for batch in batches:
        moments.add(encode_native(codec,batch,bits))
    return moments.finish()


@dataclass(frozen=True,slots=True,eq=False)
class RestoredLatent:
    latent:torch.Tensor
    diagnostics:dict


def restore_for_decoder(standardized,calibration,*,bits=None):
    """Report overflow in native latent units before the explicit projection."""
    if not isinstance(calibration,CropLatentCalibration):
        raise CropCodecError("Expected training-fitted crop calibration")
    native=calibration.restore(standardized)
    clipped=native.clamp(-1,1)
    decoded=clipped
    if bits is not None:
        decoded,_=quantize_latent(clipped,bits)
    return RestoredLatent(decoded,{
        "native_peak_before_projection":float(native.abs().max()),
        "native_outside_bound_fraction":float((native.abs()>1).float().mean()),
        "clamp_mse":float((clipped-native).double().square().mean()),
        "requantization_mse":float((decoded-clipped).double().square().mean()),
        "quantization_bits":bits,
        "explicit_decoder_bound_projection":True,
    })


class CropLatentDenoiser(nn.Module):
    """Original compact class/time-conditioned 1D velocity denoiser."""
    def __init__(self,frames=2048,width=32):
        super().__init__()
        self.frames=check_frames(frames)
        if type(width) is not int or width not in (16,32,48,64):
            raise CropCodecError("Invalid crop latent denoiser width")
        self.width=width
        self.classes=len(CATEGORIES)
        self.embedding=nn.Embedding(self.classes+1,width)
        self.time_mlp=nn.Sequential(nn.Linear(width,width*2),nn.SiLU(),nn.Linear(width*2,width))
        self.input=nn.Conv1d(LATENT_CHANNELS,width,3,padding=1)
        self.residuals=nn.ModuleList(ConditionedResidual(width,d) for d in (1,2,4,8))
        self.output=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,LATENT_CHANNELS,3,padding=1))

    def forward(self,noisy,timestep,labels):
        checked_latent(noisy,self.frames)
        if (not isinstance(timestep,torch.Tensor) or timestep.dtype!=torch.long
                or timestep.device.type!="cpu" or timestep.shape!=(len(noisy),)
                or torch.any(timestep<0) or torch.any(timestep>=1000)):
            raise CropCodecError("Invalid crop diffusion timestep")
        if (not isinstance(labels,torch.Tensor) or labels.dtype!=torch.long
                or labels.device.type!="cpu" or labels.shape!=(len(noisy),)
                or torch.any(labels<0) or torch.any(labels>self.classes)):
            raise CropCodecError("Invalid crop latent category")
        half=self.width//2
        frequency=torch.exp(-math.log(10_000)*torch.arange(half,dtype=torch.float32)/max(1,half-1))
        angles=timestep.float()[:,None]*frequency[None,:]
        condition=self.embedding(labels)+self.time_mlp(torch.cat((angles.sin(),angles.cos()),dim=1))
        value=self.input(noisy)
        for layer in self.residuals:
            value=layer(value,condition)
        result=self.output(value)
        if result.shape!=noisy.shape or not torch.isfinite(result).all():
            raise CropCodecError("Nonfinite crop velocity prediction")
        return result


def velocity_loss(model,schedule,clean,timesteps,noise,labels):
    if not isinstance(model,CropLatentDenoiser) or not isinstance(schedule,DiffusionSchedule):
        raise CropCodecError("Expected crop velocity denoiser and schedule")
    checked_latent(clean,model.frames)
    checked_latent(noise,model.frames)
    target=velocity_target(clean,noise,timesteps,schedule)
    predicted=model(schedule.corrupt(clean,timesteps,noise),timesteps,labels)
    loss=(target-predicted).square().mean()
    if not torch.isfinite(loss):
        raise CropCodecError("Nonfinite crop velocity objective")
    return loss


def reverse_indices(steps,budget):
    if (type(steps) is not int or not 8<=steps<=1000 or type(budget) is not int
            or not 2<=budget<=min(64,steps)):
        raise CropCodecError("Invalid crop latent reverse budget")
    return tuple((steps-1)*(budget-1-index)//(budget-1) for index in range(budget))


@torch.no_grad()
def sample_latent(model,schedule,noise,category,*,guidance=1.,budget=8):
    """Deterministic velocity-DDIM eta=0; no epsilon division near terminal."""
    if not isinstance(model,CropLatentDenoiser) or not isinstance(schedule,DiffusionSchedule):
        raise CropCodecError("Expected crop latent model and schedule")
    checked_latent(noise,model.frames)
    if len(noise)!=1:
        raise CropCodecError("Sampling requires one fixed latent noise tensor")
    if (type(category) is not int or not 0<=category<model.classes
            or type(guidance) not in (int,float) or not math.isfinite(guidance)
            or not 0<=guidance<=4):
        raise CropCodecError("Invalid crop latent generation condition")
    indices=reverse_indices(schedule.steps,budget)
    current=noise.clone()
    mode=model.training
    model.eval()
    try:
        for i,index in enumerate(indices):
            timestep=torch.tensor([index],dtype=torch.long)
            if guidance==0:
                prediction=model(current,timestep,torch.tensor([model.classes]))
            elif guidance==1:
                prediction=model(current,timestep,torch.tensor([category]))
            else:
                null=model(current,timestep,torch.tensor([model.classes]))
                matched=model(current,timestep,torch.tensor([category]))
                prediction=null+guidance*(matched-null)
            clean,eps=decode_velocity(current,prediction,timestep,schedule)
            if i+1<len(indices):
                previous=schedule.alpha_bars[indices[i+1]]
                current=previous.sqrt()*clean+(1-previous).sqrt()*eps
            else:
                current=clean
            checked_latent(current,model.frames)
    finally:
        model.train(mode)
    return current
