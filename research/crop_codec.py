"""Crop-compatible bounded neural codec with optional uniform bottleneck quantization.

The same convolutional architecture supports 2048 and 8192 mono samples at
8 kHz. Every arm uses a tanh-bounded latent. Quantized training uses a
straight-through estimator; evaluation decodes real fixed-width code packets.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.crop_codec_packet import (
    CropCodecError,check_frames,check_bits,code_shape,signed_limit,pack_codes,
    unpack_codes,LATENT_CHANNELS,
)

MAX_BATCH=16


def _wave(value,frames,*,bounded=True):
    if (not isinstance(value,torch.Tensor) or value.device.type!="cpu"
            or value.dtype!=torch.float32 or value.ndim!=3
            or value.shape[1:]!=(1,frames) or not 1<=len(value)<=MAX_BATCH
            or not torch.isfinite(value).all()):
        raise CropCodecError("Expected bounded finite CPU float32 mono crop batch")
    if bounded and value.abs().max()>1.0001:
        raise CropCodecError("Crop waveform exceeds normalized PCM range")


def _latent(value,frames):
    if (not isinstance(value,torch.Tensor) or value.device.type!="cpu"
            or value.dtype!=torch.float32 or value.ndim!=3
            or value.shape[1:]!=code_shape(frames) or not 1<=len(value)<=MAX_BATCH
            or not torch.isfinite(value).all() or value.abs().max()>1):
        raise CropCodecError("Expected finite tanh-bounded crop latent")


def quantize_latent(latent,bits,*,straight_through=False):
    limit=signed_limit(bits)
    if type(straight_through) is not bool:
        raise CropCodecError("Expected boolean straight-through option")
    if (not isinstance(latent,torch.Tensor) or latent.device.type!="cpu"
            or latent.dtype!=torch.float32 or not torch.isfinite(latent).all()
            or not latent.numel() or latent.numel()>MAX_BATCH*LATENT_CHANNELS*128
            or latent.abs().max()>1):
        raise CropCodecError("Invalid bounded latent for quantization")
    codes=torch.round(latent*limit).to(torch.int16)
    restored=codes.to(torch.float32)/limit
    return (restored.detach()+(latent-latent.detach()) if straight_through else restored),codes


class CropAudioCodec(nn.Module):
    def __init__(self,frames=2048):
        super().__init__()
        self.frames=check_frames(frames)
        self.encoder=nn.Sequential(
            nn.Conv1d(1,8,7,padding=3),nn.SiLU(),
            nn.Conv1d(8,12,8,stride=4,padding=2),nn.SiLU(),
            nn.Conv1d(12,16,8,stride=4,padding=2),nn.SiLU(),
            nn.Conv1d(16,LATENT_CHANNELS,8,stride=4,padding=2),nn.Tanh(),
        )
        # Resize-convolution avoids changing temporal alignment by hidden
        # output-padding choices. Each stage has an explicit factor of four.
        self.decoder=nn.Sequential(
            nn.Upsample(scale_factor=4,mode="nearest"),nn.Conv1d(LATENT_CHANNELS,16,7,padding=3),nn.SiLU(),
            nn.Upsample(scale_factor=4,mode="nearest"),nn.Conv1d(16,12,7,padding=3),nn.SiLU(),
            nn.Upsample(scale_factor=4,mode="nearest"),nn.Conv1d(12,8,7,padding=3),nn.SiLU(),
            nn.Conv1d(8,1,7,padding=3),
        )

    def encode(self,waveform):
        _wave(waveform,self.frames)
        result=self.encoder(waveform)
        _latent(result,self.frames)
        return result

    def decode_with_preactivation(self,latent):
        _latent(latent,self.frames)
        pre=self.decoder(latent)
        _wave(pre,self.frames,bounded=False)
        return pre,pre.tanh()

    def decode(self,latent):
        return self.decode_with_preactivation(latent)[1]

    def forward(self,waveform,bits=None):
        latent=self.encode(waveform)
        if bits is not None:
            latent,_=quantize_latent(latent,bits,straight_through=self.training)
        return self.decode(latent)

    @torch.no_grad()
    def encode_packets(self,waveform,bits):
        check_bits(bits)
        latent=self.encode(waveform)
        _,codes=quantize_latent(latent,bits)
        return tuple(pack_codes(code.numpy(),frames=self.frames,bits=bits) for code in codes)

    @torch.no_grad()
    def decode_packets(self,packets):
        if not isinstance(packets,(list,tuple)) or not 1<=len(packets)<=MAX_BATCH:
            raise CropCodecError("Invalid bounded latent packet batch")
        decoded=[unpack_codes(packet) for packet in packets]
        if any(item.frames!=self.frames or item.bits!=decoded[0].bits for item in decoded):
            raise CropCodecError("Latent packet batch geometry mismatch")
        codes=torch.from_numpy(np.stack([item.codes for item in decoded]))
        latent=codes.to(torch.float32)/signed_limit(decoded[0].bits)
        return self.decode_with_preactivation(latent)


def reconstruction_metrics(source,restored,*,frames):
    """Per-example differentiable waveform/spectral errors, no perception claim."""
    check_frames(frames)
    _wave(source,frames)
    _wave(restored,frames)
    if source.shape!=restored.shape:
        raise CropCodecError("Crop reconstruction batch geometry mismatch")
    a,b=source[:,0],restored[:,0]
    difference=a-b
    l1=difference.abs().mean(dim=1)
    mse=difference.square().mean(dim=1)
    preemphasis=((a[:,1:]-.95*a[:,:-1])-(b[:,1:]-.95*b[:,:-1])).abs().mean(dim=1)
    spectral=[]
    convergence=[]
    for fft in (128,256,512):
        window=torch.hann_window(fft,dtype=torch.float32)
        x=torch.stft(a,fft,hop_length=fft//4,window=window,return_complex=True).abs()
        y=torch.stft(b,fft,hop_length=fft//4,window=window,return_complex=True).abs()
        spectral.append((torch.log1p(x)-torch.log1p(y)).abs().mean((1,2)))
        convergence.append(torch.linalg.vector_norm((x-y).flatten(1),dim=1)/
                           torch.linalg.vector_norm(x.flatten(1),dim=1).clamp_min(1e-6))
    envelope=[]
    for width in (32,128,512):
        x=(a.reshape(len(a),-1,width).square().mean(dim=2)+1e-12).sqrt()
        y=(b.reshape(len(b),-1,width).square().mean(dim=2)+1e-12).sqrt()
        envelope.append((x-y).abs().mean(dim=1))
    snr=10*torch.log10((a.double().square().mean(dim=1)+1e-10)/
                        (difference.double().square().mean(dim=1)+1e-10))
    metrics={
        "waveform_l1":l1,"waveform_mse":mse,"preemphasis_l1":preemphasis,
        "log_magnitude_l1":torch.stack(spectral).mean(dim=0),
        "spectral_convergence":torch.stack(convergence).mean(dim=0),
        "envelope_l1":torch.stack(envelope).mean(dim=0),"snr_db":snr,
    }
    if any(not torch.isfinite(x).all() for x in metrics.values()):
        raise CropCodecError("Nonfinite crop reconstruction metric")
    return metrics


def codec_loss(source,restored,*,frames):
    metrics=reconstruction_metrics(source,restored,frames=frames)
    # Explicit objective is shared between all precision arms. SNR and
    # raw MSE remain evaluation-only metrics.
    result=(metrics["waveform_l1"]+.2*metrics["preemphasis_l1"]
            +.5*metrics["log_magnitude_l1"]+.15*metrics["spectral_convergence"]
            +.25*metrics["envelope_l1"]).mean()
    if not torch.isfinite(result):
        raise CropCodecError("Nonfinite crop reconstruction loss")
    return result
