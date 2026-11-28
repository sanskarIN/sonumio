"""Conditioned differentiable percussion synthesis with independent noise.

A small network learns modal and noise-bank parameters, not memorized
waveforms. A feature objective compares temporal energy and frequency
distribution without requiring random source phase alignment.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from research.percussion_data import DRUMS,MIN_FRAMES,MAX_FRAMES,MAX_BATCH
from sonumio.audio import SAMPLE_RATE

PROTOCOL="sonumio-learned-percussion-v1"
WIDTHS=(32,48,64)
MODES=6
NOISE_BANDS=3
FEATURE_NAMES=("envelope","spectrum","band_envelope","centroid")
FREQUENCY_EDGES=(20,100,250,500,1000,2000,4000,8000,12001)
FFT=512
HOP=128
EPS=1e-8


def controls(labels,velocities,frames,excitation):
    if type(frames) is not int or not MIN_FRAMES<=frames<=MAX_FRAMES:
        raise ValueError("Invalid percussion frames")
    if (not isinstance(labels,torch.Tensor) or labels.device.type!="cpu" or labels.dtype!=torch.long
            or labels.ndim!=1 or not 1<=len(labels)<=MAX_BATCH
            or bool(((labels<0)|(labels>=len(DRUMS))).any())):
        raise ValueError("Invalid percussion labels")
    if (not isinstance(velocities,torch.Tensor) or velocities.device.type!="cpu"
            or velocities.dtype!=torch.long or velocities.shape!=labels.shape
            or bool(((velocities<100)|(velocities>1000)).any())):
        raise ValueError("Invalid percussion velocities")
    if (not isinstance(excitation,torch.Tensor) or excitation.device.type!="cpu"
            or excitation.dtype!=torch.float32 or excitation.shape!=(len(labels),frames+2)
            or not bool(torch.isfinite(excitation).all()) or bool((excitation.abs()>1.).any())):
        raise ValueError("Invalid independent percussion excitation")
    return len(labels)


def _edge(frames,attack,release):
    t=torch.arange(frames,dtype=torch.float32)
    a=torch.sin(torch.clamp(t[None,:]/attack[:,None],0.,1.)*math.pi/2).square()
    r=torch.sin(torch.clamp((frames-1-t[None,:])/release[:,None],0.,1.)*math.pi/2).square()
    return a*r


def _noise_basis(excitation):
    a,b,c=excitation[:,:-2],excitation[:,1:-1],excitation[:,2:]
    return torch.stack(((a+2*b+c)/4.,(c-a)/2.,(c-2*b+a)/4.),dim=1)


class PercussionModel(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        if type(width) is not int or width not in WIDTHS:
            raise ValueError("Invalid percussion network width")
        self.width=width
        self.embedding=nn.Embedding(len(DRUMS),width)
        self.network=nn.Sequential(nn.Linear(width+2,2*width),nn.SiLU(),nn.LayerNorm(2*width),
                                   nn.Linear(2*width,2*width),nn.SiLU())
        self.gain=nn.Linear(2*width,MODES+NOISE_BANDS+1)
        self.modal=nn.Linear(2*width,MODES*3)
        self.noise_decay=nn.Linear(2*width,NOISE_BANDS)
        self.edge=nn.Linear(2*width,2)
        self.register_buffer("low_frequency",torch.tensor([35.,80.,150.,500.,1500.,4000.]))
        self.register_buffer("high_frequency",torch.tensor([160.,350.,900.,2500.,6000.,10500.]))

    def forward(self,labels,velocities,frames,excitation):
        count=controls(labels,velocities,frames,excitation)
        v=velocities.to(torch.float32)/1000.
        duration=torch.full((count,),math.log2(frames/MIN_FRAMES)/2.,dtype=torch.float32)
        hidden=self.network(torch.cat((self.embedding(labels),v[:,None],duration[:,None]),dim=1))
        gain=torch.softmax(self.gain(hidden),dim=1)
        modal=self.modal(hidden).reshape(count,3,MODES)
        frequency=self.low_frequency+(self.high_frequency-self.low_frequency)*torch.sigmoid(modal[:,0])
        decay=.005+.245*torch.sigmoid(modal[:,1])
        sweep=frequency*.8*torch.sigmoid(modal[:,2])
        nd=.003+.147*torch.sigmoid(self.noise_decay(hidden))
        edge=self.edge(hidden)
        attack=1.+127.*torch.sigmoid(edge[:,0])
        release=1.+511.*torch.sigmoid(edge[:,1])
        t=torch.arange(frames,dtype=torch.float32)/SAMPLE_RATE
        phase=2*math.pi*(frequency[:,:,None]*t+
                         sweep[:,:,None]*.022*(1-torch.exp(-t/.022)))
        modes=(torch.sin(phase)*torch.exp(-t/decay[:,:,None])*
               gain[:,:MODES,None]).sum(dim=1)
        noise=(_noise_basis(excitation)*torch.exp(-t/nd[:,:,None])*
               gain[:,MODES:MODES+NOISE_BANDS,None]).sum(dim=1)
        # Each bounded basis is weighted by a nonnegative simplex.
        output=(modes+noise)*_edge(frames,attack,release)*v[:,None]
        if output.shape!=(count,frames) or not bool(torch.isfinite(output).all()):
            raise ValueError("Invalid learned percussion waveform")
        return output


def initialize(width,seed):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid percussion initialization seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PercussionModel(width)


def broadband_control(labels,velocities,frames,excitation):
    n=controls(labels,velocities,frames,excitation)
    t=torch.arange(frames,dtype=torch.float32)/SAMPLE_RATE
    v=velocities.to(torch.float32)/1000.
    edge=_edge(frames,torch.full((n,),24.),torch.full((n,),360.))
    return .5*excitation[:,1:-1]*torch.exp(-t/.06)*edge*v[:,None]


def _waveforms(samples):
    if (not isinstance(samples,torch.Tensor) or samples.device.type!="cpu"
            or samples.dtype!=torch.float32 or samples.ndim!=2
            or not 1<=samples.shape[0]<=MAX_BATCH
            or not MIN_FRAMES<=samples.shape[1]<=MAX_FRAMES
            or not bool(torch.isfinite(samples).all())):
        raise ValueError("Invalid percussion feature waveform")
    return samples


def features(samples):
    """Absolute energy and time-frequency shape, not phase-sensitive samples."""
    x=_waveforms(samples)
    # Pool the squared waveform over the note clock, preserving duration.
    energy=F.adaptive_avg_pool1d(x[:,None,:].square(),16)[:,0,:]
    envelope=torch.log1p(1000.*energy)
    window=torch.hann_window(FFT,dtype=torch.float32)
    spectrum=torch.stft(x,n_fft=FFT,hop_length=HOP,win_length=FFT,window=window,
                        center=False,normalized=True,return_complex=True).abs().square()
    freqs=torch.arange(FFT//2+1,dtype=torch.float32)*SAMPLE_RATE/FFT
    bands=[]
    for lo,hi in zip(FREQUENCY_EDGES[:-1],FREQUENCY_EDGES[1:]):
        mask=(freqs>=lo)&(freqs<hi)
        bands.append(spectrum[:,mask,:].sum(dim=1))
    bands=torch.stack(bands,dim=1)
    band_envelope=torch.log1p(100.*F.adaptive_avg_pool1d(bands,8))
    spectral=torch.log1p(100.*bands.mean(dim=2))
    total=spectrum.sum(dim=(1,2)).clamp_min(EPS)
    centroid=(spectrum*freqs[None,:,None]).sum(dim=(1,2))/total/(SAMPLE_RATE/2)
    return {"envelope":envelope,"spectrum":spectral,
            "band_envelope":band_envelope,"centroid":centroid[:,None]}


def feature_errors(predicted,target):
    if predicted.shape!=target.shape:
        raise ValueError("Percussion feature geometry differs")
    actual=features(predicted)
    expected=features(target)
    result={}
    for key in FEATURE_NAMES:
        a=actual[key].flatten(1)
        b=expected[key].flatten(1)
        # Absolute compressed feature differences: small-energy tail bands
        # do not dominate through division by their individual energies.
        result[key]=(a-b).square().mean(dim=1)
    if any(not bool(torch.isfinite(v).all()) for v in result.values()):
        raise ValueError("Nonfinite percussion feature errors")
    return result


def objective(predicted,target):
    errors=feature_errors(predicted,target)
    return (errors["envelope"]+errors["spectrum"]+errors["band_envelope"]+
            2.*errors["centroid"]).mean()


@torch.no_grad()
def diagnostics(predicted,target):
    errors=feature_errors(predicted,target)
    errors["objective"]=errors["envelope"]+errors["spectrum"]+errors["band_envelope"]+2.*errors["centroid"]
    errors["peak"]=predicted.abs().amax(dim=1)
    errors["rms_ratio"]=torch.sqrt(predicted.square().mean(dim=1)/target.square().mean(dim=1).clamp_min(EPS))
    return errors
