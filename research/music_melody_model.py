"""Bounded CPU autoregressive relative-pitch melody model.

The network predicts the next pitch from previous pitch and supplied
section, chord, rhythm, style and mode. It does not synthesize audio.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from research.music_melody_data import (
    MelodySequence, PITCH_COUNT, BOS, PAD, CONTEXT_SIZES, MAX_SEQUENCE,
    valid_tokens,
)

MAX_BATCH=32
WIDTHS=(32,48,64)


class MelodyGRU(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        if type(width) is not int or width not in WIDTHS:
            raise ValueError("Unsupported melody GRU width")
        self.width=width
        self.previous=nn.Embedding(PAD+1,width,padding_idx=PAD)
        self.context=nn.ModuleList(nn.Embedding(n,width) for n in CONTEXT_SIZES)
        self.input_norm=nn.LayerNorm(width)
        self.recurrent=nn.GRU(width,width,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,PITCH_COUNT))

    def forward(self,previous,context,hidden=None):
        if (not isinstance(previous,torch.Tensor) or previous.device.type!="cpu"
                or previous.dtype!=torch.long or previous.ndim!=2
                or not 1<=previous.shape[0]<=MAX_BATCH
                or not 1<=previous.shape[1]<=MAX_SEQUENCE
                or (previous<0).any() or (previous>PAD).any()):
            raise ValueError("Invalid melody previous-token batch")
        if (not isinstance(context,torch.Tensor) or context.device.type!="cpu"
                or context.dtype!=torch.long
                or context.shape!=(*previous.shape,len(CONTEXT_SIZES))):
            raise ValueError("Invalid melody context batch")
        if hidden is not None and (not isinstance(hidden,torch.Tensor)
                or hidden.shape!=(1,len(previous),self.width)
                or hidden.dtype!=torch.float32 or hidden.device.type!="cpu"
                or not torch.isfinite(hidden).all()):
            raise ValueError("Invalid melody recurrent state")
        h=self.previous(previous)
        for i,(n,embedding) in enumerate(zip(CONTEXT_SIZES,self.context,strict=True)):
            values=context[:,:,i]
            if (values<0).any() or (values>=n).any():
                raise ValueError("Melody context index out of bounds")
            h=h+embedding(values)
        y,state=self.recurrent(self.input_norm(h),hidden)
        logits=self.output(y)
        if not torch.isfinite(logits).all() or not torch.isfinite(state).all():
            raise ValueError("Nonfinite melody model output")
        return logits,state


def token_loss(logits,targets):
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=3
            or logits.shape[-1]!=PITCH_COUNT or logits.dtype!=torch.float32
            or logits.device.type!="cpu" or not torch.isfinite(logits).all()
            or not isinstance(targets,torch.Tensor) or targets.dtype!=torch.long
            or targets.device.type!="cpu" or targets.shape!=logits.shape[:2]):
        raise ValueError("Invalid melody loss inputs")
    keep=targets!=-100
    if not keep.any() or ((targets[keep]<0)|(targets[keep]>=PITCH_COUNT)).any():
        raise ValueError("Invalid melody target tokens")
    result=F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-100)
    if not torch.isfinite(result):
        raise ValueError("Nonfinite melody cross entropy")
    return result


@torch.no_grad()
def generate_tokens(model,sequence,seed,*,temperature=1.,constrained=True):
    """Free-running pitch prediction, never feeds held-out true pitches."""
    if not isinstance(model,MelodyGRU) or not isinstance(sequence,MelodySequence):
        raise ValueError("Expected melody model and supplied schedule")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid melody sampling seed")
    if (type(temperature) not in (int,float) or not math.isfinite(temperature)
            or not .25<=temperature<=2 or type(constrained) is not bool):
        raise ValueError("Invalid melody sampling settings")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    mode=model.training
    model.eval()
    hidden=None
    previous=BOS
    result=[]
    try:
        for index,context in enumerate(sequence.contexts):
            x=torch.tensor([[previous]],dtype=torch.long)
            c=torch.tensor([[context]],dtype=torch.long)
            logits,hidden=model(x,c,hidden)
            scores=logits[0,0]/temperature
            if constrained:
                allowed=valid_tokens(context,final=index==len(sequence.contexts)-1)
                mask=torch.full_like(scores,float("-inf"))
                mask[list(allowed)]=0
                scores=scores+mask
            probability=torch.softmax(scores,dim=0)
            if not torch.isfinite(probability).all():
                raise ValueError("Invalid melody sampling probability")
            previous=int(torch.multinomial(probability,1,generator=generator))
            result.append(previous)
    finally:
        model.train(mode)
    return tuple(result)
