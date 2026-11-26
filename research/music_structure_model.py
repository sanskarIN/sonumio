"""Causal bar-structure network and free-running joint sampling.

The model learns p(chord | past, context) and
p(rhythm | chord, past, context). It does not learn the final-cadence
constraint; that is an explicit optional generation-time mask.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from research.music_structure_data import (
    BarSequence, CHORD_COUNT, CHORD_BOS, RHYTHM_COUNT, RHYTHM_BOS,
    CONTEXT_SIZES, MAX_BARS, allowed_chords, allowed_rhythms,
)

CHORD_PAD=CHORD_BOS+1
RHYTHM_PAD=RHYTHM_BOS+1
WIDTHS=(32,48,64)
MAX_BATCH=32


def collate(sequences):
    if (type(sequences) not in (list,tuple) or not 1<=len(sequences)<=MAX_BATCH
            or any(not isinstance(s,BarSequence) for s in sequences)):
        raise ValueError("Invalid bar batch")
    length=max(len(s.chords) for s in sequences)
    shape=(len(sequences),length)
    past_chord=torch.full(shape,CHORD_PAD,dtype=torch.long)
    past_rhythm=torch.full(shape,RHYTHM_PAD,dtype=torch.long)
    target_chord=torch.full(shape,-100,dtype=torch.long)
    target_rhythm=torch.full(shape,-100,dtype=torch.long)
    context=torch.zeros((*shape,len(CONTEXT_SIZES)),dtype=torch.long)
    for i,seq in enumerate(sequences):
        n=len(seq.chords)
        past_chord[i,:n]=torch.tensor((CHORD_BOS,*seq.chords[:-1]))
        past_rhythm[i,:n]=torch.tensor((RHYTHM_BOS,*seq.rhythms[:-1]))
        target_chord[i,:n]=torch.tensor(seq.chords)
        target_rhythm[i,:n]=torch.tensor(seq.rhythms)
        context[i,:n]=torch.tensor(seq.contexts)
    return past_chord,past_rhythm,context,target_chord,target_rhythm


def _index(tensor,shape,maximum,name):
    if (not isinstance(tensor,torch.Tensor) or tensor.device.type!="cpu"
            or tensor.dtype!=torch.long or tensor.shape!=shape
            or (tensor<0).any() or (tensor>maximum).any()):
        raise ValueError("Invalid "+name+" tensor")


class StructureGRU(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        if type(width) is not int or width not in WIDTHS:
            raise ValueError("Unsupported structure model width")
        self.width=width
        self.chord_history=nn.Embedding(CHORD_PAD+1,width,padding_idx=CHORD_PAD)
        self.rhythm_history=nn.Embedding(RHYTHM_PAD+1,width,padding_idx=RHYTHM_PAD)
        self.context=nn.ModuleList(nn.Embedding(n,width) for n in CONTEXT_SIZES)
        self.input_norm=nn.LayerNorm(width)
        self.gru=nn.GRU(width,width,batch_first=True)
        self.chord_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,CHORD_COUNT))
        self.current_chord=nn.Embedding(CHORD_COUNT,width)
        self.rhythm_head=nn.Sequential(nn.LayerNorm(width*2),nn.Linear(width*2,width),
                                       nn.GELU(),nn.Linear(width,RHYTHM_COUNT))

    def encode(self,past_chord,past_rhythm,context,hidden=None):
        if (not isinstance(past_chord,torch.Tensor) or past_chord.ndim!=2
                or not 1<=past_chord.shape[0]<=MAX_BATCH
                or not 1<=past_chord.shape[1]<=MAX_BARS):
            raise ValueError("Invalid bar history geometry")
        shape=past_chord.shape
        _index(past_chord,shape,CHORD_PAD,"chord history")
        _index(past_rhythm,shape,RHYTHM_PAD,"rhythm history")
        if (not isinstance(context,torch.Tensor) or context.device.type!="cpu"
                or context.dtype!=torch.long or context.shape!=(*shape,len(CONTEXT_SIZES))):
            raise ValueError("Invalid bar conditioning geometry")
        if hidden is not None and (
                not isinstance(hidden,torch.Tensor) or hidden.device.type!="cpu"
                or hidden.dtype!=torch.float32 or hidden.shape!=(1,shape[0],self.width)
                or not torch.isfinite(hidden).all()):
            raise ValueError("Invalid bar recurrent state")
        x=self.chord_history(past_chord)+self.rhythm_history(past_rhythm)
        for i,(limit,embedding) in enumerate(zip(CONTEXT_SIZES,self.context,strict=True)):
            tokens=context[:,:,i]
            _index(tokens,shape,limit-1,"bar context")
            x=x+embedding(tokens)
        state,next_hidden=self.gru(self.input_norm(x),hidden)
        chord=self.chord_head(state)
        if not torch.isfinite(chord).all() or not torch.isfinite(next_hidden).all():
            raise ValueError("Nonfinite structure model state")
        return state,chord,next_hidden

    def rhythm_logits(self,state,chord):
        if (not isinstance(state,torch.Tensor) or state.ndim!=3
                or state.shape[-1]!=self.width or state.device.type!="cpu"
                or state.dtype!=torch.float32 or not torch.isfinite(state).all()):
            raise ValueError("Invalid bar hidden state")
        _index(chord,state.shape[:2],CHORD_COUNT-1,"current chord")
        out=self.rhythm_head(torch.cat((state,self.current_chord(chord)),dim=-1))
        if not torch.isfinite(out).all():
            raise ValueError("Nonfinite rhythm logits")
        return out

    def forward(self,past_chord,past_rhythm,context,current_chord):
        state,chord,hidden=self.encode(past_chord,past_rhythm,context)
        rhythm=self.rhythm_logits(state,current_chord)
        return chord,rhythm,hidden


def _loss(logits,targets,count):
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=3
            or logits.shape[-1]!=count or logits.device.type!="cpu"
            or logits.dtype!=torch.float32 or not torch.isfinite(logits).all()
            or not isinstance(targets,torch.Tensor) or targets.dtype!=torch.long
            or targets.device.type!="cpu" or targets.shape!=logits.shape[:2]):
        raise ValueError("Invalid structure loss inputs")
    keep=targets!=-100
    if not keep.any() or ((targets[keep]<0)|(targets[keep]>=count)).any():
        raise ValueError("Invalid structure target")
    value=F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-100)
    if not torch.isfinite(value):
        raise ValueError("Nonfinite structure loss")
    return value


def joint_loss(chord,rhythm,target_chord,target_rhythm):
    if not torch.equal(target_chord==-100,target_rhythm==-100):
        raise ValueError("Inconsistent structure target padding")
    c=_loss(chord,target_chord,CHORD_COUNT)
    r=_loss(rhythm,target_rhythm,RHYTHM_COUNT)
    return c+r,c,r


def _sample(logits,generator,temperature,allowed=None):
    logits=logits/temperature
    if allowed is not None:
        mask=torch.full_like(logits,float("-inf"))
        mask[list(allowed)]=0.
        logits=logits+mask
    probability=torch.softmax(logits,dim=-1)
    if not torch.isfinite(probability).all():
        raise ValueError("Invalid structure sampling probabilities")
    return int(torch.multinomial(probability,1,generator=generator))


@torch.no_grad()
def sample_structure(model,contexts,seed,*,temperature=1.,constrained=True):
    """Accept context only; no reference chord or rhythm tokens are read."""
    if not isinstance(model,StructureGRU):
        raise ValueError("Expected structure model")
    if type(contexts) is not tuple or not 4<=len(contexts)<=MAX_BARS:
        raise ValueError("Invalid sampling context panel")
    # Validate structure before any random draw.
    BarSequence((0,)*len(contexts),(1,)*len(contexts),contexts)
    if (type(seed) is not int or not 0<=seed<2**32
            or type(temperature) not in (float,int) or not math.isfinite(temperature)
            or not .25<=temperature<=2 or type(constrained) is not bool):
        raise ValueError("Invalid structure sampling configuration")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    mode=model.training
    model.eval()
    chords=[]
    rhythms=[]
    previous_chord=CHORD_BOS
    previous_rhythm=RHYTHM_BOS
    hidden=None
    try:
        for i,context in enumerate(contexts):
            state,logits,hidden=model.encode(
                torch.tensor([[previous_chord]]),torch.tensor([[previous_rhythm]]),
                torch.tensor([[context]]),hidden)
            chord=_sample(logits[0,0],generator,temperature,
                          allowed_chords(i,len(contexts)) if constrained else None)
            rlogits=model.rhythm_logits(state,torch.tensor([[chord]]))
            rhythm=_sample(rlogits[0,0],generator,temperature,
                           allowed_rhythms(context) if constrained else None)
            chords.append(chord)
            rhythms.append(rhythm)
            previous_chord,previous_rhythm=chord,rhythm
    finally:
        model.train(mode)
    return BarSequence(tuple(chords),tuple(rhythms),contexts)
