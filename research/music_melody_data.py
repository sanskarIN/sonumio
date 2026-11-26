"""Original synthetic melody sequences and conditioning contracts.

A learned melody generator predicts relative pitch tokens only. Harmony,
section, rhythm and backing instruments are supplied by the procedural
arrangement. This is not end-to-end text-to-music or waveform modeling.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import numpy as np
import torch

from sonumio.arrangement import (
    Arrangement, ArrangementPlan, compose_arrangement, event_key, SECTIONS,
    chord_classes,
)
from sonumio.music import STYLES, MODES
from sonumio.errors import GenerationError

PROTOCOL="sonumio-relative-melody-sequences-v1"
PITCH_COUNT=25
BOS=25
PAD=26
CONTEXT_SIZES=(len(STYLES),len(MODES),7,32,len(SECTIONS))
MAX_SEQUENCE=192
MAX_SOURCES=192


@dataclass(frozen=True,slots=True)
class MelodySequence:
    pitches:tuple[int,...]
    contexts:tuple[tuple[int,...],...]

    def __post_init__(self):
        if (type(self.pitches) is not tuple or not 1<=len(self.pitches)<=MAX_SEQUENCE
                or any(type(p) is not int or not 0<=p<PITCH_COUNT for p in self.pitches)):
            raise ValueError("Invalid relative melody pitches")
        if type(self.contexts) is not tuple or len(self.contexts)!=len(self.pitches):
            raise ValueError("Invalid melody context length")
        for context in self.contexts:
            if (type(context) is not tuple or len(context)!=len(CONTEXT_SIZES)
                    or any(type(v) is not int or not 0<=v<n
                           for v,n in zip(context,CONTEXT_SIZES,strict=True))):
                raise ValueError("Invalid melody conditioning token")


def extract(score):
    if not isinstance(score,Arrangement):
        raise ValueError("Expected original arrangement")
    section_by_bar=[s.kind for s in score.sections for _ in range(s.bars)]
    lead=[e for e in score.events if e.track=="lead"]
    style=STYLES.index(score.plan.style)
    mode=tuple(MODES).index(score.plan.mode)
    pitches=tuple(e.pitch-score.plan.root for e in lead)
    contexts=tuple((style,mode,score.harmony[e.tick//32],e.tick%32,
                    SECTIONS.index(section_by_bar[e.tick//32])) for e in lead)
    return MelodySequence(pitches,contexts)


def fingerprint(sequence,*,pitches_only=False):
    if not isinstance(sequence,MelodySequence):
        raise ValueError("Expected melody sequence")
    value=sequence.pitches if pitches_only else (sequence.pitches,sequence.contexts)
    payload=json.dumps(value,separators=(",",":")).encode("ascii")
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+payload).hexdigest()


def valid_tokens(context,*,final=False):
    if (type(context) is not tuple or len(context)!=len(CONTEXT_SIZES)
            or any(type(v) is not int or not 0<=v<n
                   for v,n in zip(context,CONTEXT_SIZES,strict=True))):
        raise ValueError("Invalid token mask context")
    _,mode,degree,tick,_=context
    scale=MODES[tuple(MODES)[mode]]
    if final:
        classes={0}
    elif tick%8==0:
        classes={scale[(degree+i)%7] for i in (0,2,4)}
    else:
        classes=set(scale)
    return tuple(p for p in range(PITCH_COUNT) if p%12 in classes)


def replace_melody(score,pitches):
    sequence=extract(score)
    if type(pitches) is not tuple or len(pitches)!=len(sequence.pitches):
        raise ValueError("Generated melody length differs from score schedule")
    for i,(token,context) in enumerate(zip(pitches,sequence.contexts,strict=True)):
        if type(token) is not int or token not in valid_tokens(context,final=i==len(pitches)-1):
            raise ValueError("Generated token violates score constraint")
    iterator=iter(pitches)
    events=tuple(replace(e,pitch=score.plan.root+next(iterator)) if e.track=="lead" else e
                 for e in score.events)
    return replace(score,events=tuple(sorted(events,key=event_key)))


def _seed_panel(seeds):
    if (type(seeds) is not tuple or not 1<=len(seeds)<=32
            or any(type(v) is not int or not 0<=v<2**32 for v in seeds)
            or len(set(seeds))!=len(seeds)):
        raise ValueError("Invalid melody source seed panel")


def corpus(seeds,*,bars=8):
    _seed_panel(seeds)
    if type(bars) is not int or not 4<=bars<=16:
        raise ValueError("Invalid melody corpus bar count")
    rows=[]
    trace=hashlib.sha256(PROTOCOL.encode()+b"\0")
    for seed in seeds:
        for style in STYLES:
            for mode in MODES:
                score=compose_arrangement(ArrangementPlan(style=style,root=60,mode=mode,
                                                         bpm=120,bars=bars,seed=seed))
                sequence=extract(score)
                for i,(p,c) in enumerate(zip(sequence.pitches,sequence.contexts,strict=True)):
                    if p not in valid_tokens(c,final=i==len(sequence.pitches)-1):
                        raise ValueError("Source melody violates token contract")
                trace.update(seed.to_bytes(4,"little"))
                trace.update(bytes.fromhex(fingerprint(sequence)))
                rows.append((score,sequence))
    if len(rows)>MAX_SOURCES:
        raise ValueError("Melody corpus exceeds source budget")
    return tuple(rows),trace.hexdigest()


def split_corpus(training_seeds,development_seeds,*,bars=8):
    _seed_panel(training_seeds)
    _seed_panel(development_seeds)
    if set(training_seeds)&set(development_seeds):
        raise ValueError("Melody source seeds overlap")
    training,train_hash=corpus(training_seeds,bars=bars)
    development,dev_hash=corpus(development_seeds,bars=bars)
    # Reject exact normalized melody matches even if context differs.
    # Keep one representative of each full sequence in training.
    unique={}
    for score,seq in training:
        unique.setdefault(fingerprint(seq),(score,seq))
    train=tuple(unique.values())
    seen={fingerprint(seq,pitches_only=True) for _,seq in train}
    kept=[]
    for score,seq in development:
        if fingerprint(seq,pitches_only=True) not in seen:
            kept.append((score,seq))
    if not kept:
        raise ValueError("All development melodies overlap training")
    # Do not silently lose a whole style/mode combination.
    expected={(style,mode) for style in STYLES for mode in MODES}
    if {(s.plan.style,s.plan.mode) for s,_ in kept}!=expected:
        raise ValueError("Melody overlap removed a complete development stratum")
    return train,tuple(kept),{"training_source_sha256":train_hash,
                             "development_source_sha256":dev_hash,
                             "training_source_rows":len(training),"training_unique_rows":len(train),
                             "development_source_rows":len(development),
                             "development_retained_rows":len(kept),
                             "development_exact_pitch_duplicates_removed":len(development)-len(kept),
                             "cross_split_exact_relative_pitch_overlap":False}


def collate(sequences):
    if (type(sequences) not in (list,tuple) or not 1<=len(sequences)<=32
            or any(not isinstance(s,MelodySequence) for s in sequences)):
        raise ValueError("Invalid melody sequence batch")
    length=max(len(s.pitches) for s in sequences)
    previous=torch.full((len(sequences),length),PAD,dtype=torch.long)
    targets=torch.full((len(sequences),length),-100,dtype=torch.long)
    context=torch.zeros((len(sequences),length,len(CONTEXT_SIZES)),dtype=torch.long)
    for i,seq in enumerate(sequences):
        n=len(seq.pitches)
        previous[i,:n]=torch.tensor((BOS,*seq.pitches[:-1]),dtype=torch.long)
        targets[i,:n]=torch.tensor(seq.pitches,dtype=torch.long)
        context[i,:n]=torch.tensor(seq.contexts,dtype=torch.long)
    return previous,context,targets
