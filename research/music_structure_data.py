"""Bar-level harmony and rhythm representation for original music research.

A 32-bit onset mask represents a single monophonic lead rhythm in 4/4.
A duration policy reconstructs notes without overlapping or borrowing
target pitches. Chords are diatonic degrees, not audio or chord names.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import numpy as np

from sonumio.arrangement import (
    Arrangement, ArrangementPlan, Event, TRACKS, PITCHED, SECTIONS,
    compose_arrangement, sections, event_key, chord_classes, degree_pitch,
    voice_chord, TICKS_PER_BAR,
)
from sonumio.music import STYLES, MODES

PROTOCOL="sonumio-bar-structure-v1"
CHORD_COUNT=7
CHORD_BOS=7
RHYTHM_COUNT=8
RHYTHM_BOS=8
MAX_BARS=32
# Explicit original vocabulary. Silence is available only outside finale.
PATTERNS=(
    (), (0,16), (0,8,16,24), (0,6,12,16,22,28),
    (0,12,20,28), (0,4,12,16,24), (0,8,12,24,28),
    (0,4,8,12,16,20,24,28),
)
CONTEXT_SIZES=(len(STYLES),len(MODES),len(SECTIONS),4,2)
VOCAB_HASH=hashlib.sha256(json.dumps(PATTERNS,separators=(",",":")).encode()).hexdigest()


def _context(value):
    if (type(value) is not tuple or len(value)!=len(CONTEXT_SIZES)
            or any(type(x) is not int or not 0<=x<n
                   for x,n in zip(value,CONTEXT_SIZES,strict=True))):
        raise ValueError("Invalid bar context")
    return value


@dataclass(frozen=True,slots=True)
class BarSequence:
    chords:tuple[int,...]
    rhythms:tuple[int,...]
    contexts:tuple[tuple[int,...],...]

    def __post_init__(self):
        if (type(self.chords) is not tuple or not 4<=len(self.chords)<=MAX_BARS
                or any(type(x) is not int or not 0<=x<CHORD_COUNT for x in self.chords)):
            raise ValueError("Invalid bar chord sequence")
        if (type(self.rhythms) is not tuple or len(self.rhythms)!=len(self.chords)
                or any(type(x) is not int or not 0<=x<RHYTHM_COUNT for x in self.rhythms)):
            raise ValueError("Invalid bar rhythm sequence")
        if type(self.contexts) is not tuple or len(self.contexts)!=len(self.chords):
            raise ValueError("Invalid bar context sequence")
        for value in self.contexts:
            _context(value)
        if tuple(c[-1] for c in self.contexts)!=(0,)*(len(self.contexts)-1)+(1,):
            raise ValueError("Invalid final-bar context marker")


def schedule(plan):
    if not isinstance(plan,ArrangementPlan):
        raise ValueError("Expected arrangement plan")
    out=[]
    for section in sections(plan):
        for local in range(section.bars):
            bar=section.first_bar+local
            out.append((STYLES.index(plan.style),tuple(MODES).index(plan.mode),
                        SECTIONS.index(section.kind),local%4,int(bar==plan.bars-1)))
    return tuple(out)


def extract(score):
    if not isinstance(score,Arrangement):
        raise ValueError("Expected Arrangement")
    contexts=[]
    for section in score.sections:
        for local in range(section.bars):
            bar=section.first_bar+local
            contexts.append((STYLES.index(score.plan.style),tuple(MODES).index(score.plan.mode),
                             SECTIONS.index(section.kind),local%4,int(bar==score.plan.bars-1)))
    rhythm=[]
    for bar in range(score.plan.bars):
        ticks=tuple(e.tick%TICKS_PER_BAR for e in score.events
                    if e.track=="lead" and e.tick//TICKS_PER_BAR==bar)
        if ticks not in PATTERNS:
            raise ValueError("Lead rhythm outside original bar vocabulary")
        rhythm.append(PATTERNS.index(ticks))
    return BarSequence(score.harmony,tuple(rhythm),tuple(contexts))


def fingerprint(sequence,*,tokens_only=False):
    if not isinstance(sequence,BarSequence):
        raise ValueError("Expected bar sequence")
    value=(sequence.chords,sequence.rhythms) if tokens_only else (
        sequence.chords,sequence.rhythms,sequence.contexts)
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+json.dumps(value,separators=(",",":")).encode()).hexdigest()


def allowed_chords(index,count):
    if type(index) is not int or type(count) is not int or not 0<=index<count or not 4<=count<=32:
        raise ValueError("Invalid bar position")
    if index==count-1:
        return (0,)
    if index==count-2:
        return (4,)
    return tuple(range(CHORD_COUNT))


def allowed_rhythms(context):
    _context(context)
    if context[-1]:
        return tuple(range(1,RHYTHM_COUNT))
    return tuple(range(RHYTHM_COUNT))


def _panel(seeds):
    if (type(seeds) is not tuple or not 1<=len(seeds)<=32
            or any(type(x) is not int or not 0<=x<2**32 for x in seeds)
            or len(set(seeds))!=len(seeds)):
        raise ValueError("Invalid bar corpus seed panel")


def corpus(seeds,bar_counts):
    _panel(seeds)
    if (type(bar_counts) is not tuple or not 1<=len(bar_counts)<=4
            or len(set(bar_counts))!=len(bar_counts)
            or any(type(x) is not int or not 4<=x<=16 for x in bar_counts)):
        raise ValueError("Invalid corpus bar lengths")
    if len(seeds)*len(bar_counts)*len(STYLES)*len(MODES)>384:
        raise ValueError("Bar source budget exceeded")
    trace=hashlib.sha256(PROTOCOL.encode()+b"\0")
    rows=[]
    for seed in seeds:
        for bars in bar_counts:
            for style in STYLES:
                for mode in MODES:
                    plan=ArrangementPlan(style=style,mode=mode,bars=bars,bpm=120,seed=seed)
                    seq=extract(compose_arrangement(plan))
                    if any(c not in allowed_chords(i,bars) or r not in allowed_rhythms(ctx)
                           for i,(c,r,ctx) in enumerate(zip(seq.chords,seq.rhythms,seq.contexts,strict=True))):
                        raise ValueError("Original score violates structural vocabulary")
                    rows.append((plan,seq))
                    trace.update(seed.to_bytes(4,"little"))
                    trace.update(bytes.fromhex(fingerprint(seq)))
    return tuple(rows),trace.hexdigest()


def split_corpus(training_seeds,development_seeds,bar_counts=(6,8,12)):
    _panel(training_seeds)
    _panel(development_seeds)
    if set(training_seeds)&set(development_seeds):
        raise ValueError("Bar source seed overlap")
    train,th=corpus(training_seeds,bar_counts)
    dev,dh=corpus(development_seeds,bar_counts)
    # The source is deliberately finite; exact structural duplication can
    # be common. Do not pretend that a seed split excludes token overlap.
    unique={}
    for plan,seq in train:
        unique.setdefault(fingerprint(seq),(plan,seq))
    unique_dev={}
    for plan,seq in dev:
        unique_dev.setdefault(fingerprint(seq),(plan,seq))
    train=tuple(unique.values())
    dev=tuple(unique_dev.values())
    seen={fingerprint(s,tokens_only=True) for _,s in train}
    novel=tuple((p,s) for p,s in dev if fingerprint(s,tokens_only=True) not in seen)
    return train,dev,novel,{"training_sha256":th,"development_sha256":dh,
                          "training_unique_context_rows":len(train),
                          "development_unique_context_rows":len(dev),
                          "development_exact_token_overlap_rows":len(dev)-len(novel),
                          "novel_development_rows":len(novel),
                          "source_seed_overlap":False,"vocabulary_sha256":VOCAB_HASH}


def _chord_voicing(plan,degree,previous):
    return voice_chord(plan,degree,previous)


def reconstruct(plan,sequence):
    """Rebuild backing tracks from sampled harmony and rhythm.

    No original reference melody is retained. Instrument timbres and
    rhythm-to-duration policy remain deterministic procedural rules.
    """
    if not isinstance(plan,ArrangementPlan) or not isinstance(sequence,BarSequence):
        raise ValueError("Expected bar reconstruction inputs")
    if sequence.contexts!=schedule(plan):
        raise ValueError("Bar conditioning does not match output plan")
    n=len(sequence.chords)
    for i,(c,r,ctx) in enumerate(zip(sequence.chords,sequence.rhythms,sequence.contexts,strict=True)):
        if c not in allowed_chords(i,n) or r not in allowed_rhythms(ctx):
            raise ValueError("Sampled bar violates reconstruction contract")
    backing=compose_arrangement(plan)
    out=[e for e in backing.events if e.track not in PITCHED]
    previous=(plan.root-12,plan.root-8,plan.root-5)
    lead=plan.root+12
    for bar,(degree,rhythm) in enumerate(zip(sequence.chords,sequence.rhythms,strict=True)):
        start=bar*TICKS_PER_BAR
        chord=_chord_voicing(plan,degree,previous)
        previous=chord
        for pitch,pan in zip(chord,(-.65,0.,.65),strict=True):
            out.append(Event("pad",pitch,start,32,.32,pan))
        for j,tick in enumerate((0,8,16,24)):
            out.append(Event("keys",chord[j%3],start+tick,7,.4,-.3))
            root=degree_pitch(plan,degree+(4 if j%2 else 0))-12
            while root>59:
                root-=12
            out.append(Event("bass",root,start+tick,7,.5,0.))
        ticks=PATTERNS[rhythm]
        scale={(plan.root+x)%12 for x in MODES[plan.mode]}
        for j,tick in enumerate(ticks):
            pcs=set(chord_classes(plan,degree)) if tick%8==0 else scale
            if bar==n-1 and j==len(ticks)-1:
                pcs={plan.root%12}
            direction=(-2,3,1,-1)[(bar+j)%4]
            options=[p for p in range(plan.root,plan.root+25) if p%12 in pcs]
            lead=min(options,key=lambda p:(abs(p-lead-direction),abs(p-lead),p))
            stop=ticks[j+1] if j+1<len(ticks) else 32
            out.append(Event("lead",lead,start+tick,max(1,stop-tick-1),.53,.18))
    return Arrangement(plan,sections(plan),sequence.chords,tuple(sorted(out,key=event_key)))
