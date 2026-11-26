"""Smoothed bar-level joint statistical controls for the structure GRU."""
from __future__ import annotations

import numpy as np

from research.music_structure_data import (
    BarSequence,CHORD_COUNT,CHORD_BOS,RHYTHM_COUNT,RHYTHM_BOS,
)

BASELINES=("uniform","unigram","context","conditional")


class StructureCounts:
    def __init__(self,sequences,alpha=.5):
        if (type(sequences) not in (tuple,list) or not sequences
                or any(not isinstance(x,BarSequence) for x in sequences)):
            raise ValueError("Invalid count training panel")
        if type(alpha) not in (int,float) or not .01<=alpha<=10:
            raise ValueError("Invalid count smoothing")
        self.alpha=float(alpha)
        self.chords=np.zeros(CHORD_COUNT,dtype=np.int64)
        self.rhythms=np.zeros(RHYTHM_COUNT,dtype=np.int64)
        self.chord_context={}
        self.chord_joint={}
        self.rhythm_context={}
        self.rhythm_joint={}
        self.rhythm_chord={}
        for seq in sequences:
            previous_c=CHORD_BOS
            previous_r=RHYTHM_BOS
            for c,r,ctx in zip(seq.chords,seq.rhythms,seq.contexts,strict=True):
                self.chords[c]+=1
                self.rhythms[r]+=1
                self._add(self.chord_context,ctx,c,CHORD_COUNT)
                self._add(self.chord_joint,(ctx,previous_c,previous_r),c,CHORD_COUNT)
                self._add(self.rhythm_context,(ctx,c),r,RHYTHM_COUNT)
                self._add(self.rhythm_joint,(ctx,previous_c,previous_r,c),r,RHYTHM_COUNT)
                self._add(self.rhythm_chord,c,r,RHYTHM_COUNT)
                previous_c,previous_r=c,r

    @staticmethod
    def _add(table,key,token,count):
        if key not in table:
            table[key]=np.zeros(count,dtype=np.int64)
        table[key][token]+=1

    def _probability(self,counts):
        value=counts.astype(np.float64)+self.alpha
        return value/value.sum()

    def distributions(self,ctx,previous_c,previous_r,current_c,kind):
        if kind not in BASELINES:
            raise ValueError("Unknown structure baseline")
        if kind=="uniform":
            return np.full(CHORD_COUNT,1/CHORD_COUNT),np.full(RHYTHM_COUNT,1/RHYTHM_COUNT)
        chord=self.chords
        rhythm=self.rhythms
        if kind in ("context","conditional"):
            chord=self.chord_context.get(ctx,chord)
            rhythm=self.rhythm_context.get((ctx,current_c),
                self.rhythm_chord.get(current_c,rhythm))
        if kind=="conditional":
            chord=self.chord_joint.get((ctx,previous_c,previous_r),chord)
            rhythm=self.rhythm_joint.get((ctx,previous_c,previous_r,current_c),rhythm)
        return self._probability(chord),self._probability(rhythm)


def evaluate_counts(model,sequences,kind):
    if not isinstance(model,StructureCounts):
        raise ValueError("Expected StructureCounts")
    rows=[]
    for seq in sequences:
        previous_c=CHORD_BOS
        previous_r=RHYTHM_BOS
        chords=[]
        rhythms=[]
        for c,r,ctx in zip(seq.chords,seq.rhythms,seq.contexts,strict=True):
            pc,pr=model.distributions(ctx,previous_c,previous_r,c,kind)
            chords.append(pc)
            rhythms.append(pr)
            previous_c,previous_r=c,r
        rows.append((np.stack(chords),np.stack(rhythms)))
    return tuple(rows)
