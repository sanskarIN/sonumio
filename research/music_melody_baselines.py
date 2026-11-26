"""Training-only statistical baselines for conditional melody prediction."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from research.music_melody_data import MelodySequence, PITCH_COUNT, BOS


class MelodyCounts:
    """Laplace-smoothed unigram and hierarchical conditional bigram.

    Exact contextual bigram falls back to context-only, then previous
    pitch-only, then global counts. Backoff occurs only for unseen keys.
    All arrays are copied before returning a probability distribution.
    """

    def __init__(self,sequences,alpha=.5):
        if type(sequences) not in (list,tuple) or not sequences:
            raise ValueError("Expected nonempty training sequence panel")
        if type(alpha) not in (int,float) or not .01<=alpha<=10:
            raise ValueError("Invalid count smoothing")
        self.alpha=float(alpha)
        self.global_counts=np.zeros(PITCH_COUNT,dtype=np.int64)
        self.context_counts={}
        self.previous_counts={}
        self.joint_counts={}
        for seq in sequences:
            if not isinstance(seq,MelodySequence):
                raise ValueError("Invalid training sequence")
            previous=BOS
            for pitch,context in zip(seq.pitches,seq.contexts,strict=True):
                key=self.key(context)
                self.global_counts[pitch]+=1
                self._increment(self.context_counts,key,pitch)
                self._increment(self.previous_counts,previous,pitch)
                self._increment(self.joint_counts,(key,previous),pitch)
                previous=pitch

    @staticmethod
    def key(context):
        style,mode,degree,tick,section=context
        return (style,mode,degree,int(tick%8==0),section)

    @staticmethod
    def _increment(table,key,pitch):
        if key not in table:
            table[key]=np.zeros(PITCH_COUNT,dtype=np.int64)
        table[key][pitch]+=1

    def probability(self,context,previous,kind):
        if kind=="uniform":
            return np.full(PITCH_COUNT,1/PITCH_COUNT,dtype=np.float64)
        if kind=="unigram":
            counts=self.global_counts
        elif kind=="conditional_bigram":
            key=self.key(context)
            counts=self.joint_counts.get((key,previous))
            if counts is None:
                counts=self.context_counts.get(key)
            if counts is None:
                counts=self.previous_counts.get(previous,self.global_counts)
        else:
            raise ValueError("Unknown melody count baseline")
        out=counts.astype(np.float64)+self.alpha
        return out/out.sum()


def evaluate_counts(model,sequences,kind):
    rows=[]
    for seq in sequences:
        previous=BOS
        probability=[]
        for context,pitch in zip(seq.contexts,seq.pitches,strict=True):
            probability.append(model.probability(context,previous,kind))
            previous=pitch
        rows.append(np.stack(probability))
    return tuple(rows)
