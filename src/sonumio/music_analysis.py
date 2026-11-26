"""Numerical and symbolic diagnostics for original instrumental arrangements.

These measure structure and signal behavior, not subjective music quality,
novelty against copyrighted works, or human preference.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

from .arrangement import Arrangement, TRACKS, PITCHED, chord_classes, arrangement_digest
from .audio import SAMPLE_RATE
from .errors import GenerationError


def _entropy(counts):
    total=sum(counts)
    if not total:
        return 0.
    return -sum((n/total)*math.log2(n/total) for n in counts if n)


def _track_summary(events, ticks):
    if not events:
        return {"events":0,"active_fraction":0.,"pitch_min":None,"pitch_max":None,
                "pitch_class_entropy_bits":0.,"onset_entropy_bits":0.,"max_polyphony":0}
    bounds=[]
    for e in events:
        bounds.extend(((e.tick,1),(e.tick+e.length,-1)))
    active=0
    peak=0
    occupied=0
    last=0
    for tick,delta in sorted(bounds):
        if active:
            occupied+=tick-last
        active+=delta
        peak=max(active,peak)
        last=tick
    return {"events":len(events),"active_fraction":occupied/ticks,
            "pitch_min":min(e.pitch for e in events),"pitch_max":max(e.pitch for e in events),
            "pitch_class_entropy_bits":_entropy(Counter(e.pitch%12 for e in events).values()),
            "onset_entropy_bits":_entropy(Counter(e.tick%32 for e in events).values()),
            "max_polyphony":peak}


def analyze_score(score):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected Arrangement for analysis")
    notes=score.events
    lead=[e for e in notes if e.track=="lead"]
    intervals=[b.pitch-a.pitch for a,b in zip(lead,lead[1:])]
    strong=[e for e in lead if e.tick%8==0]
    chord_hits=sum(e.pitch%12 in chord_classes(score.plan,score.harmony[e.tick//32]) for e in strong)
    # Relative interval strings ignore absolute key; count motif repetition
    # only within each bar rather than across structural boundaries.
    phrases=[]
    for bar in range(score.plan.bars):
        pitches=[e.pitch for e in lead if e.tick//32==bar]
        if len(pitches)>=3:
            phrases.append(tuple(b-a for a,b in zip(pitches,pitches[1:])))
    counts=Counter(phrases)
    sections=[]
    for section in score.sections:
        start=section.first_bar*32
        end=(section.first_bar+section.bars)*32
        subset=[e for e in notes if start<=e.tick<end]
        sections.append({"kind":section.kind,"bars":section.bars,"events":len(subset),
                         "events_per_bar":len(subset)/section.bars,
                         "tracks_present":[t for t in TRACKS if any(e.track==t for e in subset)]})
    return {"score_sha256":arrangement_digest(score),"events":len(notes),
            "bars":score.plan.bars,"duration_seconds":score.plan.frame(score.plan.ticks)/SAMPLE_RATE,
            "tracks":{t:_track_summary([e for e in notes if e.track==t],score.plan.ticks)
                      for t in TRACKS},"sections":sections,
            "lead_intervals":len(intervals),
            "mean_absolute_lead_interval":sum(abs(i) for i in intervals)/len(intervals) if intervals else None,
            "maximum_absolute_lead_interval":max(map(abs,intervals)) if intervals else None,
            "strong_beat_chord_hits":chord_hits,"strong_beat_notes":len(strong),
            "final_lead_tonic":bool(lead and lead[-1].pitch%12==score.plan.root%12),
            "phrase_count":len(phrases),"distinct_interval_phrases":len(counts),
            "repeated_phrase_fraction":sum(n for n in counts.values() if n>1)/len(phrases) if phrases else None,
            "perceptual_quality_established":False}


def analyze_audio(samples):
    if (not isinstance(samples,np.ndarray) or samples.dtype!=np.float32
            or samples.ndim!=2 or samples.shape[0]!=2
            or not 1<=samples.shape[1]<=120*SAMPLE_RATE or not np.isfinite(samples).all()):
        raise GenerationError("Expected finite bounded stereo PCM")
    x=samples.astype(np.float64)
    peak=np.max(np.abs(x),axis=1)
    rms=np.sqrt(np.mean(x*x,axis=1))
    centered=x-x.mean(axis=1,keepdims=True)
    norms=np.sqrt(np.sum(centered*centered,axis=1))
    corr=float(np.dot(centered[0],centered[1])/(norms[0]*norms[1])) if np.all(norms>1e-12) else None
    mono=x.mean(axis=0)
    reference=float(np.sqrt(np.mean(x*x)))
    retention=float(np.sqrt(np.mean(mono*mono))/reference) if reference>1e-12 else None
    window=SAMPLE_RATE//2
    timeline=[]
    for start in range(0,x.shape[1],window):
        data=x[:,start:start+window]
        timeline.append({"start_frame":start,"frames":data.shape[1],
                         "rms":float(np.sqrt(np.mean(data*data))),
                         "peak":float(np.max(np.abs(data)))})
    # Report actual silence/quiet windows, not perceptual dropout claims.
    return {"sample_rate":SAMPLE_RATE,"frames":x.shape[1],"channel_peak":peak.tolist(),
            "channel_rms":rms.tolist(),"channel_dc":x.mean(axis=1).tolist(),
            "centered_stereo_correlation":None if corr is None else max(-1.,min(1.,corr)),
            "mono_mix_rms_retention":retention,
            "max_adjacent_step":float(np.max(np.abs(np.diff(x,axis=1)))) if x.shape[1]>1 else 0.,
            "quiet_half_second_windows":sum(w["rms"]<.0001 for w in timeline),
            "timeline":timeline,"sample_peak_only":True,"perceptual_quality_established":False}
