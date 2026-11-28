"""Paired render comparisons on the exact same declared musical clock.

The comparison never shifts, normalizes, time-stretches or resamples
the source files. A separate least-squares gain diagnostic helps
distinguish level changes from waveform differences but is not applied
to either file. Geometry association alone is not provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib

import numpy as np

from .arrangement import arrangement_digest, TICKS_PER_BAR
from .errors import GenerationError
from .music_render_analysis import (
    AnalysisSettings, analyze_render, pcm_view, _bounds, BLOCK, BANDS,
)

PROTOCOL="sonumio-paired-music-render-v1"


@dataclass(slots=True)
class PairMoments:
    frames:int=0
    source_squares:int=0
    candidate_squares:int=0
    dot:int=0
    absolute_difference:int=0
    maximum_difference:int=0
    differing_frames:int=0
    differing_samples:int=0

    def add(self,source,candidate):
        if (not isinstance(source,np.ndarray) or not isinstance(candidate,np.ndarray)
                or source.dtype!=np.dtype("<i2") or candidate.dtype!=np.dtype("<i2")
                or source.shape!=candidate.shape or source.ndim!=2
                or source.shape[1]!=2 or not 1<=len(source)<=BLOCK):
            raise GenerationError("Invalid paired render PCM block")
        a=source.astype(np.int64)
        b=candidate.astype(np.int64)
        difference=b-a
        absolute=np.abs(difference)
        self.frames+=len(a)
        self.source_squares+=int((a*a).sum())
        self.candidate_squares+=int((b*b).sum())
        self.dot+=int((a*b).sum())
        self.absolute_difference+=int(absolute.sum())
        self.maximum_difference=max(self.maximum_difference,int(absolute.max()))
        self.differing_frames+=int(np.any(difference!=0,axis=1).sum())
        self.differing_samples+=int((difference!=0).sum())

    def merge(self,other):
        if not isinstance(other,PairMoments) or other.frames<1:
            raise GenerationError("Expected nonempty paired PCM moments")
        for key in ("frames","source_squares","candidate_squares","dot",
                    "absolute_difference","differing_frames","differing_samples"):
            setattr(self,key,getattr(self,key)+getattr(other,key))
        self.maximum_difference=max(self.maximum_difference,other.maximum_difference)

    def report(self):
        if self.frames<1:
            raise GenerationError("Cannot report empty paired PCM moments")
        a,b,dot=self.source_squares,self.candidate_squares,self.dot
        error=a+b-2*dot
        if error<0 or dot*dot>a*b:
            raise GenerationError("Invalid paired PCM energy identity")
        gain=dot/a if a else None
        # Exact integer numerator of the minimum least-squares residual.
        # A negative gain is reported (e.g. polarity inversion), not used.
        residual_numerator=a*b-dot*dot
        residual=residual_numerator/a if a else None
        return {
            "frames":self.frames,
            "identical_frames":self.frames-self.differing_frames,
            "different_samples":self.differing_samples,
            "different_sample_fraction":self.differing_samples/(2*self.frames),
            "mean_absolute_difference":self.absolute_difference/(2*self.frames*32768.),
            "maximum_absolute_difference":self.maximum_difference/32768.,
            "absolute_wave_mse":error/(2*self.frames*32768.**2),
            "relative_wave_mse":error/a if a else None,
            "rms_ratio":math.sqrt(b/a) if a else None,
            "waveform_cosine":max(-1.,min(1.,dot/math.sqrt(a*b))) if a and b else None,
            "least_squares_gain":gain,
            "gain_fit_residual_mse":residual/(2*self.frames*32768.**2) if residual is not None else None,
            "gain_fit_relative_residual":residual/b if residual is not None and b else None,
            "integer":{"source_sum_squares":a,"candidate_sum_squares":b,
                       "dot_product":dot,"error_sum_squares":error},
        }


def compare_interval(source,candidate,start,stop):
    _bounds(source,start,stop)
    _bounds(candidate,start,stop)
    if source.shape!=candidate.shape:
        raise GenerationError("Paired render geometry differs")
    result=PairMoments()
    for cursor in range(start,stop,BLOCK):
        end=min(cursor+BLOCK,stop)
        result.add(source[cursor:end],candidate[cursor:end])
    return result


def _delta(a,b):
    return b-a if a is not None and b is not None else None


def _changes(a,b):
    source=a["signal"]
    candidate=b["signal"]
    bands={}
    for name,_,_ in BANDS:
        bands[name]=_delta(a["spectrum"]["band_fraction"][name],b["spectrum"]["band_fraction"][name])
    old={row["code"] for row in a["findings"]}
    new={row["code"] for row in b["findings"]}
    return {
        "rms_db_delta":_delta(source["rms_dbfs"],candidate["rms_dbfs"]),
        "peak_db_delta":_delta(source["peak_dbfs"],candidate["peak_dbfs"]),
        "active_fraction_delta":candidate["active_fraction"]-source["active_fraction"],
        "mono_retention_delta":_delta(source["mono_rms_retention"],candidate["mono_rms_retention"]),
        "stereo_correlation_delta":_delta(source["centered_stereo_correlation"],candidate["centered_stereo_correlation"]),
        "centroid_hz_delta":_delta(a["spectrum"]["spectral_centroid_hz"],b["spectrum"]["spectral_centroid_hz"]),
        "spectral_band_fraction_delta":bands,
        "new_findings":sorted(new-old),"resolved_findings":sorted(old-new),
        "persisting_findings":sorted(old&new),
    }


def compare_renders(score,source_wav,candidate_wav,settings=AnalysisSettings()):
    source=pcm_view(score,source_wav)
    candidate=pcm_view(score,candidate_wav)
    original=analyze_render(score,source_wav,settings)
    updated=analyze_render(score,candidate_wav,settings)
    bars=[]
    accumulators=[]
    for bar,(a,b) in enumerate(zip(original["bars"],updated["bars"],strict=True)):
        start,stop=score.plan.frame(bar*TICKS_PER_BAR),score.plan.frame((bar+1)*TICKS_PER_BAR)
        accum=compare_interval(source,candidate,start,stop)
        accumulators.append(accum)
        bars.append({"bar":bar,"start_frame":start,"stop_frame":stop,
                     "paired":accum.report(),"changes":_changes(a,b)})
    sections=[]
    total=PairMoments()
    for section,a,b in zip(score.sections,original["sections"],updated["sections"],strict=True):
        accum=PairMoments()
        for bar in range(section.first_bar,section.first_bar+section.bars):
            accum.merge(accumulators[bar])
        total.merge(accum)
        sections.append({"kind":section.kind,"first_bar":section.first_bar,"bars":section.bars,
                         "paired":accum.report(),"changes":_changes(a,b)})
    return {
        "protocol":PROTOCOL,"score_sha256":arrangement_digest(score),
        "source_wav_sha256":hashlib.sha256(source_wav).hexdigest(),
        "candidate_wav_sha256":hashlib.sha256(candidate_wav).hexdigest(),
        "paired":total.report(),"changes":_changes(original,updated),
        "bars":bars,"sections":sections,
        "source_signal":original["signal"],"candidate_signal":updated["signal"],
        "settings":original["settings"],
        "same_geometry_verified":True,"source_provenance_verified":False,
        "alignment_optimized":False,"gain_applied":False,
        "perceptual_improvement_inferred":False,
    }
