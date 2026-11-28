"""Deterministic stem rendering, linear mixing and bounded mastering.

Render dry stems once. A track's signal is independent of mute/solo,
track order, mixing gain and other instruments. Reverb is finite taps,
not feedback. No loudness-standard or true-peak claim is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import numpy as np

from .audio import SAMPLE_RATE, encode_wav_channels, metrics_channels
from .arrangement import Arrangement, TRACKS, MAX_FRAMES, arrangement_digest, real
from .music_instruments import render_event, smooth_edges
from .errors import GenerationError

DEFAULT_LEVELS=(.20,.14,.26,.25,.16,.23,.18,.12)
MAX_STEM_BYTES=256*1024*1024


@dataclass(frozen=True,slots=True)
class MixPlan:
    gains:tuple[float,...]=DEFAULT_LEVELS
    mute:tuple[str,...]=()
    solo:tuple[str,...]=()
    room:float=.15
    peak_ceiling:float=.9

    def __post_init__(self):
        if type(self.gains) is not tuple or len(self.gains)!=len(TRACKS):
            raise GenerationError("Expected one mix gain per track")
        for v in self.gains:
            real(v,0,2,"track gain")
        for names in (self.mute,self.solo):
            if (type(names) is not tuple or any(type(v) is not str or v not in TRACKS for v in names)
                    or len(set(names))!=len(names)):
                raise GenerationError("Invalid mute or solo list")
        if set(self.mute)&set(self.solo):
            raise GenerationError("Cannot mute and solo the same track")
        real(self.room,0,1,"room send")
        real(self.peak_ceiling,.05,.99,"peak ceiling")

    def active(self,track):
        return track not in self.mute and (not self.solo or track in self.solo)


def _validate_stem(stem,frames):
    if (not isinstance(stem,np.ndarray) or stem.dtype!=np.float32
            or stem.shape!=(2,frames) or not np.isfinite(stem).all()):
        raise GenerationError("Invalid stereo stem")


def render_stems(score):
    """Render all notes with the original procedural instruments."""
    return _render_stems(score, render_event)


def _render_stems(score, renderer):
    """Shared accumulation path for the original engine and local research."""
    if not callable(renderer):
        raise GenerationError("Expected callable note renderer")
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected Arrangement")
    frames=score.plan.frame(score.plan.ticks)
    if not 1<=frames<=MAX_FRAMES or len(TRACKS)*2*frames*4>MAX_STEM_BYTES:
        raise GenerationError("Stem allocation exceeds budget")
    stems={track:np.zeros((2,frames),dtype=np.float32) for track in TRACKS}
    for event in score.events:
        start=score.plan.frame(event.tick)
        stop=score.plan.frame(event.tick+event.length)
        mono=renderer(event,stop-start,score.plan.seed)
        if (not isinstance(mono,np.ndarray) or mono.dtype not in (np.float32,np.float64)
                or mono.shape!=(stop-start,) or not np.isfinite(mono).all()
                or np.max(np.abs(mono))>1.000001):
            raise GenerationError("Invalid bounded note renderer output")
        pan=(event.pan+1)*math.pi/4
        stems[event.track][0,start:stop]+=(mono*math.cos(pan)).astype(np.float32)
        stems[event.track][1,start:stop]+=(mono*math.sin(pan)).astype(np.float32)
    for stem in stems.values():
        _validate_stem(stem,frames)
        stem.setflags(write=False)
    return stems


def room_response(dry,amount):
    real(amount,0,1,"room amount")
    result=dry.copy()
    # Stereo cross-reflections with fixed delays and decaying energy.
    for ms,gain,cross in ((29,.23,False),(43,.19,True),(71,.14,True),
                          (113,.09,False),(173,.06,True),(251,.03,False)):
        shift=round(ms*SAMPLE_RATE/1000)
        if shift<dry.shape[1]:
            result[:,shift:]+=(dry[::-1,:-shift] if cross else dry[:,:-shift])*(gain*amount)
    return result


def mix_stems(stems,plan=None):
    plan=MixPlan() if plan is None else plan
    if not isinstance(plan,MixPlan) or type(stems) is not dict or set(stems)!=set(TRACKS):
        raise GenerationError("Invalid stem mix")
    first=stems[TRACKS[0]]
    if not isinstance(first,np.ndarray) or first.ndim!=2:
        raise GenerationError("Invalid first stem")
    frames=first.shape[1]
    if not 1<=frames<=MAX_FRAMES or len(TRACKS)*2*frames*4>MAX_STEM_BYTES:
        raise GenerationError("Mix exceeds frame budget")
    dry=np.zeros((2,frames),dtype=np.float64)
    track_stats={}
    for track,gain in zip(TRACKS,plan.gains,strict=True):
        stem=stems[track]
        _validate_stem(stem,frames)
        active=plan.active(track)
        if active:
            dry+=stem.astype(np.float64)*gain
        track_stats[track]={"active":active,"gain":float(gain),
                            "dry_peak":float(np.max(np.abs(stem))),
                            "dry_rms":float(np.sqrt(np.mean(stem.astype(np.float64)**2)))}
    wet=room_response(dry,plan.room)
    wet*=smooth_edges(frames,240,1200)
    pre=float(np.max(np.abs(wet)))
    scale=min(1.,plan.peak_ceiling/max(pre,1e-15))
    wet*=scale
    audio=wet.astype(np.float32)
    if not np.isfinite(audio).all():
        raise GenerationError("Nonfinite music mix")
    report={"tracks":track_stats,"pre_master_peak":pre,"master_attenuation":scale,
            "room_amount":float(plan.room),"peak_ceiling":float(plan.peak_ceiling),
            "metrics":metrics_channels(audio),"true_peak_measured":False,
            "lufs_measured":False}
    return audio,report


def generate_arrangement_audio(score,plan=None):
    stems=render_stems(score)
    samples,report=mix_stems(stems,plan)
    wav=encode_wav_channels(samples,SAMPLE_RATE)
    report.update({"sample_rate":SAMPLE_RATE,"channels":2,"frames":samples.shape[1],
                   "score_sha256":arrangement_digest(score),
                   "wav_sha256":hashlib.sha256(wav).hexdigest(),
                   "learned_model":False,"external_recordings_used":False})
    return wav,report
