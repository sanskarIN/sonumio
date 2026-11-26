"""Original soundtrack assembly from symbolic music and nature scenes.

Use a common integer frame clock. No external recordings, neural model
weights, or automatic loudness claims. The renderer mixes two known
procedural sources, with optional envelope-driven ambience ducking.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import hashlib
import json
import math
import numpy as np

from .arrangement import Arrangement,arrangement_digest,MAX_FRAMES,SAMPLE_RATE
from .music_mix import MixPlan,render_stems,mix_stems
from .nature_scene import NatureScene,scene_digest,render_scene
from .audio import encode_wav_channels,metrics_channels
from .errors import GenerationError

PROTOCOL="sonumio-original-soundtrack-v1"
MAX_TRACKS=2
CONTROL_HOP=240
MAX_RENDER_FRAMES=MAX_FRAMES*2


def _int(value,low,high,name):
    if type(value) is not int or not low<=value<=high:
        raise GenerationError("Invalid soundtrack "+name)
    return value


def _real(value,low,high,name):
    if type(value) not in (int,float) or not low<=value<=high:
        raise GenerationError("Invalid soundtrack "+name)
    return float(value)


@dataclass(frozen=True,slots=True)
class Placement:
    source_start:int=0
    destination_start:int=0
    frames:int|None=None
    gain_db:float=0.
    fade_in:int=480
    fade_out:int=1200

    def __post_init__(self):
        _int(self.source_start,0,MAX_FRAMES-1,"source start")
        _int(self.destination_start,0,MAX_FRAMES-1,"destination start")
        if self.frames is not None:
            _int(self.frames,1,MAX_FRAMES,"placement frames")
        _real(self.gain_db,-60,12,"gain")
        _int(self.fade_in,0,MAX_FRAMES,"fade in")
        _int(self.fade_out,0,MAX_FRAMES,"fade out")

    def resolve(self,source_frames,destination_frames):
        _int(source_frames,1,MAX_FRAMES,"source frames")
        _int(destination_frames,1,MAX_FRAMES,"destination frames")
        n=source_frames-self.source_start if self.frames is None else self.frames
        if (n<1 or self.source_start+n>source_frames
                or self.destination_start+n>destination_frames):
            raise GenerationError("Soundtrack placement outside frame geometry")
        if self.fade_in+self.fade_out>n:
            raise GenerationError("Soundtrack fades overlap")
        return n


@dataclass(frozen=True,slots=True)
class Ducking:
    enabled:bool=False
    threshold_dbfs:float=-30.
    ratio:float=3.
    max_reduction_db:float=9.
    attack_seconds:float=.05
    release_seconds:float=.3

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise GenerationError("Invalid ducking enabled flag")
        _real(self.threshold_dbfs,-80,-3,"duck threshold")
        _real(self.ratio,1,20,"duck ratio")
        _real(self.max_reduction_db,0,30,"duck maximum reduction")
        _real(self.attack_seconds,.01,2,"duck attack")
        _real(self.release_seconds,.01,5,"duck release")


@dataclass(frozen=True,slots=True)
class SoundtrackPlan:
    frames:int
    music:Placement=Placement(gain_db=-3.)
    ambience:Placement=Placement(gain_db=-6.)
    music_mix:MixPlan=MixPlan()
    ducking:Ducking=Ducking()
    peak_ceiling:float=.9

    def __post_init__(self):
        _int(self.frames,1,MAX_FRAMES,"duration")
        if not isinstance(self.music,Placement) or not isinstance(self.ambience,Placement):
            raise GenerationError("Invalid soundtrack source placements")
        if not isinstance(self.music_mix,MixPlan) or not isinstance(self.ducking,Ducking):
            raise GenerationError("Invalid soundtrack mix controls")
        _real(self.peak_ceiling,.05,.99,"peak ceiling")


def _pcm(value):
    if (not isinstance(value,np.ndarray) or value.dtype not in (np.float32,np.float64)
            or value.ndim!=2 or value.shape[0]!=2
            or not 1<=value.shape[1]<=MAX_FRAMES or not np.isfinite(value).all()
            or np.max(np.abs(value))>16):
        raise GenerationError("Expected bounded finite stereo soundtrack source")
    return value


def _fade(count,attack,release):
    value=np.ones(count,dtype=np.float64)
    if attack==1:
        value[0]=0.
    elif attack:
        value[:attack]=np.sin(np.linspace(0,math.pi/2,attack))**2
    if release==1:
        value[-1]=0.
    elif release:
        value[-release:]=np.cos(np.linspace(0,math.pi/2,release))**2
    return value


def place(source,placement,frames):
    source=_pcm(source)
    if not isinstance(placement,Placement):
        raise GenerationError("Expected Placement")
    n=placement.resolve(source.shape[1],frames)
    out=np.zeros((2,frames),dtype=np.float64)
    src=source[:,placement.source_start:placement.source_start+n].astype(np.float64)
    src*=10**(placement.gain_db/20)
    src*=_fade(n,placement.fade_in,placement.fade_out)
    out[:,placement.destination_start:placement.destination_start+n]=src
    return out


def ducking_envelope(sidechain,config):
    """Causal block detector with a one-hop delay.

    RMS in a completed 10 ms block sets the gain target of the next
    block. Interpolate between previous and smoothed target gain in dB.
    No future audio samples influence earlier gain.
    """
    x=_pcm(sidechain)
    if not isinstance(config,Ducking):
        raise GenerationError("Expected Ducking")
    frames=x.shape[1]
    gain=np.ones(frames,dtype=np.float64)
    if not config.enabled or config.ratio==1 or config.max_reduction_db==0:
        return gain,{"enabled":config.enabled,"max_reduction_db":0.,
                     "mean_reduction_db":0.,"control_hop_frames":CONTROL_HOP}
    state=0.
    target=0.
    reductions=[]
    for start in range(0,frames,CONTROL_HOP):
        stop=min(frames,start+CONTROL_HOP)
        n=stop-start
        # Attack means increasing attenuation; release means returning to zero.
        tau=config.attack_seconds if target<state else config.release_seconds
        next_state=target+(state-target)*math.exp(-n/(SAMPLE_RATE*tau))
        db=np.linspace(state,next_state,n,endpoint=False)
        gain[start:stop]=10**(db/20)
        reductions.append(float(-db.sum()))
        state=next_state
        power=float(np.mean(x[:,start:stop].astype(np.float64)**2))
        level=10*math.log10(max(power,1e-15))
        above=max(0.,level-config.threshold_dbfs)
        target=-min(config.max_reduction_db,above*(1-1/config.ratio))
    return gain,{"enabled":True,"max_reduction_db":float(-20*np.log10(gain.min())),
                 "mean_reduction_db":sum(reductions)/frames,"control_hop_frames":CONTROL_HOP}


def _source_stats(audio):
    return {"peak":float(np.max(np.abs(audio))),"rms":float(np.sqrt(np.mean(audio**2))),
            "active_frames":int(np.count_nonzero(np.max(np.abs(audio),axis=0)>1e-5))}


def combine(music,ambience,plan):
    if not isinstance(plan,SoundtrackPlan):
        raise GenerationError("Expected SoundtrackPlan")
    m=place(music,plan.music,plan.frames)
    a=place(ambience,plan.ambience,plan.frames)
    envelope,control=ducking_envelope(m,plan.ducking)
    ducked=a*envelope
    mix=m+ducked
    peak=float(np.max(np.abs(mix)))
    scale=min(1.,plan.peak_ceiling/max(peak,1e-15))
    mix*=scale
    output=mix.astype(np.float32)
    if not np.isfinite(output).all():
        raise GenerationError("Nonfinite soundtrack PCM")
    return output,{"music_placed":_source_stats(m),"ambience_placed":_source_stats(a),
                   "ambience_after_ducking":_source_stats(ducked),"ducking":control,
                   "pre_master_peak":peak,"master_attenuation":scale,
                   "metrics":metrics_channels(output),"true_peak_measured":False,
                   "lufs_measured":False}


def soundtrack_digest(score,scene,plan):
    if (not isinstance(score,Arrangement) or not isinstance(scene,NatureScene)
            or not isinstance(plan,SoundtrackPlan)):
        raise GenerationError("Invalid soundtrack identity")
    payload={"score_sha256":arrangement_digest(score),"scene_sha256":scene_digest(scene),
             "plan":asdict(plan)}
    value=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+value).hexdigest()


def render_soundtrack(score,scene,plan):
    if not isinstance(score,Arrangement) or not isinstance(scene,NatureScene):
        raise GenerationError("Expected original music and nature sources")
    if not isinstance(plan,SoundtrackPlan):
        raise GenerationError("Expected SoundtrackPlan")
    music_frames=score.plan.frame(score.plan.ticks)
    if music_frames+scene.frames>MAX_RENDER_FRAMES:
        raise GenerationError("Soundtrack source render budget exceeded")
    # Reject invalid crops and placements before allocating complete stems.
    plan.music.resolve(music_frames,plan.frames)
    plan.ambience.resolve(scene.frames,plan.frames)
    stems=render_stems(score)
    music,music_report=mix_stems(stems,plan.music_mix)
    del stems
    ambience,nature_report=render_scene(scene)
    samples,report=combine(music,ambience,plan)
    report.update({"protocol":PROTOCOL,"soundtrack_sha256":soundtrack_digest(score,scene,plan),
                   "score_sha256":arrangement_digest(score),"scene_sha256":scene_digest(scene),
                   "frames":plan.frames,"sample_rate":SAMPLE_RATE,"channels":2,
                   "music_source_metrics":music_report["metrics"],
                   "nature_source_metrics":nature_report["metrics"],
                   "learned_model":False,"external_recordings_used":False,
                   "perceptual_quality_established":False})
    return samples,report


def generate_soundtrack(score,scene,plan):
    samples,report=render_soundtrack(score,scene,plan)
    wav=encode_wav_channels(samples,SAMPLE_RATE)
    report["wav_sha256"]=hashlib.sha256(wav).hexdigest()
    return wav,report
