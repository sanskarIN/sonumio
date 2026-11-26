"""Timed nature scenes with explicit independent stereo layers.

This is a separate original DSP renderer. Existing procedural-sfx-v0
output and its synthetic neural research corpus are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
import math
import numpy as np

from .nature_synthesis import (
    NatureVoice,render_voice,KINDS,MAX_FRAMES,SAMPLE_RATE,_real,_int,_envelope,
)
from .audio import encode_wav_channels,metrics_channels
from .errors import GenerationError

PROTOCOL="sonumio-nature-scene-v1"
MAX_LAYERS=12
MAX_LAYER_FRAMES=MAX_FRAMES*6
MAX_KEYS=16


@dataclass(frozen=True,slots=True)
class AutomationPoint:
    frame:int
    value:float

    def __post_init__(self):
        _int(self.frame,0,MAX_FRAMES,"automation frame")
        _real(self.value,-60,12,"automation value")


def _validate_curve(points,frames,lower,upper):
    if type(points) is not tuple or not 1<=len(points)<=MAX_KEYS:
        raise GenerationError("Invalid nature automation curve")
    if any(not isinstance(p,AutomationPoint) or not lower<=p.value<=upper for p in points):
        raise GenerationError("Invalid nature automation point")
    positions=[p.frame for p in points]
    if positions[0]!=0 or positions[-1]>=frames or positions!=sorted(set(positions)):
        raise GenerationError("Automation must start at frame zero and increase strictly")


def _curve(points,frames):
    # Interpolate absolute frame positions. Last value holds to the end.
    return np.interp(np.arange(frames,dtype=np.float64),
                     [p.frame for p in points],[p.value for p in points])


@dataclass(frozen=True,slots=True)
class NatureLayer:
    voice:NatureVoice
    start:int
    frames:int
    gain:tuple[AutomationPoint,...]=(AutomationPoint(0,-6.),)
    pan:tuple[AutomationPoint,...]=(AutomationPoint(0,0.),)
    width:float=.35
    fade_in:int=480
    fade_out:int=1200

    def __post_init__(self):
        if not isinstance(self.voice,NatureVoice):
            raise GenerationError("Expected NatureVoice layer")
        _int(self.start,0,MAX_FRAMES-1,"layer start")
        _int(self.frames,SAMPLE_RATE//10,MAX_FRAMES,"layer frames")
        if self.start+self.frames>MAX_FRAMES:
            raise GenerationError("Nature layer exceeds scene duration bound")
        _validate_curve(self.gain,self.frames,-60,12)
        _validate_curve(self.pan,self.frames,-1,1)
        _real(self.width,0,1,"stereo width")
        _int(self.fade_in,0,self.frames,"fade in")
        _int(self.fade_out,0,self.frames,"fade out")
        if self.fade_in+self.fade_out>self.frames:
            raise GenerationError("Nature layer fade durations overlap")


@dataclass(frozen=True,slots=True)
class NatureScene:
    frames:int
    layers:tuple[NatureLayer,...]
    peak_ceiling:float=.9
    room:float=.12

    def __post_init__(self):
        _int(self.frames,SAMPLE_RATE//10,MAX_FRAMES,"scene frames")
        if (type(self.layers) is not tuple or not 1<=len(self.layers)<=MAX_LAYERS
                or any(not isinstance(l,NatureLayer) or l.start+l.frames>self.frames for l in self.layers)):
            raise GenerationError("Invalid bounded nature scene layers")
        if sum(l.frames for l in self.layers)>MAX_LAYER_FRAMES:
            raise GenerationError("Nature scene render budget exceeded")
        _real(self.peak_ceiling,.05,.99,"scene peak")
        _real(self.room,0,1,"scene room")


def _edge(frames,attack,release):
    # Unlike the primitive safety envelope, honor exact user fade lengths.
    result=np.ones(frames,dtype=np.float64)
    if attack==1:
        result[0]=0
    elif attack:
        result[:attack]=np.sin(np.linspace(0,math.pi/2,attack))**2
    if release==1:
        result[-1]=0
    elif release:
        result[-release:]=np.cos(np.linspace(0,math.pi/2,release))**2
    return result


def render_layer(layer):
    if not isinstance(layer,NatureLayer):
        raise GenerationError("Expected validated nature layer")
    mono=render_voice(layer.voice,layer.frames).astype(np.float64)
    # Width uses a bounded decorrelated delay, not another source render.
    # Equal-power panning determines direct gain. For width=0 the two
    # channels are exact scaled copies of the original waveform.
    delay=min(211,layer.frames-1)
    delayed=np.zeros_like(mono)
    delayed[delay:]=mono[:-delay]
    side=(mono-delayed)*.35*layer.width
    pan=_curve(layer.pan,layer.frames)
    angle=(pan+1)*math.pi/4
    gain=np.power(10.,_curve(layer.gain,layer.frames)/20.)
    gain*=_edge(layer.frames,layer.fade_in,layer.fade_out)
    result=np.stack(((mono+side)*np.cos(angle)*gain,
                     (mono-side)*np.sin(angle)*gain))
    if not np.isfinite(result).all():
        raise GenerationError("Nonfinite nature stereo layer")
    return result


def _room(dry,amount):
    output=dry.copy()
    for delay,gain,cross in ((.037,.11,True),(.089,.07,False),(.157,.045,True)):
        shift=round(delay*SAMPLE_RATE)
        if shift<dry.shape[1]:
            output[:,shift:]+=(dry[::-1,:-shift] if cross else dry[:,:-shift])*gain*amount
    return output


def scene_digest(scene):
    if not isinstance(scene,NatureScene):
        raise GenerationError("Expected NatureScene")
    data=json.dumps(asdict(scene),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+data).hexdigest()


def render_scene(scene):
    if not isinstance(scene,NatureScene):
        raise GenerationError("Expected NatureScene")
    dry=np.zeros((2,scene.frames),dtype=np.float64)
    reports=[]
    for i,layer in enumerate(scene.layers):
        rendered=render_layer(layer)
        dry[:,layer.start:layer.start+layer.frames]+=rendered
        reports.append({"index":i,"kind":layer.voice.kind,"start":layer.start,"frames":layer.frames,
                        "peak":float(np.max(np.abs(rendered))),
                        "rms":float(np.sqrt(np.mean(rendered*rendered)))})
    wet=_room(dry,scene.room)
    wet*=_edge(scene.frames,min(240,scene.frames//2),min(1200,scene.frames//2))
    peak=float(np.max(np.abs(wet)))
    attenuation=min(1.,scene.peak_ceiling/max(peak,1e-15))
    samples=(wet*attenuation).astype(np.float32)
    if not np.isfinite(samples).all():
        raise GenerationError("Nonfinite nature scene mix")
    return samples,{"protocol":PROTOCOL,"scene_sha256":scene_digest(scene),
                    "sample_rate":SAMPLE_RATE,"channels":2,"frames":scene.frames,
                    "layers":reports,"pre_master_peak":peak,"master_attenuation":attenuation,
                    "metrics":metrics_channels(samples),"learned_model":False,
                    "external_recordings_used":False,"perceptual_quality_established":False}


def generate_scene_audio(scene):
    samples,report=render_scene(scene)
    audio=encode_wav_channels(samples,SAMPLE_RATE)
    report["wav_sha256"]=hashlib.sha256(audio).hexdigest()
    return audio,report


def preset(name,seconds=8.,seed=0):
    _real(seconds,1,30,"preset seconds")
    _int(seed,0,2**32-1,"preset seed")
    frames=round(seconds*SAMPLE_RATE)
    if name not in ("forest","storm","coast","creek","campfire"):
        raise GenerationError("Unknown nature scene preset")
    def layer(kind,index,level,pan,width,intensity,texture,motion,start=0):
        count=frames-start
        if count<SAMPLE_RATE//10:
            raise GenerationError("Preset layer duration too short")
        voice=NatureVoice(kind,intensity,texture,motion,(seed+index*2654435761)%(2**32))
        return NatureLayer(voice,start,count,(AutomationPoint(0,level),),
                           (AutomationPoint(0,pan),),width)
    if name=="forest":
        layers=(layer("wind",1,-9.,-.15,.65,.5,.45,.6),
                layer("birds",2,-3.,.3,.3,.7,.35,.8),
                layer("stream",3,-13.,-.45,.4,.4,.5,.5))
    elif name=="storm":
        layers=(layer("rain",1,-3.,0.,.8,.9,.8,.5),
                layer("wind",2,-7.,-.2,.6,.75,.65,.8),
                layer("thunder",3,1.,.15,.5,.9,.5,.6,frames//5))
    elif name=="coast":
        layers=(layer("ocean",1,-2.,0.,.8,.85,.8,.5),
                layer("wind",2,-11.,-.3,.6,.55,.6,.7))
    elif name=="creek":
        layers=(layer("stream",1,-2.,0.,.65,.85,.6,.7),
                layer("birds",2,-9.,.5,.35,.55,.4,.5))
    else:
        layers=(layer("fire",1,-2.,0.,.4,.75,.7,.4),
                layer("wind",2,-14.,-.3,.7,.35,.4,.5))
    return NatureScene(frames,layers)
