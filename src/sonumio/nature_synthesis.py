"""Bounded, original procedural nature sound primitives.

All sources are mathematical signals or locally seeded noise. The module
does not contain recordings, learned weights, external sample packs or
voice synthesis. Rendering is at the original 24 kHz sample clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from .audio import SAMPLE_RATE
from .errors import GenerationError

KINDS=("rain","wind","ocean","thunder","birds","stream","fire")
MAX_SECONDS=30
MAX_FRAMES=MAX_SECONDS*SAMPLE_RATE
EPS=1e-12


def _int(value,lo,hi,name):
    if type(value) is not int or not lo<=value<=hi:
        raise GenerationError("Invalid nature sound "+name)
    return value


def _real(value,lo,hi,name):
    if type(value) not in (int,float) or not lo<=value<=hi:
        raise GenerationError("Invalid nature sound "+name)
    return float(value)


@dataclass(frozen=True,slots=True)
class NatureVoice:
    kind:str
    intensity:float=.5
    texture:float=.5
    motion:float=.5
    seed:int=0

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise GenerationError("Unknown nature sound kind")
        _real(self.intensity,0,1,"intensity")
        _real(self.texture,0,1,"texture")
        _real(self.motion,0,1,"motion")
        _int(self.seed,0,2**32-1,"seed")


def _rng(voice,domain):
    return np.random.default_rng(np.random.SeedSequence([voice.seed,KINDS.index(voice.kind),domain]))


def _clock(frames):
    return np.arange(frames,dtype=np.float64)/SAMPLE_RATE


def _noise(rng,frames,low,high):
    """Finite FFT-domain band-shaped noise with smooth transition bands.

    Generate real PCM directly on the requested sample clock. DC is zero
    and no bins above Nyquist are available. Normalize RMS, not peak.
    The finite periodic FFT basis is subsequently edge-enveloped.
    """
    if not 0<=low<high<=SAMPLE_RATE/2:
        raise GenerationError("Invalid nature noise passband")
    x=rng.standard_normal(frames)
    freq=np.fft.rfftfreq(frames,1/SAMPLE_RATE)
    # Smooth shoulders avoid an abrupt rectangular spectral cutoff.
    lo_width=max(20.,low*.25)
    hi_width=max(50.,high*.15)
    lower=np.clip((freq-low)/lo_width,0,1) if low else np.ones_like(freq)
    upper=np.clip((high-freq)/hi_width,0,1)
    weight=np.sin(lower*math.pi/2)**2*np.sin(upper*math.pi/2)**2
    weight[0]=0
    values=np.fft.irfft(np.fft.rfft(x)*weight,n=frames)
    return values/max(float(np.sqrt(np.mean(values*values))),EPS)


def _slow_curve(rng,frames,rate):
    """Interpolated bounded modulation on an independent slow clock."""
    seconds=frames/SAMPLE_RATE
    knots=max(2,math.ceil(seconds*rate)+1)
    value=rng.uniform(-1,1,size=knots)
    position=np.linspace(0,knots-1,frames)
    left=np.floor(position).astype(np.int64)
    right=np.minimum(left+1,knots-1)
    fraction=position-left
    weight=fraction*fraction*(3-2*fraction)
    return value[left]*(1-weight)+value[right]*weight


def _envelope(frames,attack,release):
    y=np.ones(frames,dtype=np.float64)
    a=min(attack,frames//2)
    r=min(release,frames-a)
    if a:
        y[:a]=np.sin(np.linspace(0,math.pi/2,a,endpoint=False))**2
    if r==1:
        y[-1]=0
    elif r:
        y[-r:]=np.cos(np.linspace(0,math.pi/2,r))**2
    return y


def _event_positions(rng,frames,events_per_second,*,maximum=8000):
    count=min(maximum,int(rng.poisson(events_per_second*frames/SAMPLE_RATE)))
    if not count:
        return np.empty(0,dtype=np.int64)
    return np.sort(rng.integers(0,frames,size=count))


def _add(out,start,signal):
    stop=min(len(out),start+len(signal))
    if stop>start:
        out[start:stop]+=signal[:stop-start]


def _rain(voice,frames):
    t=_clock(frames)
    low=_noise(_rng(voice,1),frames,180,3800)
    high=_noise(_rng(voice,2),frames,2200,10800)
    variation=.85+.15*_slow_curve(_rng(voice,3),frames,.3+1.2*voice.motion)
    output=(.035*low+(.012+.045*voice.texture)*high)*variation
    rng=_rng(voice,4)
    positions=_event_positions(rng,frames,18+220*voice.intensity)
    for start in positions:
        length=int(rng.integers(40,200))
        local=_clock(length)
        frequency=float(rng.uniform(1600,7200))
        decay=float(rng.uniform(.001,.005))
        pulse=(np.sin(2*math.pi*frequency*local)+.3*rng.standard_normal(length))*np.exp(-local/decay)
        _add(output,int(start),pulse*float(rng.uniform(.015,.055))*(.5+voice.texture))
    return output


def _wind(voice,frames):
    t=_clock(frames)
    low=_noise(_rng(voice,1),frames,35,600+1200*voice.texture)
    airy=_noise(_rng(voice,2),frames,450,5200)
    gust=_slow_curve(_rng(voice,3),frames,.25+1.5*voice.motion)
    modulation=np.clip(.65+.32*gust+.13*np.sin(2*math.pi*.17*t),.1,1.2)
    whistle_frequency=190+180*voice.texture+40*_slow_curve(_rng(voice,4),frames,.5)
    phase=2*math.pi*np.cumsum(whistle_frequency)/SAMPLE_RATE
    whistle=.008*voice.texture*np.sin(phase)*modulation**2
    return (.065*low+.018*voice.texture*airy)*modulation+whistle


def _ocean(voice,frames):
    t=_clock(frames)
    period=3.8+3.5*(1-voice.motion)
    phase=2*math.pi*t/period
    swell=(.5+.5*np.sin(phase-.8))**2
    foam=(.5+.5*np.sin(phase-1.5))**5
    deep=_noise(_rng(voice,1),frames,45,850)
    surf=_noise(_rng(voice,2),frames,320,4800)
    fizz=_noise(_rng(voice,3),frames,2400,10500)
    irregular=.8+.2*_slow_curve(_rng(voice,4),frames,.6)
    return irregular*(.075*deep*(.3+.7*swell)+.045*surf*swell+.025*voice.texture*fizz*foam)


def _thunder(voice,frames):
    t=_clock(frames)
    rng=_rng(voice,1)
    center=min(max(.15,frames/SAMPLE_RATE*.22),frames/SAMPLE_RATE*.75)
    seconds=np.maximum(0,t-center)
    attack=np.clip(seconds/.045,0,1)
    decay=np.exp(-seconds/(.55+1.9*voice.motion))
    rumble=_noise(_rng(voice,2),frames,28,460+480*voice.texture)
    output=.12*rumble*attack*decay*(t>=center)
    for i in range(1+int(3*voice.texture)):
        position=center+i*.11+float(rng.uniform(0,.035))
        d=np.maximum(0,t-position)
        pulse=np.clip(d/.008,0,1)*np.exp(-d/(.055+.04*i))*(t>=position)
        band=_noise(_rng(voice,10+i),frames,160,5000)
        output+=.025*band*pulse/(i+1)
    return output


def _birds(voice,frames):
    output=np.zeros(frames,dtype=np.float64)
    rng=_rng(voice,1)
    seconds=frames/SAMPLE_RATE
    count=max(1,min(100,int(round(seconds*(.6+2*voice.intensity)))))
    positions=np.linspace(.08,max(.08,seconds-.22),count)
    for i,time in enumerate(positions):
        time=max(0,min(seconds-.05,float(time+rng.uniform(-.07,.07))))
        start=round(time*SAMPLE_RATE)
        syllables=2+int(rng.integers(3))
        f0=float(rng.uniform(1100,2400))
        direction=-1 if i%3==0 else 1
        for j in range(syllables):
            length=int(rng.integers(1000,2600))
            t=_clock(length)
            sweep=direction*(900+2400*voice.motion)
            phase=2*math.pi*((f0+120*j)*t+.5*sweep*t*t)
            chirp=np.sin(phase)+.13*voice.texture*np.sin(2*phase)
            envelope=np.sin(np.linspace(0,math.pi,length))**2
            _add(output,start+j*int(.07*SAMPLE_RATE),chirp*envelope*.11/(1+.13*j))
    return output


def _stream(voice,frames):
    t=_clock(frames)
    water=_noise(_rng(voice,1),frames,220,4400)
    treble=_noise(_rng(voice,2),frames,1600,9600)
    modulation=.65+.25*_slow_curve(_rng(voice,3),frames,1.2+voice.motion*2)
    output=(.045*water+.017*voice.texture*treble)*modulation
    rng=_rng(voice,4)
    for start in _event_positions(rng,frames,10+65*voice.intensity,maximum=4000):
        length=int(rng.integers(180,950))
        local=_clock(length)
        f=float(rng.uniform(280,1400))
        phase=2*math.pi*(f*local+600*local*local)
        pulse=np.sin(phase)*np.exp(-local/.012)*_envelope(length,18,50)
        _add(output,int(start),pulse*float(rng.uniform(.008,.03)))
    return output


def _fire(voice,frames):
    low=_noise(_rng(voice,1),frames,70,1400)
    high=_noise(_rng(voice,2),frames,1200,10000)
    output=.024*low+.012*high
    rng=_rng(voice,3)
    for start in _event_positions(rng,frames,4+44*voice.intensity,maximum=3000):
        length=int(rng.integers(25,240))
        t=_clock(length)
        crackle=rng.standard_normal(length)*np.exp(-t/(.0007+.002*voice.texture))
        _add(output,int(start),crackle*float(rng.uniform(.035,.13)))
    return output


_RENDERERS={"rain":_rain,"wind":_wind,"ocean":_ocean,"thunder":_thunder,
            "birds":_birds,"stream":_stream,"fire":_fire}


def render_voice(voice,frames):
    if not isinstance(voice,NatureVoice):
        raise GenerationError("Expected NatureVoice")
    _int(frames,SAMPLE_RATE//10,MAX_FRAMES,"frames")
    if voice.intensity==0:
        return np.zeros(frames,dtype=np.float32)
    result=_RENDERERS[voice.kind](voice,frames)
    result*=voice.intensity
    result*=_envelope(frames,round(.015*SAMPLE_RATE),round(.035*SAMPLE_RATE))
    if result.shape!=(frames,) or not np.isfinite(result).all():
        raise GenerationError("Invalid nature sound waveform")
    peak=float(np.max(np.abs(result)))
    if peak>.8:
        result*=.8/peak
    return result.astype(np.float32)
