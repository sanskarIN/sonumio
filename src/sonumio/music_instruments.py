"""Original instrument synthesis for the arrangement engine.

All pitched voices are finite additive/modal models with explicit
bandwidth limits. No sample library, external recordings or AI weights.
"""
from __future__ import annotations

import math
import numpy as np

from .audio import SAMPLE_RATE
from .arrangement import Event, integer
from .errors import GenerationError

MAX_NOTE_FRAMES = SAMPLE_RATE*4
NYQUIST_GUARD = SAMPLE_RATE*.46


def smooth_edges(frames,attack,release):
    out=np.ones(frames,dtype=np.float64)
    a=min(attack,frames//2)
    r=min(release,frames-a)
    if a:
        out[:a]=np.sin(np.linspace(0,math.pi/2,a,endpoint=False))**2
    if r==1:
        out[-1]=0.
    elif r:
        out[-r:]=np.cos(np.linspace(0,math.pi/2,r))**2
    return out


def _clock(frames):
    return np.arange(frames,dtype=np.float64)/SAMPLE_RATE


def _additive(t,frequency,partials,decays,*,inharmonicity=0.,vibrato=0.):
    signal=np.zeros(len(t),dtype=np.float64)
    normalizer=0.
    # Sinusoidal FM phase modulation has bounded small deviation; the
    # conservative frequency guard leaves headroom but is not a proof of
    # strict bandlimiting for its infinite theoretical sidebands.
    modulation=vibrato*np.sin(2*math.pi*5.1*t)
    for index,(amplitude,decay) in enumerate(zip(partials,decays,strict=True),1):
        freq=frequency*index*math.sqrt(1+inharmonicity*index*index)
        if freq>=NYQUIST_GUARD:
            continue
        envelope=np.exp(-t/decay)
        phase=2*math.pi*freq*t+index*modulation
        signal+=amplitude*envelope*np.sin(phase)
        normalizer+=abs(amplitude)
    if normalizer==0:
        raise GenerationError("Instrument has no representable partials")
    return signal/normalizer


def _keys(t,f,v):
    # Bright hammer transient followed by slower-decaying lower modes.
    body=_additive(t,f,(1.,.57,.32,.18,.1,.06,.04,.02),
                   (2.4,1.5,.9,.55,.34,.23,.16,.11),inharmonicity=.00012)
    transient=_additive(t,f,(0.,.1,.19,.2,.13,.06),(.03,)*6)
    return (body+.12*v*transient)*smooth_edges(len(t),72,1800)


def _pad(t,f,v):
    base=_additive(t,f,(1.,.43,.24,.12,.07),(12.,9.,8.,6.,5.),vibrato=.045)
    # Gentle detuned second voice, no random phase drift.
    upper=_additive(t,f*2**(4/1200),(1.,.35,.15),(10.,7.,5.),vibrato=.023)
    env=smooth_edges(len(t),3600,5400)
    return (.72*base+.28*upper)*env*(.92+.08*np.cos(2*math.pi*.8*t))


def _bass(t,f,v):
    body=_additive(t,f,(1.,.42,.19,.08), (3.,.9,.45,.25))
    return body*smooth_edges(len(t),120,1200)


def _lead(t,f,v):
    body=_additive(t,f,(1.,.5,.31,.14,.07,.03), (6.,4.,3.,2.,1.,.7),vibrato=.08)
    return body*smooth_edges(len(t),240,1440)


def _pluck(t,f,v):
    # Inharmonic, quickly damped modes approximate a bright plucked string.
    body=_additive(t,f,(1.,.67,.41,.27,.13,.08,.04,.025),
                   (.65,.45,.32,.23,.14,.1,.075,.05),inharmonicity=.00004)
    return body*smooth_edges(len(t),48,840)


def _kick(t,rng):
    phase=2*math.pi*(43*t+85*.022*(1-np.exp(-t/.022)))
    body=np.sin(phase)*np.exp(-t/.085)
    click=np.diff(rng.standard_normal(len(t)+1))*.04*np.exp(-t/.006)
    return (body+click)*smooth_edges(len(t),24,360)


def _snare(t,rng):
    noise=rng.standard_normal(len(t)+2)
    high=noise[2:]-2*noise[1:-1]+noise[:-2]
    body=.25*np.sin(2*math.pi*175*t)*np.exp(-t/.045)
    ring=.12*np.sin(2*math.pi*330*t)*np.exp(-t/.025)
    return (body+ring+.15*high*np.exp(-t/.075))*smooth_edges(len(t),24,360)


def _hat(t,rng):
    noise=np.diff(rng.standard_normal(len(t)+1))
    metallic=sum(np.sin(2*math.pi*f*t) for f in (3171,4219,5981,7877))*.06
    return (.24*noise+metallic)*np.exp(-t/.032)*smooth_edges(len(t),12,180)


PITCHED_RENDERERS={"keys":_keys,"pad":_pad,"bass":_bass,"lead":_lead,"pluck":_pluck}
DRUM_RENDERERS={"kick":_kick,"snare":_snare,"hat":_hat}


def render_event(event,frames,seed):
    if not isinstance(event,Event):
        raise GenerationError("Expected validated instrument event")
    integer(frames,2,MAX_NOTE_FRAMES,"instrument frames")
    integer(seed,0,2**32-1,"instrument seed")
    t=_clock(frames)
    if event.track in PITCHED_RENDERERS:
        f=440*2**((event.pitch-69)/12)
        mono=PITCHED_RENDERERS[event.track](t,f,event.velocity)
    else:
        # Seed per event: mute/solo and event reordering cannot change
        # another percussion event's noise realization.
        rng=np.random.default_rng(np.random.SeedSequence(
            [seed,event.tick,event.pitch,0x4452554D]))
        mono=DRUM_RENDERERS[event.track](t,rng)
    mono=np.asarray(mono,dtype=np.float64)*event.velocity
    if mono.shape!=(frames,) or not np.isfinite(mono).all():
        raise GenerationError("Invalid instrument waveform")
    # Bound pathological noise peaks without boosting quiet voices.
    peak=float(np.max(np.abs(mono)))
    if peak>1:
        mono/=peak
    return mono
