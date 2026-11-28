"""Bounded declarative additive sound design for original pitched instruments.

Patch parameters are data, never executable code, external audio, plugins,
or model weights. Every note uses an absolute sample clock, bounded
amplitude and a conservative oscillator frequency guard.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import struct

import numpy as np

from .arrangement import Event, PITCHED, TRACKS, integer, real
from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music_instruments import MAX_NOTE_FRAMES, NYQUIST_GUARD

MAX_PARTIALS=16
MAX_BANKS=4
NOISE_COLORS=("white","low","high")


def _real(value,lo,hi,label):
    return real(value,lo,hi,"patch "+label)


@dataclass(frozen=True,slots=True)
class Partial:
    harmonic:int
    amplitude:float
    decay_seconds:float=0.
    phase_cycles:float=0.

    def __post_init__(self):
        integer(self.harmonic,1,MAX_PARTIALS,"patch harmonic")
        object.__setattr__(self,"amplitude",_real(self.amplitude,0.,1.,"partial amplitude"))
        object.__setattr__(self,"decay_seconds",_real(self.decay_seconds,0.,30.,"partial decay"))
        if 0.<self.decay_seconds<.005:
            raise GenerationError("Nonzero patch decay must be at least 5 ms")
        object.__setattr__(self,"phase_cycles",_real(self.phase_cycles,0.,1.,"partial phase"))


@dataclass(frozen=True,slots=True)
class OscillatorBank:
    partials:tuple[Partial,...]
    gain:float=1.
    detune_cents:float=0.
    octave:int=0
    inharmonicity:float=0.
    vibrato_rate_hz:float=5.
    vibrato_depth_cycles:float=0.

    def __post_init__(self):
        if (type(self.partials) is not tuple or not 1<=len(self.partials)<=MAX_PARTIALS
                or any(not isinstance(p,Partial) for p in self.partials)):
            raise GenerationError("Invalid patch oscillator partials")
        harmonics=tuple(p.harmonic for p in self.partials)
        if harmonics!=tuple(sorted(set(harmonics))):
            raise GenerationError("Patch harmonics must be unique and increasing")
        if not 0.<math.fsum(p.amplitude for p in self.partials)<=1.:
            raise GenerationError("Patch partial amplitude budget must be in (0,1]")
        for name,lo,hi in (
            ("gain",0.,1.),("detune_cents",-100.,100.),("inharmonicity",0.,.003),
            ("vibrato_rate_hz",.1,12.),("vibrato_depth_cycles",0.,.1),
        ):
            object.__setattr__(self,name,_real(getattr(self,name),lo,hi,name))
        integer(self.octave,-2,2,"patch oscillator octave")


@dataclass(frozen=True,slots=True)
class Envelope:
    attack_frames:int=120
    decay_frames:int=1200
    sustain:float=.8
    release_frames:int=1200

    def __post_init__(self):
        integer(self.attack_frames,1,24000,"patch attack")
        integer(self.decay_frames,0,48000,"patch envelope decay")
        integer(self.release_frames,1,48000,"patch release")
        object.__setattr__(self,"sustain",_real(self.sustain,0.,1.,"sustain"))


@dataclass(frozen=True,slots=True)
class NoiseLayer:
    gain:float=0.
    color:str="white"
    taps:int=8
    decay_seconds:float=.08

    def __post_init__(self):
        object.__setattr__(self,"gain",_real(self.gain,0.,1.,"noise gain"))
        if type(self.color) is not str or self.color not in NOISE_COLORS:
            raise GenerationError("Unknown patch noise color")
        integer(self.taps,2,32,"patch noise taps")
        object.__setattr__(self,"decay_seconds",_real(self.decay_seconds,.005,30.,"noise decay"))


@dataclass(frozen=True,slots=True)
class InstrumentPatch:
    banks:tuple[OscillatorBank,...]
    envelope:Envelope=Envelope()
    noise:NoiseLayer=NoiseLayer()
    velocity_exponent:float=1.

    def __post_init__(self):
        if (type(self.banks) is not tuple or not 1<=len(self.banks)<=MAX_BANKS
                or any(not isinstance(b,OscillatorBank) for b in self.banks)
                or not isinstance(self.envelope,Envelope) or not isinstance(self.noise,NoiseLayer)):
            raise GenerationError("Invalid instrument patch components")
        budget=math.fsum([self.noise.gain,*(bank.gain for bank in self.banks)])
        if not 0.<budget<=1.:
            raise GenerationError("Patch layer gain budget must be in (0,1]")
        if not any(bank.gain>0 for bank in self.banks):
            raise GenerationError("Pitched patch requires a sounding oscillator bank")
        object.__setattr__(self,"velocity_exponent",_real(self.velocity_exponent,.25,3.,"velocity exponent"))


def envelope_samples(envelope,frames):
    """Continuous bounded envelope with exact zero first and last sample.

    Attack and release share the available note length when both requested
    durations cannot fit. Decay is truncated, not restarted, by note release.
    """
    if not isinstance(envelope,Envelope):
        raise GenerationError("Expected validated patch envelope")
    integer(frames,2,MAX_NOTE_FRAMES,"patch note frames")
    available=frames-1
    requested=envelope.attack_frames+envelope.release_frames
    if requested>available:
        # Preserve one sample step for release. Two-frame notes are
        # necessarily silent under the zero-endpoint contract.
        attack=max(1,min(available-1,available*envelope.attack_frames//requested))
        release=max(1,available-attack)
    else:
        attack=envelope.attack_frames
        release=envelope.release_frames
    clock=np.arange(frames,dtype=np.float64)
    rising=np.minimum(1.,clock/attack)
    falling=np.minimum(1.,(available-clock)/release)
    result=np.sin(rising*math.pi/2)**2*np.sin(falling*math.pi/2)**2
    if envelope.decay_frames:
        progress=np.clip((clock-attack)/envelope.decay_frames,0.,1.)
        result*=1.-(1.-envelope.sustain)*progress
    else:
        # No decay means sustain begins after the attack. Scaling
        # the attack avoids a discontinuity at that boundary.
        result*=envelope.sustain
    result[0]=result[-1]=0.
    return result


def partial_frequency(pitch,bank,partial):
    integer(pitch,24,96,"patch pitch")
    if not isinstance(bank,OscillatorBank) or not isinstance(partial,Partial):
        raise GenerationError("Expected validated patch oscillator controls")
    frequency=440.*2**((pitch-69)/12+bank.octave+bank.detune_cents/1200.)
    return frequency*partial.harmonic*math.sqrt(1.+bank.inharmonicity*partial.harmonic**2)


def active_partials(pitch,bank):
    result=[]
    for partial in bank.partials:
        frequency=partial_frequency(pitch,bank,partial)
        # Bound instantaneous FM deviation and leave the existing guard
        # margin. Infinite-order FM sidebands are not strictly bandlimited.
        deviation=partial.harmonic*bank.vibrato_depth_cycles*bank.vibrato_rate_hz
        if partial.amplitude>0 and frequency+deviation<NYQUIST_GUARD:
            result.append((partial,frequency))
    return tuple(result)


def patch_cost(event,frames,patch):
    if not isinstance(event,Event) or event.track not in PITCHED or not isinstance(patch,InstrumentPatch):
        raise GenerationError("Expected pitched event and patch")
    integer(frames,2,MAX_NOTE_FRAMES,"patch note frames")
    count=sum(len(active_partials(event.pitch,bank)) for bank in patch.banks if bank.gain)
    if count==0:
        raise GenerationError("Patch has no representable partial at event pitch")
    return frames*(count+(patch.noise.taps+1 if patch.noise.gain else 0)+4)


def _noise(event,frames,seed,layer):
    # A domain-separated seed is independent of note order, track mute,
    # the patch's oscillator controls and other instrument notes.
    trace=hashlib.sha256(b"sonumio-patch-noise-v1\0")
    trace.update(struct.pack("<IIII",seed,TRACKS.index(event.track),event.pitch,event.tick))
    rng=np.random.default_rng(np.random.SeedSequence(
        np.frombuffer(trace.digest(),dtype="<u4").tolist()))
    noise=rng.uniform(-1.,1.,frames+layer.taps-1)
    if layer.color=="white":
        return noise[layer.taps-1:]
    low=np.convolve(noise,np.full(layer.taps,1./layer.taps,dtype=np.float64),mode="valid")
    if layer.color=="low":
        return low
    return (noise[layer.taps-1:]-low)*.5


def render_patch(event,frames,seed,patch):
    integer(seed,0,2**32-1,"patch seed")
    patch_cost(event,frames,patch)
    clock=np.arange(frames,dtype=np.float64)/SAMPLE_RATE
    signal=np.zeros(frames,dtype=np.float64)
    for bank in patch.banks:
        if not bank.gain:
            continue
        modulation=bank.vibrato_depth_cycles*np.sin(2*math.pi*bank.vibrato_rate_hz*clock)
        for partial,frequency in active_partials(event.pitch,bank):
            phase=2*math.pi*(frequency*clock+partial.phase_cycles+partial.harmonic*modulation)
            amplitude=bank.gain*partial.amplitude
            wave=amplitude*np.sin(phase)
            if partial.decay_seconds:
                wave*=np.exp(-clock/partial.decay_seconds)
            signal+=wave
    if patch.noise.gain:
        signal+=patch.noise.gain*_noise(event,frames,seed,patch.noise)*np.exp(-clock/patch.noise.decay_seconds)
    signal*=envelope_samples(patch.envelope,frames)*event.velocity**patch.velocity_exponent
    if signal.shape!=(frames,) or not np.isfinite(signal).all() or np.max(np.abs(signal))>1.000001:
        raise GenerationError("Invalid bounded instrument patch waveform")
    return signal
