"""Declarative modal percussion synthesis with bounded deterministic excitation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrangement import Event, TRACKS, PITCHED, integer, real
from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music_instruments import MAX_NOTE_FRAMES, NYQUIST_GUARD
from .music_patch_synthesis import NoiseLayer, Envelope, envelope_samples, _noise

DRUMS=TRACKS[len(PITCHED):]
MAX_MODES=12
MAX_NOISE_LAYERS=3


@dataclass(frozen=True,slots=True)
class DrumMode:
    start_hz:float
    end_hz:float
    sweep_seconds:float
    amplitude:float
    decay_seconds:float
    phase_cycles:float=0.

    def __post_init__(self):
        for name,lo,hi in (
            ("start_hz",20.,NYQUIST_GUARD-1.),("end_hz",20.,NYQUIST_GUARD-1.),
            ("sweep_seconds",.005,2.),("amplitude",0.,1.),
            ("decay_seconds",.005,10.),("phase_cycles",0.,1.),
        ):
            object.__setattr__(self,name,real(getattr(self,name),lo,hi,"drum "+name))

    @property
    def static(self):
        return self.start_hz==self.end_hz


@dataclass(frozen=True,slots=True)
class DrumNoise:
    layer:NoiseLayer
    attack_frames:int=1

    def __post_init__(self):
        if not isinstance(self.layer,NoiseLayer) or self.layer.gain<=0:
            raise GenerationError("Expected sounding drum noise layer")
        integer(self.attack_frames,1,2400,"drum noise attack")


@dataclass(frozen=True,slots=True)
class DrumPatch:
    modes:tuple[DrumMode,...]
    noise:tuple[DrumNoise,...]=()
    attack_frames:int=24
    release_frames:int=360
    velocity_exponent:float=1.

    def __post_init__(self):
        if (type(self.modes) is not tuple or len(self.modes)>MAX_MODES
                or any(not isinstance(mode,DrumMode) for mode in self.modes)
                or type(self.noise) is not tuple or len(self.noise)>MAX_NOISE_LAYERS
                or any(not isinstance(layer,DrumNoise) for layer in self.noise)):
            raise GenerationError("Invalid drum patch layers")
        budget=math.fsum([*(mode.amplitude for mode in self.modes),
                          *(noise.layer.gain for noise in self.noise)])
        if not 0.<budget<=1.:
            raise GenerationError("Drum patch layer amplitude budget must be in (0,1]")
        integer(self.attack_frames,1,4800,"drum attack")
        integer(self.release_frames,1,12000,"drum release")
        object.__setattr__(self,"velocity_exponent",real(self.velocity_exponent,.25,3.,"drum velocity exponent"))


def phase_cycles(clock,mode):
    """Analytic integral of an exponential glide, continuous at sample zero."""
    if (not isinstance(clock,np.ndarray) or clock.dtype!=np.float64 or clock.ndim!=1
            or not 1<=len(clock)<=MAX_NOTE_FRAMES or not np.isfinite(clock).all()
            or np.any(clock<0.) or np.any(clock>MAX_NOTE_FRAMES/SAMPLE_RATE)
            or not isinstance(mode,DrumMode)):
        raise GenerationError("Invalid drum phase clock")
    if mode.static:
        return mode.start_hz*clock+mode.phase_cycles
    # expm1 avoids cancellation at the beginning of a slow sweep.
    return (mode.end_hz*clock-(mode.start_hz-mode.end_hz)*mode.sweep_seconds
            *np.expm1(-clock/mode.sweep_seconds)+mode.phase_cycles)


def drum_cost(event,frames,patch):
    if not isinstance(event,Event) or event.track not in DRUMS or not isinstance(patch,DrumPatch):
        raise GenerationError("Expected percussion event and drum patch")
    integer(frames,2,MAX_NOTE_FRAMES,"drum frames")
    return frames*(4+sum(2 if mode.static else 4 for mode in patch.modes if mode.amplitude)
                   +sum(noise.layer.taps+4 for noise in patch.noise))


def render_drum_patch(event,frames,seed,patch):
    drum_cost(event,frames,patch)
    integer(seed,0,2**32-1,"drum seed")
    clock=np.arange(frames,dtype=np.float64)/SAMPLE_RATE
    signal=np.zeros(frames,dtype=np.float64)
    for mode in patch.modes:
        if mode.amplitude:
            signal+=mode.amplitude*np.sin(2*math.pi*phase_cycles(clock,mode))*np.exp(-clock/mode.decay_seconds)
    for index,noise in enumerate(patch.noise):
        # The existing deterministic event noise generator is domain
        # separated by a reversible seed translation for each layer.
        layer_seed=(seed+0x9E3779B9*(index+1))%(2**32)
        values=_noise(event,frames,layer_seed,noise.layer)
        ramp=np.minimum(1.,np.arange(frames,dtype=np.float64)/noise.attack_frames)
        signal+=noise.layer.gain*values*ramp*np.exp(-clock/noise.layer.decay_seconds)
    envelope=Envelope(patch.attack_frames,0,1.,patch.release_frames)
    signal*=envelope_samples(envelope,frames)*event.velocity**patch.velocity_exponent
    if signal.shape!=(frames,) or not np.isfinite(signal).all() or np.max(np.abs(signal))>1.000001:
        raise GenerationError("Invalid bounded percussion patch waveform")
    return signal
