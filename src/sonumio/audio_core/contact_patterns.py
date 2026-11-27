"""Bounded seeded contact event patterns for original sound design.

The pattern generator is a scheduling tool, not a recording of a person
or a calibrated simulation of gait or physical motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import check_rate, finite_float
from .errors import AudioContractError, AudioLimitError
from .modal_synthesis import (
    ContactScene, ContactEvent, MATERIALS, EXCITATIONS, MAX_EVENTS,
    material_resonator,
)


@dataclass(frozen=True, slots=True)
class ContactPattern:
    material: str = "wood"
    excitation: str = "step"
    count: int = 8
    interval_ms: int = 500
    event_ms: int = 250
    initial_delay_ms: int = 0
    timing_jitter_ms: int = 0
    strength: float = .5
    strength_jitter: float = .15
    hardness: float = .5
    fundamental_hz: float = 240.
    size: float = 1.
    damping: float = 1.
    pitch_jitter: float = 1.
    alternating_pan: float = .25
    seed: int = 0
    sample_rate: int = 24000
    peak_ceiling: float | None = .9

    def __post_init__(self):
        if self.material not in MATERIALS or self.excitation not in EXCITATIONS:
            raise AudioContractError("Unknown contact material or excitation")
        check_rate(self.sample_rate)
        for name,lower,upper in (
            ("count",1,MAX_EVENTS),("interval_ms",20,10000),("event_ms",10,8000),
            ("initial_delay_ms",0,10000),("timing_jitter_ms",0,self.interval_ms//3),
            ("seed",0,2**32-1),
        ):
            value=getattr(self,name)
            if type(value) is not int or not lower<=value<=upper:
                raise AudioContractError("Invalid contact pattern "+name)
        for name,lower,upper in (
            ("strength",0,1),("strength_jitter",0,1),("hardness",0,1),
            ("pitch_jitter",0,12),("alternating_pan",0,1),
        ):
            finite_float(getattr(self,name),name,minimum=lower,maximum=upper)
        material_resonator(self.material,fundamental_hz=self.fundamental_hz,
                           size=self.size,damping=self.damping)
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling,"peak ceiling",minimum=.001,maximum=1.)
        if self.initial_delay_ms+(self.count-1)*self.interval_ms+self.timing_jitter_ms+self.event_ms>120000:
            raise AudioLimitError("Contact pattern exceeds maximum duration")


def pattern_scene(pattern):
    """Compute each event on the same integer clock, without cumulative drift."""
    if not isinstance(pattern,ContactPattern):
        raise AudioContractError("Expected ContactPattern")
    profile=material_resonator(pattern.material,fundamental_hz=pattern.fundamental_hz,
                               size=pattern.size,damping=pattern.damping)
    rate=pattern.sample_rate
    frames=(pattern.event_ms*rate+500)//1000
    events=[]
    for index in range(pattern.count):
        rng=np.random.default_rng(np.random.SeedSequence([pattern.seed,index,0x50415454]))
        timing,strength,pitch=rng.uniform(-1,1,3)
        seed=int(rng.integers(0,2**32,dtype=np.uint64))
        nominal=pattern.initial_delay_ms+index*pattern.interval_ms
        # Bound the first event at frame zero. Others remain strictly ordered.
        start=max(0,round((nominal+timing*pattern.timing_jitter_ms)*rate/1000))
        level=float(np.clip(pattern.strength*(1+strength*pattern.strength_jitter),0,1))
        events.append(ContactEvent(start,frames,profile,excitation=pattern.excitation,
            strength=level,hardness=pattern.hardness,
            pitch_semitones=float(pitch*pattern.pitch_jitter),
            pan=pattern.alternating_pan*(-1 if index%2==0 else 1),seed=seed,
            fade_out_frames=min(frames,round(rate*.008))))
    end=max(e.start_frame+e.frames for e in events)
    return ContactScene(rate,end,tuple(events),peak_ceiling=pattern.peak_ceiling)
