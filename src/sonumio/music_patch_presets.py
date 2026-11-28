"""Small hand-authored first-party sound-design presets.

These describe oscillator mathematics only. The preset names do not
refer to third-party synthesizers, musicians or recordings.
"""
from __future__ import annotations

from .arrangement import PITCHED,arrangement_digest
from .errors import GenerationError
from .music_patch_synthesis import InstrumentPatch,OscillatorBank,Partial,Envelope,NoiseLayer
from .music_patch_bank import PatchBank,PatchRoute

NAMES=("glass","warm","reed","spark","sub","air")


def _bank(amplitudes,decays,**kwargs):
    return OscillatorBank(tuple(Partial(i,a,d) for i,(a,d) in enumerate(zip(amplitudes,decays,strict=True),1)),**kwargs)


def preset(name):
    if type(name) is not str or name not in NAMES:
        raise GenerationError("Unknown original instrument patch preset")
    if name=="glass":
        return InstrumentPatch((
            _bank((.53,.12,.18,.08,.06), (3.,.8,1.8,.4,.25),gain=.78,inharmonicity=.0007),
            _bank((.55,.18,.12,.08), (1.5,.6,.35,.2),gain=.18,octave=1,detune_cents=5.),
        ),Envelope(48,3600,.45,1500),NoiseLayer(.04,"high",6,.025))
    if name=="warm":
        return InstrumentPatch((
            _bank((.56,.24,.1,.05), (12.,10.,6.,4.),gain=.64,vibrato_depth_cycles=.008),
            _bank((.68,.2,.07), (15.,8.,6.),gain=.36,detune_cents=-6.,vibrato_rate_hz=4.2,vibrato_depth_cycles=.012),
        ),Envelope(1800,8400,.85,4200))
    if name=="reed":
        return InstrumentPatch((
            OscillatorBank((Partial(1,.44,8.),Partial(3,.26,6.),Partial(5,.13,3.),
                            Partial(7,.07,2.)),gain=.92,vibrato_rate_hz=5.7,vibrato_depth_cycles=.02),
        ),Envelope(360,2400,.8,1400),NoiseLayer(.08,"high",12,.15),.85)
    if name=="spark":
        return InstrumentPatch((
            _bank((.36,.22,.16,.12,.07,.04),(.7,.3,.2,.14,.09,.05),gain=.85,inharmonicity=.0003),
        ),Envelope(24,1500,.3,720),NoiseLayer(.15,"white",4,.018),1.2)
    if name=="sub":
        return InstrumentPatch((
            _bank((.88,.08,.03),(8.,2.,.7),gain=.7,octave=-1),
            _bank((.7,.2,.08),(3.,1.,.4),gain=.3),
        ),Envelope(96,2400,.75,1400))
    return InstrumentPatch((
        _bank((.68,.13,.08,.04),(14.,8.,4.,3.),gain=.53,detune_cents=7.,vibrato_depth_cycles=.014),
        _bank((.75,.16,.05),(14.,6.,3.),gain=.37,detune_cents=-7.,vibrato_depth_cycles=.011),
    ),Envelope(3200,6000,.72,5800),NoiseLayer(.1,"low",24,2.))


def preset_bank(score,assignments):
    if type(assignments) is not dict or not set(assignments)<=set(PITCHED):
        raise GenerationError("Invalid original preset assignments")
    return PatchBank(arrangement_digest(score),tuple(PatchRoute(track,preset(assignments[track]))
                    for track in PITCHED if track in assignments))
