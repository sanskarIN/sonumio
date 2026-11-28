"""Original modal percussion presets, without samples or third-party names."""
from __future__ import annotations

from .arrangement import arrangement_digest
from .errors import GenerationError
from .music_patch_synthesis import NoiseLayer
from .music_drum_patches import DRUMS,DrumMode,DrumNoise,DrumPatch
from .music_drum_bank import DrumBank,DrumRoute

PRESETS={"kick":("round","tight","deep"),"snare":("brush","snap","ring"),
         "hat":("soft","bright","metal")}


def _mode(start,end,amplitude,decay,sweep=.03):
    return DrumMode(start,end,sweep,amplitude,decay)


def _noise(gain,color,taps,decay,attack=1):
    return DrumNoise(NoiseLayer(gain,color,taps,decay),attack)


def drum_preset(track,name):
    if type(track) is not str or type(name) is not str or track not in PRESETS or name not in PRESETS[track]:
        raise GenerationError("Unknown original percussion preset")
    if track=="kick":
        if name=="round":
            return DrumPatch((_mode(116,48,.8,.15),_mode(180,96,.12,.05)),
                             (_noise(.08,"low",8,.018),),24,480)
        if name=="tight":
            return DrumPatch((_mode(150,60,.75,.055,.012),_mode(300,110,.15,.025,.009)),
                             (_noise(.1,"high",6,.008),),12,240)
        return DrumPatch((_mode(95,34,.85,.24,.06),_mode(120,68,.1,.12,.08)),
                         (_noise(.05,"low",12,.035),),48,720)
    if track=="snare":
        if name=="brush":
            return DrumPatch((_mode(190,160,.12,.07),),
                (_noise(.55,"white",8,.13,72),_noise(.25,"low",20,.19,120)),72,480)
        if name=="snap":
            return DrumPatch((_mode(230,185,.25,.055),_mode(410,330,.13,.04)),
                (_noise(.5,"high",6,.07,8),_noise(.12,"white",8,.025,2)),12,360)
        return DrumPatch((_mode(180,180,.25,.14),_mode(349,349,.16,.11),
                          _mode(522,508,.11,.08)),
                         (_noise(.43,"high",10,.1,24),),24,480)
    if name=="soft":
        return DrumPatch((_mode(4400,4400,.08,.035),_mode(6100,6100,.07,.03)),
                         (_noise(.65,"high",12,.055,24),),24,240)
    if name=="bright":
        return DrumPatch((_mode(5200,5200,.12,.04),_mode(7900,7900,.1,.025)),
                         (_noise(.74,"high",4,.04,4),),8,180)
    return DrumPatch(tuple(_mode(hz,hz,gain,decay) for hz,gain,decay in
                          ((3171,.16,.08),(4219,.14,.07),(5981,.12,.045),(7877,.1,.035))),
                     (_noise(.38,"high",8,.05,8),),12,240)


def preset_drum_bank(score,assignments):
    if type(assignments) is not dict or not set(assignments)<=set(DRUMS):
        raise GenerationError("Invalid percussion preset assignments")
    return DrumBank(arrangement_digest(score),tuple(DrumRoute(track,drum_preset(track,assignments[track]))
                    for track in DRUMS if track in assignments))
