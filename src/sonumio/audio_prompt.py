"""Bounded controlled-language prompt planning for original audio synthesis.

This is a transparent parser, not a language model. Every recognized
directive is consumed exactly once. Unsupported wording raises rather
than silently guessing a prompt's meaning. English and Hindi aliases
identify a small documented set of styles, scene presets and parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from .arrangement import ArrangementPlan,compose_arrangement
from .nature_scene import NatureScene,preset
from .errors import GenerationError

MAX_PROMPT=800
MAX_TOKENS=48
MUSIC_WORDS=("music","instrumental","संगीत","वाद्य")
NATURE_WORDS=("nature","sound","प्रकृति","ध्वनि")
STYLES={"calm":"calm","शांत":"calm","cinematic":"cinematic",
        "सिनेमैटिक":"cinematic","rhythmic":"rhythmic","लयबद्ध":"rhythmic"}
MODES={"major":"major","minor":"minor","मेजर":"major","माइनर":"minor"}
PRESETS={"forest":"forest","जंगल":"forest","storm":"storm","तूफान":"storm",
         "coast":"coast","समुद्र":"coast","creek":"creek","झरना":"creek",
         "campfire":"campfire","अलाव":"campfire"}
KEYS={"c":0,"c#":1,"db":1,"d":2,"d#":3,"eb":3,"e":4,"f":5,
      "f#":6,"gb":6,"g":7,"g#":8,"ab":8,"a":9,"a#":10,"bb":10,"b":11}
DIRECTIVES={"bpm":"bpm","tempo":"bpm","गति":"bpm","bars":"bars","बार":"bars",
            "key":"key","कुंजी":"key","swing":"swing","स्विंग":"swing",
            "seed":"seed","सीड":"seed","seconds":"seconds","सेकंड":"seconds"}
INTEGER=re.compile(r"[0-9]{1,10}\Z")
DECIMAL=re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3})?\Z")


def tokenize(prompt):
    if type(prompt) is not str or not 1<=len(prompt)<=MAX_PROMPT:
        raise GenerationError("Invalid controlled audio prompt length")
    # Reject controls and invisible formatting instead of erasing them.
    if any(unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"
           for ch in prompt):
        raise GenerationError("Audio prompt contains unsupported control characters")
    normalized=unicodedata.normalize("NFC",prompt).casefold()
    normalized=normalized.replace("♯","#").replace("♭","b")
    tokens=tuple(normalized.split())
    if not 1<=len(tokens)<=MAX_TOKENS:
        raise GenerationError("Invalid bounded audio prompt tokens")
    return tokens


def _int(token,name,lower,upper):
    if INTEGER.fullmatch(token) is None:
        raise GenerationError("Expected ASCII integer for "+name)
    value=int(token)
    if not lower<=value<=upper:
        raise GenerationError("Audio prompt "+name+" out of bounds")
    return value


def _decimal(token,name,lower,upper):
    if DECIMAL.fullmatch(token) is None:
        raise GenerationError("Expected bounded decimal for "+name)
    try:
        value=Decimal(token)
    except InvalidOperation as exc:
        raise GenerationError("Invalid decimal for "+name) from exc
    if not Decimal(str(lower))<=value<=Decimal(str(upper)):
        raise GenerationError("Audio prompt "+name+" out of bounds")
    return float(value)


def _key(token):
    if token in KEYS:
        return 60+KEYS[token]
    # MIDI tonic can be explicit, retaining existing arrangement bounds.
    return _int(token,"key",48,71)


def _assign(values,name,value):
    if name in values:
        raise GenerationError("Duplicate audio prompt directive: "+name)
    values[name]=value


@dataclass(frozen=True,slots=True)
class PromptPlan:
    kind:str
    music:ArrangementPlan|None=None
    nature:NatureScene|None=None
    preset_name:str|None=None

    def __post_init__(self):
        if self.kind=="music":
            if not isinstance(self.music,ArrangementPlan) or self.nature is not None or self.preset_name is not None:
                raise GenerationError("Invalid music prompt plan")
        elif self.kind=="nature":
            if (not isinstance(self.nature,NatureScene) or self.music is not None
                    or self.preset_name not in PRESETS.values()):
                raise GenerationError("Invalid nature prompt plan")
        else:
            raise GenerationError("Unknown audio prompt kind")


def parse_prompt(prompt,*,default_seed=0):
    if type(default_seed) is not int or not 0<=default_seed<2**32:
        raise GenerationError("Invalid default audio prompt seed")
    tokens=tokenize(prompt)
    head=tokens[0]
    if head in MUSIC_WORDS:
        kind="music"
    elif head in NATURE_WORDS:
        kind="nature"
    else:
        raise GenerationError("Prompt must start with music or nature")
    values={}
    i=1
    while i<len(tokens):
        token=tokens[i]
        if token in DIRECTIVES:
            name=DIRECTIVES[token]
            if i+1>=len(tokens):
                raise GenerationError("Missing value for "+name)
            raw=tokens[i+1]
            if name=="seed":
                value=_int(raw,name,0,2**32-1)
            elif kind=="music" and name=="bpm":
                value=_int(raw,name,60,160)
            elif kind=="music" and name=="bars":
                value=_int(raw,name,4,32)
            elif kind=="music" and name=="key":
                value=_key(raw)
                name="root"
            elif kind=="music" and name=="swing":
                value=_decimal(raw,name,0,.6)
            elif kind=="nature" and name=="seconds":
                value=_decimal(raw,name,1,30)
            else:
                raise GenerationError("Directive not supported for "+kind+": "+name)
            _assign(values,name,value)
            i+=2
            continue
        if kind=="music" and token in STYLES:
            _assign(values,"style",STYLES[token])
        elif kind=="music" and token in MODES:
            _assign(values,"mode",MODES[token])
        elif kind=="nature" and token in PRESETS:
            _assign(values,"preset_name",PRESETS[token])
        else:
            raise GenerationError("Unsupported audio prompt token: "+token[:40])
        i+=1
    values.setdefault("seed",default_seed)
    if kind=="music":
        return PromptPlan("music",music=ArrangementPlan(**values))
    if "preset_name" not in values:
        raise GenerationError("Nature prompt requires a supported preset")
    name=values.pop("preset_name")
    return PromptPlan("nature",nature=preset(name,**values),preset_name=name)


def canonical_prompt(plan):
    if not isinstance(plan,PromptPlan):
        raise GenerationError("Expected parsed audio prompt")
    if plan.kind=="music":
        p=plan.music
        return ("music "+p.style+" "+p.mode+" key "+str(p.root)+" bpm "+str(p.bpm)
                +" bars "+str(p.bars)+" swing "+format(p.swing,".3f")
                +" seed "+str(p.seed))
    # A nature plan may be constructed directly by callers. Only emit a
    # preset command if it exactly reconstructs the entire scene.
    p=plan.nature
    seconds=p.frames/24000
    seed=(p.layers[0].voice.seed-2654435761)%(2**32)
    rounded=float(format(seconds,".3f"))
    if preset(plan.preset_name,rounded,seed)!=p:
        raise GenerationError("Nature scene is not a canonical preset prompt")
    return ("nature "+plan.preset_name+" seconds "+format(rounded,".3f")+" seed "+str(seed))


def build_source(plan):
    if not isinstance(plan,PromptPlan):
        raise GenerationError("Expected parsed audio prompt plan")
    return compose_arrangement(plan.music) if plan.kind=="music" else plan.nature
