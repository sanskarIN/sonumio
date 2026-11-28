"""Canonical score-pinned patch assignment and bounded music rendering."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json

from .arrangement import Arrangement,PITCHED,TRACKS,arrangement_digest
from .audio import SAMPLE_RATE,encode_wav_channels
from .errors import GenerationError
from .music_automation import AutomationPlan,mix_automated_stems
from .music_mix import MixPlan,mix_stems,_render_stems
from .music_instruments import render_event
from .music_patch_synthesis import (
    InstrumentPatch,OscillatorBank,Partial,Envelope,NoiseLayer,MAX_BANKS,MAX_PARTIALS,
    patch_cost,render_patch,active_partials,
)
from .music_score_io import _unique,_nonfinite
from .music_automation import _object

FORMAT="sonumio-original-instrument-patches-v1"
MAX_BYTES=128*1024
MAX_COST=120_000_000


@dataclass(frozen=True,slots=True)
class PatchRoute:
    track:str
    patch:InstrumentPatch

    def __post_init__(self):
        if type(self.track) is not str or self.track not in PITCHED:
            raise GenerationError("Patch route requires a pitched instrument track")
        if not isinstance(self.patch,InstrumentPatch):
            raise GenerationError("Expected validated instrument patch")


@dataclass(frozen=True,slots=True)
class PatchBank:
    score_sha256:str
    routes:tuple[PatchRoute,...]=()

    def __post_init__(self):
        if (type(self.score_sha256) is not str or len(self.score_sha256)!=64
                or any(c not in "0123456789abcdef" for c in self.score_sha256)):
            raise GenerationError("Invalid patch score SHA256")
        if (type(self.routes) is not tuple or len(self.routes)>len(PITCHED)
                or any(not isinstance(r,PatchRoute) for r in self.routes)):
            raise GenerationError("Invalid patch routes")
        tracks=tuple(r.track for r in self.routes)
        if tracks!=tuple(t for t in PITCHED if t in tracks):
            raise GenerationError("Patch routes must be unique and canonical")

    @property
    def sha256(self):
        return hashlib.sha256(encode_patch_bank(self)).hexdigest()


def encode_patch_bank(bank):
    if not isinstance(bank,PatchBank):
        raise GenerationError("Expected PatchBank")
    data=json.dumps({"format":FORMAT,"score_sha256":bank.score_sha256,
                     "routes":[asdict(route) for route in bank.routes]},
                    sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
    if len(data)>MAX_BYTES:
        raise GenerationError("Patch document exceeds byte budget")
    return data


def _record(value,cls,required=()):
    if type(value) is not dict:
        raise GenerationError("Expected patch parameter object")
    return _object(value,required,tuple(k for k in cls.__dataclass_fields__ if k not in required))


def _array(value,maximum):
    if type(value) is not list or not 1<=len(value)<=maximum:
        raise GenerationError("Invalid bounded patch array")
    return value


def decode_patch(value):
    row=_record(value,InstrumentPatch,("banks",))
    banks=[]
    for raw in _array(row["banks"],MAX_BANKS):
        fields=_record(raw,OscillatorBank,("partials",))
        partials=tuple(Partial(**_record(p,Partial,("harmonic","amplitude")))
                       for p in _array(fields["partials"],MAX_PARTIALS))
        banks.append(OscillatorBank(partials,**{k:v for k,v in fields.items() if k!="partials"}))
    envelope=Envelope(**_record(row.get("envelope",{}),Envelope))
    noise=NoiseLayer(**_record(row.get("noise",{}),NoiseLayer))
    return InstrumentPatch(tuple(banks),envelope,noise,row.get("velocity_exponent",1.))


def decode_patch_bank(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_BYTES:
        raise GenerationError("Invalid patch document byte count")
    try:
        row=_object(json.loads(data.decode("utf-8"),object_pairs_hook=_unique,
                               parse_constant=_nonfinite),("format","score_sha256","routes"))
        if row["format"]!=FORMAT or type(row["routes"]) is not list or len(row["routes"])>len(PITCHED):
            raise GenerationError("Invalid instrument patch document")
        routes=[]
        for raw in row["routes"]:
            route=_object(raw,("track","patch"))
            routes.append(PatchRoute(route["track"],decode_patch(route["patch"])))
        bank=PatchBank(row["score_sha256"],tuple(routes))
        if expected_sha256 is not None and bank.sha256!=expected_sha256:
            raise GenerationError("Instrument patch bank SHA256 mismatch")
        return bank
    except (ValueError,TypeError,OverflowError,UnicodeError,RecursionError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed instrument patch document") from exc


def read_patch_bank(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_patch_bank(stream.read(MAX_BYTES+1),expected_sha256=expected_sha256)


def describe_patch_bank(score,bank):
    if not isinstance(score,Arrangement) or not isinstance(bank,PatchBank):
        raise GenerationError("Expected arrangement and patch bank")
    if arrangement_digest(score)!=bank.score_sha256:
        raise GenerationError("Patch bank score SHA256 mismatch")
    mapping={route.track:route.patch for route in bank.routes}
    tracks={t:{"events":0,"custom":t in mapping,"cost":0,
               "suppressed_partials":0,"note_frames":0} for t in TRACKS}
    for event in score.events:
        frames=score.plan.frame(event.tick+event.length)-score.plan.frame(event.tick)
        report=tracks[event.track]
        report["events"]+=1
        report["note_frames"]+=frames
        if event.track in mapping:
            patch=mapping[event.track]
            report["cost"]+=patch_cost(event,frames,patch)
            for layer in patch.banks:
                if layer.gain:
                    report["suppressed_partials"]+=sum(p.amplitude>0 for p in layer.partials)-len(active_partials(event.pitch,layer))
    cost=sum(report["cost"] for report in tracks.values())
    if cost>MAX_COST:
        raise GenerationError("Instrument patch synthesis exceeds processing budget")
    return {"format":FORMAT,"score_sha256":bank.score_sha256,"patch_bank_sha256":bank.sha256,
            "frames":score.plan.frame(score.plan.ticks),"sample_rate":SAMPLE_RATE,"channels":2,
            "tracks":tracks,"custom_note_events":sum(v["events"] for v in tracks.values() if v["custom"]),
            "cost":cost,"maximum_cost":MAX_COST,"rendered":False,"learned_model":False,
            "external_recordings_used":False,"strictly_bandlimited_claimed":False}


def render_patched_stems(score,bank):
    describe_patch_bank(score,bank)
    mapping={route.track:route.patch for route in bank.routes}
    def render(event,frames,seed):
        if event.track in mapping:
            return render_patch(event,frames,seed,mapping[event.track])
        return render_event(event,frames,seed)
    return _render_stems(score,render)


def generate_patched_music(score,bank,mix=None):
    report=describe_patch_bank(score,bank)
    mix=MixPlan() if mix is None else mix
    if isinstance(mix,AutomationPlan):
        mix.validate_score(score)
    elif not isinstance(mix,MixPlan):
        raise GenerationError("Expected patch mix or automation")
    stems=render_patched_stems(score,bank)
    if isinstance(mix,AutomationPlan):
        samples,details=mix_automated_stems(score,stems,mix)
    else:
        samples,details=mix_stems(stems,mix)
    wav=encode_wav_channels(samples,SAMPLE_RATE)
    report.update(rendered=True,wav_sha256=hashlib.sha256(wav).hexdigest(),mix=details)
    return wav,report
