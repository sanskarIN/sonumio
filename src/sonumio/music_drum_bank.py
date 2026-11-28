"""Canonical percussion patch assignments with score-pinned mixed rendering."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json

from .arrangement import Arrangement,TRACKS,arrangement_digest
from .audio import SAMPLE_RATE,encode_wav_channels
from .errors import GenerationError
from .music_automation import AutomationPlan,mix_automated_stems
from .music_mix import MixPlan,mix_stems,_render_stems
from .music_instruments import render_event
from .music_patch_synthesis import NoiseLayer
from .music_drum_patches import (
    DRUMS,MAX_MODES,MAX_NOISE_LAYERS,DrumMode,DrumNoise,DrumPatch,drum_cost,render_drum_patch,
)
from .music_patch_bank import PatchBank,describe_patch_bank,render_patch,_record
from .music_score_io import _object,_unique,_nonfinite

FORMAT="sonumio-original-percussion-patches-v1"
MAX_BYTES=64*1024
MAX_COST=100_000_000
MAX_COMBINED_COST=150_000_000


@dataclass(frozen=True,slots=True)
class DrumRoute:
    track:str
    patch:DrumPatch

    def __post_init__(self):
        if type(self.track) is not str or self.track not in DRUMS or not isinstance(self.patch,DrumPatch):
            raise GenerationError("Expected percussion patch route")


@dataclass(frozen=True,slots=True)
class DrumBank:
    score_sha256:str
    routes:tuple[DrumRoute,...]=()

    def __post_init__(self):
        if (type(self.score_sha256) is not str or len(self.score_sha256)!=64
                or any(c not in "0123456789abcdef" for c in self.score_sha256)):
            raise GenerationError("Invalid percussion score SHA256")
        if (type(self.routes) is not tuple or len(self.routes)>len(DRUMS)
                or any(not isinstance(r,DrumRoute) for r in self.routes)):
            raise GenerationError("Invalid percussion patch routes")
        names=tuple(r.track for r in self.routes)
        if names!=tuple(t for t in DRUMS if t in names):
            raise GenerationError("Percussion patch routes must be unique and canonical")

    @property
    def sha256(self):
        return hashlib.sha256(encode_drum_bank(self)).hexdigest()


def encode_drum_bank(bank):
    if not isinstance(bank,DrumBank):
        raise GenerationError("Expected DrumBank")
    data=json.dumps({"format":FORMAT,"score_sha256":bank.score_sha256,
                     "routes":[asdict(route) for route in bank.routes]},
                    sort_keys=True,separators=(",",":"),allow_nan=False).encode("ascii")
    if len(data)>MAX_BYTES:
        raise GenerationError("Percussion patch document exceeds byte budget")
    return data


def _array(value,limit):
    if type(value) is not list or len(value)>limit:
        raise GenerationError("Invalid percussion patch array")
    return value


def decode_drum_patch(raw):
    fields=_record(raw,DrumPatch,("modes",))
    modes=tuple(DrumMode(**_record(row,DrumMode,("start_hz","end_hz","sweep_seconds",
                                                "amplitude","decay_seconds")))
                for row in _array(fields["modes"],MAX_MODES))
    noise=[]
    for row in _array(fields.get("noise",[]),MAX_NOISE_LAYERS):
        item=_record(row,DrumNoise,("layer",))
        noise.append(DrumNoise(NoiseLayer(**_record(item["layer"],NoiseLayer)),item.get("attack_frames",1)))
    return DrumPatch(modes,tuple(noise),**{k:v for k,v in fields.items() if k not in ("modes","noise")})


def decode_drum_bank(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_BYTES:
        raise GenerationError("Invalid percussion patch byte count")
    try:
        row=_object(json.loads(data.decode("utf-8"),object_pairs_hook=_unique,
                               parse_constant=_nonfinite),("format","score_sha256","routes"))
        if row["format"]!=FORMAT:
            raise GenerationError("Unknown percussion patch format")
        routes=[]
        for raw in _array(row["routes"],len(DRUMS)):
            route=_object(raw,("track","patch"))
            routes.append(DrumRoute(route["track"],decode_drum_patch(route["patch"])))
        bank=DrumBank(row["score_sha256"],tuple(routes))
        if expected_sha256 is not None and bank.sha256!=expected_sha256:
            raise GenerationError("Percussion patch bank SHA256 mismatch")
        return bank
    except (ValueError,TypeError,OverflowError,UnicodeError,RecursionError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed percussion patch document") from exc


def read_drum_bank(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_drum_bank(stream.read(MAX_BYTES+1),expected_sha256=expected_sha256)


def describe_drum_bank(score,bank):
    if not isinstance(score,Arrangement) or not isinstance(bank,DrumBank):
        raise GenerationError("Expected arrangement and percussion bank")
    if arrangement_digest(score)!=bank.score_sha256:
        raise GenerationError("Percussion patch score SHA256 mismatch")
    mapping={route.track:route.patch for route in bank.routes}
    tracks={track:{"custom":track in mapping,"events":0,"cost":0} for track in DRUMS}
    for event in score.events:
        if event.track not in DRUMS:
            continue
        row=tracks[event.track]
        row["events"]+=1
        if event.track in mapping:
            frames=score.plan.frame(event.tick+event.length)-score.plan.frame(event.tick)
            row["cost"]+=drum_cost(event,frames,mapping[event.track])
    cost=sum(row["cost"] for row in tracks.values())
    if cost>MAX_COST:
        raise GenerationError("Percussion patch synthesis exceeds processing budget")
    return {"format":FORMAT,"score_sha256":bank.score_sha256,"drum_bank_sha256":bank.sha256,
            "tracks":tracks,"cost":cost,"maximum_cost":MAX_COST,
            "frames":score.plan.frame(score.plan.ticks),"sample_rate":SAMPLE_RATE,"channels":2,
            "rendered":False,"learned_model":False,"external_recordings_used":False}


def describe_combined_banks(score,drums,pitched=None):
    drum_report=describe_drum_bank(score,drums)
    pitched_report=describe_patch_bank(score,pitched) if pitched is not None else None
    cost=drum_report["cost"]+(pitched_report["cost"] if pitched_report is not None else 0)
    if cost>MAX_COMBINED_COST:
        raise GenerationError("Combined patch synthesis exceeds processing budget")
    return {"protocol":"sonumio-combined-patch-render-v1","score_sha256":drums.score_sha256,
            "drum_bank_sha256":drums.sha256,"pitched_bank_sha256":pitched.sha256 if pitched is not None else None,
            "drums":drum_report,"pitched":pitched_report,"cost":cost,"maximum_cost":MAX_COMBINED_COST,
            "frames":drum_report["frames"],"sample_rate":SAMPLE_RATE,"channels":2,
            "rendered":False,"learned_model":False,"external_recordings_used":False}


def render_combined_stems(score,drums,pitched=None):
    describe_combined_banks(score,drums,pitched)
    drum_map={row.track:row.patch for row in drums.routes}
    pitched_map={row.track:row.patch for row in pitched.routes} if pitched is not None else {}
    def render(event,frames,seed):
        if event.track in drum_map:
            return render_drum_patch(event,frames,seed,drum_map[event.track])
        if event.track in pitched_map:
            return render_patch(event,frames,seed,pitched_map[event.track])
        return render_event(event,frames,seed)
    return _render_stems(score,render)


def generate_combined_music(score,drums,pitched=None,mix=None):
    report=describe_combined_banks(score,drums,pitched)
    mix=MixPlan() if mix is None else mix
    if isinstance(mix,AutomationPlan):
        mix.validate_score(score)
    elif not isinstance(mix,MixPlan):
        raise GenerationError("Expected combined patch mixer")
    stems=render_combined_stems(score,drums,pitched)
    if isinstance(mix,AutomationPlan):
        samples,details=mix_automated_stems(score,stems,mix)
    else:
        samples,details=mix_stems(stems,mix)
    wav=encode_wav_channels(samples,SAMPLE_RATE)
    report.update(rendered=True,mix=details,wav_sha256=hashlib.sha256(wav).hexdigest())
    return wav,report
