"""Original procedural sound-library plans and deterministic WAV assets.

A pack is a source-only DSP artifact: no downloads, learned weights, external
recordings, voice imitation, or unverified rights declarations. The archive
layer verifies the actual encoded integer samples separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re

import numpy as np

from ..audio import SAMPLE_RATE
from ..engines import procedural
from ..nature_synthesis import NatureVoice, KINDS, render_voice
from ..validation import validate_request
from .contracts import finite_float, make_buffer
from .dither import QuantizationSettings, unpack_pcm
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from .hashing import canonical_json_bytes, parse_canonical_json, strict_sha256
from .wave_reader import _decode_canonical_header
from .waveio import encode_wav_bytes

MAX_RECIPES = 48
MAX_ASSETS = 128
MAX_VARIANTS = 24
MAX_TOTAL_FRAMES = 6_000_000
MAX_MEMBER_BYTES = 3 * 1024 * 1024
MAX_PAYLOAD_BYTES = 24 * 1024 * 1024
MAX_PLAN_BYTES = 128 * 1024
MANIFEST_PROTOCOL = "sonumio-original-sound-pack-v1"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,47}")
_TAG = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1,10)),
             *(f"lpt{i}" for i in range(1,10))}


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be an integer in [{low}, {high}]")
    return value


def identifier(value):
    if type(value) is not str or _ID.fullmatch(value) is None or value in _RESERVED:
        raise AudioContractError("Invalid portable sound asset identifier")
    return value


@dataclass(frozen=True, slots=True)
class AssetRecipe:
    asset_id: str
    frames: int
    variants: int = 1
    source: str = "nature"
    nature_kind: str | None = "rain"
    prompt: str | None = None
    intensity: float = .5
    texture: float = .5
    motion: float = .5
    gain_db: float = 0.
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        identifier(self.asset_id)
        _integer(self.frames,"asset frames",SAMPLE_RATE//10,15*SAMPLE_RATE)
        _integer(self.variants,"asset variants",1,MAX_VARIANTS)
        if self.source not in ("nature","procedural"):
            raise AudioContractError("Unknown original sound asset source")
        for name in ("intensity","texture","motion"):
            finite_float(getattr(self,name),name,minimum=0,maximum=1)
        finite_float(self.gain_db,"asset gain",minimum=-60,maximum=12)
        if (type(self.tags) is not tuple or len(self.tags)>16
                or any(type(t) is not str or _TAG.fullmatch(t) is None for t in self.tags)
                or tuple(sorted(set(self.tags)))!=self.tags):
            raise AudioContractError("Tags must be sorted unique portable labels")
        if self.source=="nature":
            if self.nature_kind not in KINDS or self.prompt is not None:
                raise AudioContractError("Nature source requires a known kind and no prompt")
        else:
            if self.nature_kind is not None or (self.intensity,self.texture,self.motion)!=(.5,.5,.5):
                raise AudioContractError("Nature controls cannot override a procedural prompt")
            if self.frames < SAMPLE_RATE//2:
                raise AudioContractError("Prompt source requires at least half a second")
            prompt=validate_request(self.prompt,self.frames/SAMPLE_RATE)
            if prompt!=self.prompt:
                raise AudioContractError("Prompt must already be trimmed")
            procedural.recognized_sound_classes(prompt)


@dataclass(frozen=True, slots=True)
class PackQuality:
    minimum_rms: float = .0001
    allow_rails: bool = False
    allow_duplicate_audio: bool = True

    def __post_init__(self):
        finite_float(self.minimum_rms,"minimum RMS",minimum=0,maximum=1)
        if type(self.allow_rails) is not bool or type(self.allow_duplicate_audio) is not bool:
            raise AudioContractError("Pack quality flags must be boolean")


@dataclass(frozen=True, slots=True)
class SoundPackPlan:
    pack_id: str
    recipes: tuple[AssetRecipe, ...]
    seed: int = 0
    output_bits: int = 16
    dither: bool = False
    peak_ceiling: float = .9
    quality: PackQuality = PackQuality()

    def __post_init__(self):
        identifier(self.pack_id)
        _integer(self.seed,"pack seed",0,2**32-1)
        if type(self.dither) is not bool:
            raise AudioContractError("Pack dither must be boolean")
        QuantizationSettings(bits=self.output_bits,dither=self.dither,
                             seed=0 if self.dither else None)
        finite_float(self.peak_ceiling,"floating peak ceiling",minimum=.001,maximum=1)
        if not isinstance(self.quality,PackQuality):
            raise AudioContractError("Expected PackQuality")
        if (type(self.recipes) is not tuple or not 1<=len(self.recipes)<=MAX_RECIPES
                or any(not isinstance(r,AssetRecipe) for r in self.recipes)):
            raise AudioContractError("Expected bounded asset recipe tuple")
        names=[r.asset_id for r in self.recipes]
        if names!=sorted(set(names)):
            raise AudioContractError("Recipes must have sorted unique asset identifiers")
        if self.asset_count>MAX_ASSETS or self.total_frames>MAX_TOTAL_FRAMES:
            raise AudioLimitError("Sound pack exceeds asset-count or frame budget")
        if self.payload_bytes>MAX_PAYLOAD_BYTES:
            raise AudioLimitError("Sound pack exceeds PCM byte budget")

    @property
    def asset_count(self):
        return sum(r.variants for r in self.recipes)

    @property
    def total_frames(self):
        return sum(r.variants*r.frames for r in self.recipes)

    @property
    def payload_bytes(self):
        return sum(r.variants*wav_size(r.frames,self.output_bits) for r in self.recipes)

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


def wav_size(frames,bits):
    n=frames*(bits//8)
    return 44+n+(n%2)


def _object(value,cls):
    if type(value) is not dict or not set(value)<=set(cls.__dataclass_fields__):
        raise AudioContractError("Unexpected fields in "+cls.__name__)
    return dict(value)


def parse_pack_plan(raw):
    obj=_object(parse_canonical_json(raw,max_bytes=MAX_PLAN_BYTES),SoundPackPlan)
    items=obj.get("recipes")
    if type(items) is not list or not 1<=len(items)<=MAX_RECIPES:
        raise AudioContractError("Expected bounded recipe array")
    recipes=[]
    for value in items:
        recipe=_object(value,AssetRecipe)
        if "tags" in recipe:
            if type(recipe["tags"]) is not list:
                raise AudioContractError("Asset tags must be an array")
            recipe["tags"]=tuple(recipe["tags"])
        recipes.append(AssetRecipe(**recipe))
    obj["recipes"]=tuple(recipes)
    if "quality" in obj:
        obj["quality"]=PackQuality(**_object(obj["quality"],PackQuality))
    return SoundPackPlan(**obj)


@dataclass(frozen=True, slots=True)
class AssetJob:
    recipe: AssetRecipe
    variant: int
    seed: int
    dither_seed: int | None

    @property
    def member(self):
        return f"audio/{self.recipe.asset_id}-{self.variant:03d}.wav"


def _seed(master,asset,variant,domain):
    payload=b"sonumio-asset-seed-v1\x00"+master.to_bytes(4,"little")
    payload+=asset.encode("ascii")+b"\x00"+variant.to_bytes(2,"little")+domain
    return int.from_bytes(hashlib.sha256(payload).digest()[:4],"little")


def plan_jobs(plan):
    if not isinstance(plan,SoundPackPlan):
        raise AudioContractError("Expected SoundPackPlan")
    return tuple(
        AssetJob(recipe,index,_seed(plan.seed,recipe.asset_id,index,b"audio"),
                 _seed(plan.seed,recipe.asset_id,index,b"dither") if plan.dither else None)
        for recipe in plan.recipes for index in range(1,recipe.variants+1)
    )


@dataclass(frozen=True, slots=True)
class IntegerSignal:
    """Exact integer PCM diagnostics (not a listening score or LUFS)."""
    samples: int
    minimum_code: int
    maximum_code: int
    peak_code: int
    nonzero_samples: int
    positive_rails: int
    negative_rails: int
    sum_codes: int
    sum_squared_codes: int

    def rms(self,bits):
        return math.sqrt(self.sum_squared_codes/self.samples)/(1<<(bits-1))

    def rail_count(self):
        return self.positive_rails+self.negative_rails


def integer_signal(codes,bits):
    if (not isinstance(codes,np.ndarray) or codes.dtype!=np.int64
            or codes.ndim!=2 or codes.shape[0]!=1 or not codes.size
            or codes.size>15*SAMPLE_RATE):
        raise AudioContractError("Expected bounded mono integer PCM codes")
    QuantizationSettings(bits=bits)
    low=-(1<<(bits-1))
    high=-low-1
    minimum,maximum=int(codes.min()),int(codes.max())
    if minimum<low or maximum>high:
        raise AudioContractError("PCM code exceeds declared bit depth")
    # Square in split 16-bit limbs. Every vectorized sum fits int64 at
    # the declared frame cap, including full-scale signed 32-bit PCM.
    absolute=np.abs(codes)
    upper=absolute>>16
    lower=absolute&65535
    aa=int(np.sum(upper*upper,dtype=np.int64))
    bb=int(np.sum(lower*lower,dtype=np.int64))
    ab=int(np.sum(upper*lower,dtype=np.int64))
    square=(aa<<32)+(ab<<17)+bb
    return IntegerSignal(int(codes.size),minimum,maximum,max(abs(minimum),abs(maximum)),
        int(np.count_nonzero(codes)),int(np.count_nonzero(codes==high)),
        int(np.count_nonzero(codes==low)),int(codes.sum(dtype=np.int64)),square)


@dataclass(frozen=True, slots=True)
class AssetRecord:
    member: str
    asset_id: str
    variant: int
    source: str
    seed: int
    dither_seed: int | None
    frames: int
    sample_rate: int
    output_bits: int
    bytes: int
    wav_sha256: str
    pcm_sha256: str
    signal: IntegerSignal

    def as_dict(self):
        return asdict(self)


def inspect_asset(job,encoded,bits):
    """Verify one canonical WAV and derive its exact integer record."""
    if not isinstance(job,AssetJob):
        raise AudioContractError("Expected planned asset job")
    if not isinstance(encoded,bytes) or not 44<len(encoded)<=MAX_MEMBER_BYTES:
        raise AudioFormatError("Invalid asset WAV payload")
    expected=wav_size(job.recipe.frames,bits)
    if len(encoded)!=expected:
        raise AudioIntegrityError("Asset byte count differs from planned geometry")
    info=_decode_canonical_header(encoded[:44],len(encoded))
    if (info.frames!=job.recipe.frames or info.channels!=1
            or info.sample_rate!=SAMPLE_RATE or info.bits_per_sample!=bits):
        raise AudioIntegrityError("Asset WAV differs from planned sample geometry")
    if info.data_bytes%2 and encoded[-1]!=0:
        raise AudioFormatError("Nonzero canonical WAV pad")
    payload=encoded[44:44+info.data_bytes]
    codes=unpack_pcm(payload,bits=bits,channels=1)
    signal=integer_signal(codes,bits)
    return AssetRecord(job.member,job.recipe.asset_id,job.variant,job.recipe.source,
        job.seed,job.dither_seed,info.frames,info.sample_rate,bits,len(encoded),
        hashlib.sha256(encoded).hexdigest(),hashlib.sha256(payload).hexdigest(),signal)


def render_asset(job,plan):
    if not isinstance(job,AssetJob) or not isinstance(plan,SoundPackPlan):
        raise AudioContractError("Expected planned asset and sound pack")
    recipe=job.recipe
    if recipe.source=="nature":
        voice=NatureVoice(recipe.nature_kind,recipe.intensity,recipe.texture,
                          recipe.motion,job.seed)
        values=render_voice(voice,recipe.frames)
    else:
        values=procedural.generate(recipe.prompt,recipe.frames/SAMPLE_RATE,job.seed).samples
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(recipe.frames,) or not np.isfinite(values).all():
        raise AudioIntegrityError("Asset renderer returned incorrect or nonfinite geometry")
    values*=10**(recipe.gain_db/20)
    peak=float(np.max(np.abs(values)))
    if peak>plan.peak_ceiling:
        values*=plan.peak_ceiling/peak
    audio=make_buffer(values.astype(np.float32),SAMPLE_RATE,source_note="original procedural pack")
    settings=QuantizationSettings(bits=plan.output_bits,dither=plan.dither,seed=job.dither_seed)
    return encode_wav_bytes(audio,settings=settings)


def duplicate_groups(records):
    groups={}
    for record in records:
        groups.setdefault(record.wav_sha256,[]).append(record.member)
    return tuple(tuple(group) for group in groups.values() if len(group)>1)


def enforce_quality(plan,records):
    for record in records:
        if record.signal.rms(record.output_bits)<plan.quality.minimum_rms:
            raise AudioIntegrityError("Asset RMS below declared minimum: "+record.member)
        if not plan.quality.allow_rails and record.signal.rail_count():
            raise AudioIntegrityError("Asset integer rail occupancy rejected: "+record.member)
    if not plan.quality.allow_duplicate_audio and duplicate_groups(records):
        raise AudioIntegrityError("Identical WAV assets rejected by pack policy")


def build_manifest(plan,records):
    jobs=plan_jobs(plan)
    if type(records) is not tuple or len(records)!=len(jobs):
        raise AudioContractError("Asset receipt count differs from plan")
    for record,job in zip(records,jobs,strict=True):
        if not isinstance(record,AssetRecord) or (
            record.member,record.asset_id,record.variant,record.source,record.seed,
            record.dither_seed,record.frames,record.sample_rate,record.output_bits
        )!=(job.member,job.recipe.asset_id,job.variant,job.recipe.source,job.seed,
            job.dither_seed,job.recipe.frames,SAMPLE_RATE,plan.output_bits):
            raise AudioIntegrityError("Asset receipt differs from planned job")
        strict_sha256(record.wav_sha256)
        strict_sha256(record.pcm_sha256)
    enforce_quality(plan,records)
    return {
        "protocol":MANIFEST_PROTOCOL,
        "plan":asdict(plan),
        "plan_sha256":plan.sha256,
        "asset_count":len(records),
        "total_frames":sum(r.frames for r in records),
        "payload_bytes":sum(r.bytes for r in records),
        "assets":[r.as_dict() for r in records],
        "duplicate_groups":duplicate_groups(records),
        "model_inference":False,
        "recording_sources_used":False,
        "perceptual_quality_established":False,
    }


def generate_pack_members(plan):
    """Return bounded original WAV members and their deterministic manifest."""
    jobs=plan_jobs(plan)
    members={}
    records=[]
    for job in jobs:
        encoded=render_asset(job,plan)
        record=inspect_asset(job,encoded,plan.output_bits)
        members[job.member]=encoded
        records.append(record)
    manifest=build_manifest(plan,tuple(records))
    return members,manifest
