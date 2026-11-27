"""Strict declarative moving sound scenes with verified local PCM sources.

A JSON scene never selects filesystem paths. Callers bind its declared
source identifiers to explicit local inputs. All source snapshots finish
verification before rendering. No intermediate file is created.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib
import math
import os
import re
import stat
import numpy as np

from .contracts import AudioSpec,AudioLayout,check_rate,finite_float
from .errors import AudioContractError,AudioFormatError,AudioIntegrityError,AudioLimitError
from .hashing import canonical_json_bytes,parse_canonical_json,strict_sha256,digest_audio
from .wave_snapshot import read_verified_pcm
from .waveio import encode_wav_bytes
from .dither import QuantizationSettings
from .spatial_motion import (
    Position,MotionKey,Trajectory,Receiver,Room,MovingSource,SpatialScene,
    SpatialReport,render_spatial_scene,MAX_SOURCES,
)

MAX_PLAN_BYTES=256*1024
MAX_INPUT_PCM_BYTES=24*1024*1024
MAX_OUTPUT_PCM_BYTES=24*1024*1024
_SOURCE_ID=re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


def source_id(value):
    if type(value) is not str or _SOURCE_ID.fullmatch(value) is None:
        raise AudioContractError("Invalid spatial source identifier")
    return value


@dataclass(frozen=True,slots=True)
class SpatialPlacement:
    source_id: str
    trajectory: Trajectory
    start_frame: int=0
    gain_db: float=0.

    def __post_init__(self):
        source_id(self.source_id)
        if not isinstance(self.trajectory,Trajectory):
            raise AudioContractError("Expected placement Trajectory")
        if type(self.start_frame) is not int or not 0<=self.start_frame<2**31:
            raise AudioContractError("Invalid placement frame offset")
        finite_float(self.gain_db,"placement gain",minimum=-60,maximum=12)


@dataclass(frozen=True,slots=True)
class SpatialPlan:
    sample_rate: int
    placements: tuple[SpatialPlacement,...]
    frames: int | None=None
    receiver: Receiver=Receiver()
    room: Room | None=None
    peak_ceiling: float | None=.95
    tail_frames: int=0

    def __post_init__(self):
        check_rate(self.sample_rate)
        if (type(self.placements) is not tuple or not 1<=len(self.placements)<=MAX_SOURCES
                or any(not isinstance(p,SpatialPlacement) for p in self.placements)):
            raise AudioContractError("Expected bounded spatial placements")
        if self.frames is not None:
            AudioSpec(self.sample_rate,AudioLayout.for_channels(2),self.frames)
        if type(self.tail_frames) is not int or not 0<=self.tail_frames<=self.sample_rate*5:
            raise AudioContractError("Invalid spatial tail frame count")
        if self.frames is not None and self.tail_frames:
            raise AudioContractError("Tail frames apply only to automatic scene duration")
        if not isinstance(self.receiver,Receiver):
            raise AudioContractError("Expected Receiver")
        if self.room is not None and not isinstance(self.room,Room):
            raise AudioContractError("Expected Room")
        if self.peak_ceiling is not None:
            finite_float(self.peak_ceiling,"peak ceiling",minimum=.001,maximum=1)

    @property
    def source_ids(self):
        return tuple(sorted({p.source_id for p in self.placements}))

    @property
    def sha256(self):
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()

    def bind(self,audios):
        if type(audios) is not dict or set(audios)!=set(self.source_ids):
            raise AudioContractError("Source bindings must exactly match scene identifiers")
        sources=tuple(MovingSource(audios[p.source_id],p.trajectory,p.start_frame,p.gain_db)
                      for p in self.placements)
        if any(s.audio.sample_rate!=self.sample_rate for s in sources):
            raise AudioContractError("Source sample clock does not match spatial plan")
        frames=self.frames
        if frames is None:
            frames=automatic_scene_frames(sources,self.receiver,self.room,self.sample_rate)+self.tail_frames
        return SpatialScene(self.sample_rate,frames,sources,self.receiver,self.room,self.peak_ceiling)


def automatic_scene_frames(sources,receiver,room,rate):
    """Bound the complete last arrival for all paths and both channels.

    Since all trajectories are bounded subsonic piecewise-linear motion,
    arrival time increases strictly on every direct/image-source path.
    Only each source's last emission position is needed for the final
    arrival bound. The receiver end includes its ceiling integer frame.
    """
    check_rate(rate)
    if not isinstance(receiver,Receiver) or (room is not None and not isinstance(room,Room)):
        raise AudioContractError("Invalid spatial receiver or room")
    if type(sources) is not tuple or not sources or any(not isinstance(s,MovingSource) for s in sources):
        raise AudioContractError("Expected moving sources")
    ears=receiver.geometry()[:2]
    last=0.
    for source in sources:
        if source.audio.sample_rate!=rate:
            raise AudioContractError("Spatial sample clock mismatch")
        position=source.trajectory.keys[-1].position
        x,y=position.x,position.y
        positions=[(x,y)]
        if room is not None:
            for wall in room.active_walls:
                positions.append({
                    "left":(-x,y),"right":(2*room.width-x,y),
                    "bottom":(x,-y),"top":(x,2*room.depth-y),
                }[wall])
        for px,py in positions:
            for ear in ears:
                delay=math.hypot(px-ear[0],py-ear[1])*rate/receiver.sound_speed
                last=max(last,source.start_frame+source.audio.frames-1+delay)
    return math.ceil(last)+1


def _object(value,cls):
    if type(value) is not dict or not set(value)<=set(cls.__dataclass_fields__):
        raise AudioContractError("Unexpected fields in "+cls.__name__)
    return dict(value)


def parse_spatial_plan(raw):
    data=_object(parse_canonical_json(raw,max_bytes=MAX_PLAN_BYTES),SpatialPlan)
    if "receiver" in data:
        r=_object(data["receiver"],Receiver)
        if "position" in r:
            r["position"]=Position(**_object(r["position"],Position))
        data["receiver"]=Receiver(**r)
    if data.get("room") is not None:
        data["room"]=Room(**_object(data["room"],Room))
    items=data.get("placements")
    if type(items) is not list or not 1<=len(items)<=MAX_SOURCES:
        raise AudioContractError("Expected bounded spatial placement array")
    placements=[]
    for item in items:
        p=_object(item,SpatialPlacement)
        trajectory=_object(p.get("trajectory"),Trajectory)
        keys=trajectory.get("keys")
        if type(keys) is not list or not 1<=len(keys)<=128:
            raise AudioContractError("Expected bounded motion keys")
        parsed=[]
        for key in keys:
            k=_object(key,MotionKey)
            k["position"]=Position(**_object(k.get("position"),Position))
            parsed.append(MotionKey(**k))
        p["trajectory"]=Trajectory(tuple(parsed))
        placements.append(SpatialPlacement(**p))
    data["placements"]=tuple(placements)
    return SpatialPlan(**data)


def load_spatial_plan(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise AudioFormatError("Expected ordinary spatial JSON file")
    flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)|getattr(os,"O_BINARY",0)
    fd=os.open(path,flags)
    with os.fdopen(fd,"rb") as stream:
        meta=os.fstat(stream.fileno())
        if not stat.S_ISREG(meta.st_mode):
            raise AudioFormatError("Expected ordinary spatial JSON descriptor")
        if meta.st_size>MAX_PLAN_BYTES:
            raise AudioLimitError("Spatial JSON exceeds byte budget")
        raw=stream.read(MAX_PLAN_BYTES+1)
    return parse_spatial_plan(raw)


@dataclass(frozen=True,slots=True)
class SourceBinding:
    path: Path
    expected_sha256: str | None=None

    def __post_init__(self):
        if not isinstance(self.path,Path):
            raise AudioContractError("Source binding requires local Path")
        if self.expected_sha256 is not None:
            strict_sha256(self.expected_sha256)


@dataclass(frozen=True,slots=True)
class SpatialExport:
    plan_sha256: str
    source_receipts: dict
    output_sha256: str
    output_pcm_sha256: str
    output_bytes: int
    output_bits: int
    dither_seed: int | None
    render: SpatialReport
    protocol: str="sonumio-verified-spatial-wav-v1"

    def as_dict(self):
        return {
            "protocol":self.protocol,
            "plan_sha256":self.plan_sha256,
            "sources":self.source_receipts,
            "output_sha256":self.output_sha256,
            "pre_encoding_pcm_sha256":self.output_pcm_sha256,
            "output_bytes":self.output_bytes,
            "output_bits":self.output_bits,
            "dither_seed":self.dither_seed,
            "render":self.render.as_dict(),
            "model_inference":False,
        }


def _distinct(source,target):
    if source.resolve()==target.resolve():
        raise AudioFormatError("Spatial output must differ from source")
    if target.exists():
        try:
            if os.path.samefile(source,target):
                raise AudioFormatError("Spatial output shares source file identity")
        except FileNotFoundError:
            pass


def export_spatial_plan(plan,bindings,destination,*,output_bits=16,dither_seed=None,
                        packet_frames=8192,expected_output_sha256=None,
                        overwrite=False,create_parent=False):
    """Verify all named mono WAV snapshots, then render and publish once."""
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(plan,SpatialPlan):
        raise AudioContractError("Expected SpatialPlan")
    if type(bindings) is not dict or set(bindings)!=set(plan.source_ids):
        raise AudioContractError("Source bindings must exactly match scene identifiers")
    if any(not isinstance(b,SourceBinding) for b in bindings.values()):
        raise AudioContractError("Expected SourceBinding values")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid spatial publication flags")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    quantization=QuantizationSettings(bits=output_bits,dither=dither_seed is not None,seed=dither_seed)
    target=Path(destination)
    if target.suffix.lower()!=".wav":
        raise AudioFormatError("Spatial output must end in .wav")
    for binding in bindings.values():
        _distinct(binding.path,target)
    audios={}
    receipts={}
    remaining=MAX_INPUT_PCM_BYTES
    for name in plan.source_ids:
        if remaining<4:
            raise AudioLimitError("Spatial source PCM budget exhausted")
        binding=bindings[name]
        snapshot=read_verified_pcm(binding.path,expected_sha256=binding.expected_sha256,
            packet_frames=packet_frames,maximum_pcm_bytes=remaining)
        if snapshot.audio.channels!=1 or snapshot.audio.sample_rate!=plan.sample_rate:
            raise AudioContractError("Spatial sources must be mono on the declared clock")
        remaining-=snapshot.audio.spec.byte_count
        audios[name]=snapshot.audio
        receipts[name]={**snapshot.as_dict(),"source_pcm_sha256":digest_audio(snapshot.audio)}
    scene=plan.bind(audios)
    if scene.frames*8>MAX_OUTPUT_PCM_BYTES:
        raise AudioLimitError("Spatial output exceeds offline PCM byte budget")
    result=render_spatial_scene(scene)
    if np.max(np.abs(result.audio.samples))>1:
        raise AudioContractError("Spatial output exceeds normalized PCM")
    encoded=encode_wav_bytes(result.audio,settings=quantization)
    for binding in bindings.values():
        _distinct(binding.path,target)
    publication=publish_generated_wav(target,encoded,overwrite=overwrite,
        create_parent=create_parent,expected_sha256=expected_output_sha256)
    return SpatialExport(plan.sha256,receipts,publication.wav_sha256,
        digest_audio(result.audio),publication.bytes_written,output_bits,dither_seed,result.report)
