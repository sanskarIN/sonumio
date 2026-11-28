"""Seed-separated first-party percussion source and independent excitation.

Source waveforms use the existing procedural instrument engine. The learned
model receives a separate random excitation stream, not the source noise or
its exact realization. The objective compares distributions of features,
rather than requiring a random waveform to reproduce another random phase.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch

from sonumio.arrangement import Event
from sonumio.music_instruments import render_event

PROTOCOL = "sonumio-percussion-source-v1"
DRUMS = ("kick", "snare", "hat")
PITCHES = (36, 38, 42)
MIN_FRAMES = 2048
MAX_FRAMES = 8192
MAX_BATCH = 16
MAX_ROWS = 512
MAX_SOURCE_SAMPLES = 4_000_000
MAX_UPDATES = 256
PANELS = ("train", "seed", "velocity", "duration", "joint")


def _int(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError("Invalid percussion " + label)
    return value


def _axis(values, lo, hi, label):
    if (type(values) is not tuple or not 1 <= len(values) <= 8
            or any(type(v) is not int or not lo <= v <= hi for v in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError("Invalid percussion " + label + " axis")


@dataclass(frozen=True, slots=True)
class DrumSpec:
    drum: str
    velocity_milli: int
    frames: int
    seed: int

    def __post_init__(self):
        if type(self.drum) is not str or self.drum not in DRUMS:
            raise ValueError("Unknown percussion class")
        _int(self.velocity_milli,100,1000,"velocity")
        _int(self.frames,MIN_FRAMES,MAX_FRAMES,"duration")
        _int(self.seed,0,2**32-1,"source seed")

    @property
    def index(self):
        return DRUMS.index(self.drum)

    @property
    def identity(self):
        payload = json.dumps([self.drum,self.velocity_milli,self.frames,self.seed],separators=(",",":"))
        return hashlib.sha256(PROTOCOL.encode()+b"\0"+payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class DrumExample:
    spec: DrumSpec
    raw: bytes

    def __post_init__(self):
        if not isinstance(self.spec,DrumSpec) or type(self.raw) is not bytes or len(self.raw)!=4*self.spec.frames:
            raise ValueError("Invalid immutable percussion PCM")
        if not np.isfinite(self.samples).all() or not np.any(self.samples) or np.max(np.abs(self.samples))>1.000001:
            raise ValueError("Invalid percussion source waveform")

    @property
    def samples(self):
        return np.frombuffer(self.raw,dtype="<f4")

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()


@dataclass(frozen=True, slots=True)
class DrumGrid:
    training_seeds: tuple = (7,19,31)
    evaluation_seeds: tuple = (103,107)
    training_velocities: tuple = (300,700,950)
    evaluation_velocities: tuple = (500,850)
    training_frames: tuple = (4096,8192)
    evaluation_frames: tuple = (6144,)

    def __post_init__(self):
        for a,b,lo,hi,label in (
            (self.training_seeds,self.evaluation_seeds,0,2**32-1,"seed"),
            (self.training_velocities,self.evaluation_velocities,100,1000,"velocity"),
            (self.training_frames,self.evaluation_frames,MIN_FRAMES,MAX_FRAMES,"duration"),
        ):
            _axis(a,lo,hi,label)
            _axis(b,lo,hi,label)
            if set(a)&set(b):
                raise ValueError("Percussion training/evaluation "+label+" overlap")
        specs = tuple(spec for panel in PANELS for spec in self.specs(panel))
        if len(specs)>MAX_ROWS or sum(spec.frames for spec in specs)>MAX_SOURCE_SAMPLES:
            raise ValueError("Percussion source budget exceeded")

    def specs(self,panel):
        if panel not in PANELS:
            raise ValueError("Unknown percussion source panel")
        seeds = self.training_seeds if panel=="train" else self.evaluation_seeds
        velocities = self.evaluation_velocities if panel in ("velocity","joint") else self.training_velocities
        lengths = self.evaluation_frames if panel in ("duration","joint") else self.training_frames
        return tuple(DrumSpec(d,v,n,s) for d in DRUMS for v in velocities for n in lengths for s in seeds)


def render_source(spec):
    if not isinstance(spec,DrumSpec):
        raise ValueError("Expected DrumSpec")
    event=Event(spec.drum,PITCHES[spec.index],0,1,spec.velocity_milli/1000.,0.)
    pcm=render_event(event,spec.frames,spec.seed)
    return DrumExample(spec,np.ascontiguousarray(pcm,dtype="<f4").tobytes())


@dataclass(frozen=True, slots=True)
class DrumCorpus:
    grid: DrumGrid
    panels: tuple

    def __post_init__(self):
        if not isinstance(self.grid,DrumGrid) or type(self.panels) is not tuple or len(self.panels)!=len(PANELS):
            raise ValueError("Invalid percussion corpus")
        all_ids=set()
        train_pcm=set()
        for name,rows in zip(PANELS,self.panels,strict=True):
            if (type(rows) is not tuple or any(not isinstance(row,DrumExample) for row in rows)
                    or tuple(row.spec for row in rows)!=self.grid.specs(name)):
                raise ValueError("Unexpected percussion source panel")
            ids={row.spec.identity for row in rows}
            pcm={row.sha256 for row in rows}
            if len(ids)!=len(rows) or ids&all_ids:
                raise ValueError("Percussion source identity overlap")
            all_ids.update(ids)
            if name=="train":
                train_pcm=pcm
            elif pcm&train_pcm:
                raise ValueError("Percussion source PCM overlaps training")

    def get(self,panel):
        return self.panels[PANELS.index(panel)]

    def report(self):
        result={}
        for name in PANELS:
            rows=self.get(name)
            trace=hashlib.sha256(PROTOCOL.encode()+b"\0panel\0")
            for row in rows:
                trace.update(bytes.fromhex(row.spec.identity))
                trace.update(bytes.fromhex(row.sha256))
            result[name]={"rows":len(rows),"samples":sum(row.spec.frames for row in rows),
                          "source_seeds":sorted({row.spec.seed for row in rows}),"sha256":trace.hexdigest()}
        return result


def build_corpus(grid=None):
    grid=DrumGrid() if grid is None else grid
    if not isinstance(grid,DrumGrid):
        raise ValueError("Expected DrumGrid")
    return DrumCorpus(grid,tuple(tuple(render_source(spec) for spec in grid.specs(panel)) for panel in PANELS))


@dataclass(frozen=True,slots=True)
class DrumBatch:
    labels: torch.Tensor
    velocities: torch.Tensor
    frames: int
    targets: torch.Tensor
    excitation: torch.Tensor


def excitation(spec,domain,replica):
    """Generate reproducible but source-independent excitation for one hit."""
    if not isinstance(spec,DrumSpec) or domain not in ("train","evaluation"):
        raise ValueError("Invalid excitation domain")
    _int(replica,0,2**32-1,"excitation replica")
    # Independent namespace and source-recipe hash, with no source PCM read.
    digest=hashlib.sha256(b"sonumio-independent-excitation-v1\0"+domain.encode()+b"\0"+
                          bytes.fromhex(spec.identity)+replica.to_bytes(4,"little")).digest()
    rng=np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype="<u4").tolist()))
    return rng.uniform(-1.,1.,spec.frames+2).astype(np.float32)


def collate(rows,*,domain,replica):
    if (type(rows) is not tuple or not 1<=len(rows)<=MAX_BATCH or
            any(not isinstance(row,DrumExample) for row in rows)):
        raise ValueError("Invalid percussion batch")
    frames=rows[0].spec.frames
    if any(row.spec.frames!=frames for row in rows):
        raise ValueError("Percussion batch must share duration")
    return DrumBatch(torch.tensor([row.spec.index for row in rows],dtype=torch.long),
                     torch.tensor([row.spec.velocity_milli for row in rows],dtype=torch.long),frames,
                     torch.from_numpy(np.stack([row.samples for row in rows]).astype(np.float32)),
                     torch.from_numpy(np.stack([excitation(row.spec,domain,replica) for row in rows])))


def _groups(rows):
    if type(rows) is not tuple or not rows or any(not isinstance(row,DrumExample) for row in rows):
        raise ValueError("Expected percussion rows")
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(row.spec.frames,[]).append(i)
    return {n:tuple(groups[n]) for n in sorted(groups)}


def schedule(rows,*,updates,batch_size,seed):
    _int(updates,1,MAX_UPDATES,"updates")
    _int(batch_size,1,MAX_BATCH,"batch size")
    _int(seed,0,2**32-1,"sampler seed")
    groups=_groups(rows)
    lengths=tuple(groups)
    rng=np.random.default_rng(np.random.SeedSequence([seed,0x4452554D]))
    pending={n:[] for n in lengths}
    batches=[]
    trace=hashlib.sha256(PROTOCOL.encode()+b"\0schedule\0")
    for step in range(updates):
        n=lengths[step%len(lengths)]
        while len(pending[n])<batch_size:
            pending[n].extend(int(i) for i in rng.permutation(groups[n]))
        indices=tuple(pending[n][:batch_size])
        del pending[n][:batch_size]
        batches.append(indices)
        for i in indices:
            trace.update(bytes.fromhex(rows[i].spec.identity))
    return tuple(batches),trace.hexdigest()


def evaluation_batches(rows,batch_size):
    _int(batch_size,1,MAX_BATCH,"evaluation batch size")
    for group in _groups(rows).values():
        for i in range(0,len(group),batch_size):
            yield tuple(rows[j] for j in group[i:i+batch_size])
