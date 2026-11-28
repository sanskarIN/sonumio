"""Deterministic edge-distance sampling for sustained instrument research.

The same schedule is used by every factorial arm and initialization seed.
The source examples come exclusively from the existing disjoint training
panel. The sampler covers the actual endpoint and multiple distances
from it, avoiding an edge objective trained only at the last 2048 samples.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from research.instrument_long_data import (
    Example,WindowBatch,REGIONS,WINDOW,MAX_ROWS,MAX_UPDATES,MAX_BATCH,region_span,
)

PROTOCOL="sonumio-edge-distance-schedule-v1"
DISTANCES=(0,512,1536,3072,6144,12288)


@dataclass(frozen=True,slots=True)
class Schedule:
    batches:tuple[WindowBatch,...]
    sha256:str
    distances:tuple[int,...]

    def __post_init__(self):
        if (type(self.batches) is not tuple or not 1<=len(self.batches)<=MAX_UPDATES
                or any(not isinstance(b,WindowBatch) for b in self.batches)
                or type(self.distances) is not tuple or len(self.distances)!=len(self.batches)
                or any(type(d) is not int or d<0 for d in self.distances)
                or type(self.sha256) is not str or len(self.sha256)!=64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("Invalid edge-distance schedule")


def build_schedule(rows,*,updates,batch_size,seed):
    if (type(rows) is not tuple or not 1<=len(rows)<=MAX_ROWS
            or any(not isinstance(row,Example) for row in rows)
            or len({row.note.identity for row in rows})!=len(rows)
            or type(updates) is not int or not 1<=updates<=MAX_UPDATES
            or type(batch_size) is not int or not 1<=batch_size<=MAX_BATCH
            or type(seed) is not int or not 0<=seed<2**32):
        raise ValueError("Invalid edge-distance sampling context")
    groups=defaultdict(list)
    for index,row in enumerate(rows):
        groups[row.note.frames].append(index)
    lengths=tuple(sorted(groups))
    pending={n:[] for n in lengths}
    rng=np.random.default_rng(np.random.SeedSequence([seed,0x45444745]))
    batches=[]
    distances=[]
    trace=hashlib.sha256(PROTOCOL.encode("ascii")+b"\0")
    for step in range(updates):
        region=REGIONS[step%len(REGIONS)]
        n=lengths[(step//len(REGIONS))%len(lengths)]
        cycle=step//(len(REGIONS)*len(lengths))
        lo,hi=region_span(n,region)
        # A batch's duration is homogeneous. Edge distances are clamped
        # only to the region's valid window domain, never across a region.
        available=hi-lo-WINDOW
        distance=min(DISTANCES[cycle%len(DISTANCES)],available)
        if region=="attack":
            start=lo+distance
        elif region=="release":
            start=hi-WINDOW-distance
        else:
            start=int(rng.integers(lo,hi-WINDOW+1))
            distance=start-lo
        while len(pending[n])<batch_size:
            pending[n].extend(int(i) for i in rng.permutation(groups[n]))
        indices=tuple(pending[n][:batch_size])
        del pending[n][:batch_size]
        batch=WindowBatch(indices,n,start,WINDOW,region)
        batches.append(batch)
        distances.append(distance)
        trace.update(json.dumps([n,start,WINDOW,region,distance],separators=(",",":")).encode("ascii"))
        for i in indices:
            trace.update(bytes.fromhex(rows[i].note.identity))
    return Schedule(tuple(batches),trace.hexdigest(),tuple(distances))


def describe_schedule(schedule,rows):
    if not isinstance(schedule,Schedule) or type(rows) is not tuple:
        raise ValueError("Expected verified edge schedule and source rows")
    by_region={name:{"batches":0,"samples":0,"distances":set(),"notes":set()} for name in REGIONS}
    for batch,distance in zip(schedule.batches,schedule.distances,strict=True):
        row=by_region[batch.region]
        row["batches"]+=1
        row["samples"]+=len(batch.indices)*batch.count
        row["distances"].add(distance)
        for index in batch.indices:
            if index>=len(rows) or rows[index].note.frames!=batch.frames:
                raise ValueError("Edge-distance schedule references invalid source")
            row["notes"].add(rows[index].note.identity)
    return {"sha256":schedule.sha256,"updates":len(schedule.batches),
            "by_region":{region:{"batches":x["batches"],"samples":x["samples"],
                "distances":sorted(x["distances"]),"unique_notes":len(x["notes"])}
                for region,x in by_region.items()}}
