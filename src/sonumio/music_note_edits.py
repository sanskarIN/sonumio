"""Bounded note-level editing of deterministic studio compositions.

Edits address the original canonical event index, pinned by the original
score digest. All removals, replacements and additions form one atomic
transaction; intermediate invalid polyphony is not rendered.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .arrangement import Arrangement, Event, TRACKS, MAX_EVENTS, event_key, arrangement_digest
from .errors import GenerationError

MAX_NOTE_EDITS = 128
MAX_STUDIO_TICKS = 16 * 32
Track = Literal["keys","pad","bass","lead","pluck","kick","snare","hat"]
Digest = Annotated[str,Field(min_length=64,max_length=64,pattern=r"^[0-9a-f]{64}$")]


class EditModel(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True,frozen=True)


class StudioNote(EditModel):
    track: Track
    pitch: int=Field(ge=24,le=96)
    tick: int=Field(ge=0,lt=MAX_STUDIO_TICKS)
    length: int=Field(ge=1,le=MAX_STUDIO_TICKS)
    velocity: float=Field(ge=.001,le=1,allow_inf_nan=False)
    pan: float=Field(ge=-1,le=1,allow_inf_nan=False)

    def event(self):
        return Event(**self.model_dump())


class Replacement(EditModel):
    index: int=Field(ge=0,lt=MAX_EVENTS)
    note: StudioNote


class NoteEdits(EditModel):
    base_score_sha256: Digest
    remove: list[Annotated[int,Field(ge=0,lt=MAX_EVENTS)]]=Field(default_factory=list,max_length=MAX_NOTE_EDITS)
    replace: list[Replacement]=Field(default_factory=list,max_length=MAX_NOTE_EDITS)
    add: list[StudioNote]=Field(default_factory=list,max_length=MAX_NOTE_EDITS)


def validate_edits(edits):
    if not isinstance(edits,NoteEdits):
        raise GenerationError("Expected bounded studio note transaction")
    count=len(edits.remove)+len(edits.replace)+len(edits.add)
    if not 1<=count<=MAX_NOTE_EDITS:
        raise GenerationError("Studio note transaction must contain 1–128 edits")
    removes=tuple(edits.remove)
    replaces=tuple(item.index for item in edits.replace)
    if removes!=tuple(sorted(set(removes))) or replaces!=tuple(sorted(set(replaces))):
        raise GenerationError("Studio note edit indices must be unique and sorted")
    if set(removes)&set(replaces):
        raise GenerationError("Cannot remove and replace the same source note")
    return count


def apply_note_edits(score,edits):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected validated studio base score")
    count=validate_edits(edits)
    before=arrangement_digest(score)
    if before!=edits.base_score_sha256:
        raise GenerationError("Studio base score digest mismatch; recompose before editing")
    removed=set(edits.remove)
    changed={item.index:item.note.event() for item in edits.replace}
    if any(index>=len(score.events) for index in (*removed,*changed)):
        raise GenerationError("Studio note index does not exist in the original score")
    output=[]
    for index,event in enumerate(score.events):
        if index not in removed:
            output.append(changed.get(index,event))
    output.extend(note.event() for note in edits.add)
    result=replace(score,events=tuple(sorted(output,key=event_key)))
    if arrangement_digest(score)!=before:
        raise GenerationError("Studio note transaction mutated the source score")
    return result,{"base_score_sha256":before,"score_sha256":arrangement_digest(result),
                   "operations":count,"removed":len(removed),"replaced":len(changed),
                   "added":len(edits.add),"original_unchanged":True}
