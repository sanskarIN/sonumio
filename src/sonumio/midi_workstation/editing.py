"""Atomic, digest-pinned symbolic score edit transactions.

Each operation works on a detached in-memory copy of the score. The source
score remains immutable, and all proposed changes undergo complete Score
validation before any new score is returned. This module has no GitHub,
filesystem, network or database side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, TypeAlias

from .contracts import (
    MAX_OPERATION_ITEMS, MAX_SCORE_TRACKS, MAX_TICK,
    MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    integer, sha256, text_value,
)
from .score import (
    Score, ScoreNote, ScoreTrack, note_sort_key,
)
from .tempo import TempoPoint, MeterPoint


def _track_id(value: str) -> str:
    if type(value) is not str or not value or len(value) > 64:
        raise MidiWorkstationError("Invalid symbolic track ID")
    return value


def _note_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    if type(value) is not tuple or not value or len(value) > MAX_OPERATION_ITEMS:
        raise MidiWorkstationError("Expected bounded nonempty note ID tuple")
    if any(type(item) is not str or not item for item in value):
        raise MidiWorkstationError("Invalid note identifier in selection")
    if len(value) != len(set(value)):
        raise MidiWorkstationError("Duplicate note identifier in selection")
    return value


@dataclass(frozen=True, slots=True)
class AddNote:
    track_id: str
    note: ScoreNote

    def __post_init__(self):
        _track_id(self.track_id)
        if not isinstance(self.note, ScoreNote):
            raise MidiWorkstationError("AddNote requires typed ScoreNote")


@dataclass(frozen=True, slots=True)
class RemoveNotes:
    track_id: str
    note_ids: tuple[str, ...]

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)


@dataclass(frozen=True, slots=True)
class ReplaceNote:
    track_id: str
    note_id: str
    replacement: ScoreNote

    def __post_init__(self):
        _track_id(self.track_id)
        _track_id(self.note_id)
        if not isinstance(self.replacement, ScoreNote):
            raise MidiWorkstationError("Replacement requires ScoreNote")
        if self.replacement.id != self.note_id:
            raise MidiWorkstationError("Replacement must retain stable note identifier")


@dataclass(frozen=True, slots=True)
class ShiftNotes:
    track_id: str
    note_ids: tuple[str, ...]
    ticks: int

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)
        integer(self.ticks, "note shift", -MAX_TICK, MAX_TICK)


@dataclass(frozen=True, slots=True)
class TransposeNotes:
    track_id: str
    note_ids: tuple[str, ...]
    semitones: int

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)
        integer(self.semitones, "note transposition", -127, 127)


@dataclass(frozen=True, slots=True)
class ScaleNoteVelocities:
    track_id: str
    note_ids: tuple[str, ...]
    numerator: int
    denominator: int

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)
        integer(self.numerator, "velocity multiplier numerator", 0, 1024)
        integer(self.denominator, "velocity multiplier denominator", 1, 1024)


@dataclass(frozen=True, slots=True)
class ResizeNotes:
    track_id: str
    note_ids: tuple[str, ...]
    numerator: int
    denominator: int
    pivot: str = "onset"

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)
        integer(self.numerator, "note stretch numerator", 1, 1024)
        integer(self.denominator, "note stretch denominator", 1, 1024)
        if self.pivot not in ("onset", "release"):
            raise MidiWorkstationError("Unknown note stretch anchor")


@dataclass(frozen=True, slots=True)
class DuplicateNotes:
    track_id: str
    note_ids: tuple[str, ...]
    ticks: int
    suffix: str

    def __post_init__(self):
        _track_id(self.track_id)
        _note_ids(self.note_ids)
        integer(self.ticks, "duplicate offset", -MAX_TICK, MAX_TICK)
        if (type(self.suffix) is not str or not 1 <= len(self.suffix) <= 24
                or not self.suffix.replace("-", "").replace("_", "").isalnum()):
            raise MidiWorkstationError("Invalid duplicate identifier suffix")


@dataclass(frozen=True, slots=True)
class SplitNote:
    track_id: str
    note_id: str
    split_tick: int
    right_id: str

    def __post_init__(self):
        _track_id(self.track_id)
        _track_id(self.note_id)
        _track_id(self.right_id)
        integer(self.split_tick, "note split tick", 0, MAX_TICK)
        if self.note_id == self.right_id:
            raise MidiWorkstationError("Split halves require distinct note IDs")


@dataclass(frozen=True, slots=True)
class DeleteRegion:
    track_id: str
    start: int
    stop: int
    ripple: bool = False
    split_crossing: bool = True

    def __post_init__(self):
        _track_id(self.track_id)
        integer(self.start, "region start", 0, MAX_TICK)
        integer(self.stop, "region stop", self.start + 1, MAX_TICK)
        if type(self.ripple) is not bool or type(self.split_crossing) is not bool:
            raise MidiWorkstationError("Region edit options must be bool")


@dataclass(frozen=True, slots=True)
class SetTrackFlags:
    track_id: str
    muted: bool
    solo: bool

    def __post_init__(self):
        _track_id(self.track_id)
        if type(self.muted) is not bool or type(self.solo) is not bool:
            raise MidiWorkstationError("Track flags must be bool")


@dataclass(frozen=True, slots=True)
class RenameTrack:
    track_id: str
    name: str

    def __post_init__(self):
        _track_id(self.track_id)
        text_value(self.name, "track display name")


@dataclass(frozen=True, slots=True)
class ChangePatch:
    track_id: str
    program: int
    bank_msb: int
    bank_lsb: int

    def __post_init__(self):
        _track_id(self.track_id)
        integer(self.program, "program number", 0, 127)
        integer(self.bank_msb, "bank MSB", 0, 127)
        integer(self.bank_lsb, "bank LSB", 0, 127)


@dataclass(frozen=True, slots=True)
class AddTrack:
    track: ScoreTrack

    def __post_init__(self):
        if not isinstance(self.track, ScoreTrack):
            raise MidiWorkstationError("AddTrack requires ScoreTrack")


@dataclass(frozen=True, slots=True)
class RemoveTrack:
    track_id: str

    def __post_init__(self):
        _track_id(self.track_id)


@dataclass(frozen=True, slots=True)
class SetTempo:
    tick: int
    microseconds_per_quarter: int

    def __post_init__(self):
        TempoPoint(self.tick, self.microseconds_per_quarter)


@dataclass(frozen=True, slots=True)
class RemoveTempo:
    tick: int

    def __post_init__(self):
        integer(self.tick, "tempo removal tick", 1, MAX_TICK)


@dataclass(frozen=True, slots=True)
class SetMeter:
    point: MeterPoint

    def __post_init__(self):
        if not isinstance(self.point, MeterPoint):
            raise MidiWorkstationError("Expected MeterPoint")


Operation: TypeAlias = (
    AddNote | RemoveNotes | ReplaceNote | ShiftNotes | TransposeNotes
    | ScaleNoteVelocities | ResizeNotes | DuplicateNotes | SplitNote
    | DeleteRegion | SetTrackFlags | RenameTrack | ChangePatch
    | AddTrack | RemoveTrack | SetTempo | RemoveTempo | SetMeter
)
_OPERATION_TYPES = (
    AddNote, RemoveNotes, ReplaceNote, ShiftNotes, TransposeNotes,
    ScaleNoteVelocities, ResizeNotes, DuplicateNotes, SplitNote,
    DeleteRegion, SetTrackFlags, RenameTrack, ChangePatch,
    AddTrack, RemoveTrack, SetTempo, RemoveTempo, SetMeter,
)


@dataclass(frozen=True, slots=True)
class EditTransaction:
    base_sha256: str
    operations: tuple[Operation, ...]
    expected_revision: int | None = None

    def __post_init__(self):
        sha256(self.base_sha256, "base score digest")
        if (type(self.operations) is not tuple
                or not 1 <= len(self.operations) <= MAX_OPERATION_ITEMS
                or any(not isinstance(op, _OPERATION_TYPES) for op in self.operations)):
            raise MidiWorkstationError("Invalid atomic score operation tuple")
        if self.expected_revision is not None:
            integer(self.expected_revision, "expected score revision", 0, 2**31 - 1)


@dataclass(frozen=True, slots=True)
class EditReceipt:
    base_sha256: str
    result_sha256: str
    previous_revision: int
    new_revision: int
    operations: int
    notes_before: int
    notes_after: int
    tracks_before: int
    tracks_after: int
    created_notes: int
    deleted_notes: int
    altered_notes: int
    protocol: str = "sonumio-workstation-atomic-edit-v1"

    def as_dict(self) -> dict:
        from dataclasses import asdict
        data = asdict(self)
        data["source_mutated"] = False
        data["operations_atomic"] = True
        data["model_inference"] = False
        return data


def _rounded_ratio(number: int, numerator: int, denominator: int) -> int:
    return (number * numerator * 2 + denominator) // (denominator * 2)


def _replace_ids(track: ScoreTrack, ids: tuple[str, ...], callback) -> ScoreTrack:
    selected = set(ids)
    original = {note.id for note in track.notes}
    if not selected <= original:
        raise MidiWorkstationError("Edit references a nonexistent note identifier")
    notes = tuple(callback(n) if n.id in selected else n for n in track.notes)
    return track.replace_notes(notes)


def _region_notes(track: ScoreTrack, op: DeleteRegion) -> ScoreTrack:
    selected: list[ScoreNote] = []
    offset = op.stop - op.start
    for note in track.notes:
        a, b = note.start, note.stop
        if b <= op.start:
            selected.append(note)
        elif a >= op.stop:
            selected.append(note.shift(-offset) if op.ripple else note)
        elif not op.split_crossing and (a < op.start or b > op.stop):
            raise MidiWorkstationError("Region cuts a note but split_crossing=False")
        else:
            if a < op.start:
                selected.append(note.resized(op.start - a))
            if b > op.stop:
                remnant = replace(note, id=note.id + "_right", start=op.stop,
                                  duration=b - op.stop)
                selected.append(remnant.shift(-offset) if op.ripple else remnant)
    return track.replace_notes(tuple(selected))


def _apply_track_operation(track: ScoreTrack, op: Operation) -> ScoreTrack:
    if isinstance(op, AddNote):
        if any(n.id == op.note.id for n in track.notes):
            raise MidiWorkstationError("Cannot add duplicate note ID")
        return track.replace_notes(track.notes + (op.note,))
    if isinstance(op, RemoveNotes):
        selected = set(op.note_ids)
        if not selected <= {n.id for n in track.notes}:
            raise MidiWorkstationError("Cannot remove unknown note")
        return track.replace_notes(tuple(n for n in track.notes if n.id not in selected))
    if isinstance(op, ReplaceNote):
        return _replace_ids(track, (op.note_id,), lambda _: op.replacement)
    if isinstance(op, ShiftNotes):
        return _replace_ids(track, op.note_ids, lambda note: note.shift(op.ticks))
    if isinstance(op, TransposeNotes):
        return _replace_ids(track, op.note_ids, lambda note: note.transpose(op.semitones))
    if isinstance(op, ScaleNoteVelocities):
        return _replace_ids(track, op.note_ids, lambda note: note.with_velocity(
            max(1, min(127, _rounded_ratio(note.velocity, op.numerator, op.denominator)))))
    if isinstance(op, ResizeNotes):
        def resize(note):
            size = max(1, _rounded_ratio(note.duration, op.numerator, op.denominator))
            if op.pivot == "release":
                return replace(note, start=note.stop - size, duration=size)
            return note.resized(size)
        return _replace_ids(track, op.note_ids, resize)
    if isinstance(op, DuplicateNotes):
        selected = set(op.note_ids)
        if not selected <= {n.id for n in track.notes}:
            raise MidiWorkstationError("Cannot duplicate unknown note")
        copied = tuple(replace(note.shift(op.ticks), id=note.id + "_" + op.suffix)
                       for note in track.notes if note.id in selected)
        return track.replace_notes(track.notes + copied)
    if isinstance(op, SplitNote):
        source = next((note for note in track.notes if note.id == op.note_id), None)
        if source is None or not source.start < op.split_tick < source.stop:
            raise MidiWorkstationError("Split must occur inside an existing note")
        if any(n.id == op.right_id for n in track.notes):
            raise MidiWorkstationError("Split right note ID already exists")
        left = source.resized(op.split_tick - source.start)
        right = replace(source, id=op.right_id,
                        start=op.split_tick, duration=source.stop - op.split_tick)
        return track.replace_notes(tuple(n for n in track.notes if n.id != op.note_id)
                                   + (left, right))
    if isinstance(op, DeleteRegion):
        return _region_notes(track, op)
    if isinstance(op, SetTrackFlags):
        return replace(track, muted=op.muted, solo=op.solo)
    if isinstance(op, RenameTrack):
        return replace(track, name=op.name)
    if isinstance(op, ChangePatch):
        return replace(track, program=op.program,
                       bank_msb=op.bank_msb, bank_lsb=op.bank_lsb)
    raise MidiWorkstationError("Unsupported track operation")


def apply_transaction(score: Score, tx: EditTransaction) -> tuple[Score, EditReceipt]:
    if not isinstance(score, Score) or not isinstance(tx, EditTransaction):
        raise MidiWorkstationError("Expected Score and EditTransaction")
    original_sha = score.sha256
    if tx.base_sha256 != original_sha:
        raise MidiIntegrityError("Score digest changed since edit plan was prepared")
    if tx.expected_revision is not None and tx.expected_revision != score.revision:
        raise MidiIntegrityError("Score revision does not match edit transaction")
    tracks = {track.id: track for track in score.tracks}
    order = [track.id for track in score.tracks]
    tempo = {point.tick: point for point in score.tempo}
    meter = {point.tick: point for point in score.meter}
    earliest = None
    for op in tx.operations:
        if isinstance(op, AddTrack):
            if op.track.id in tracks:
                raise MidiWorkstationError("Track already exists")
            if len(tracks) >= MAX_SCORE_TRACKS:
                raise MidiLimitError("Too many tracks for score")
            tracks[op.track.id] = op.track
            order.append(op.track.id)
            continue
        if isinstance(op, RemoveTrack):
            if op.track_id not in tracks or len(tracks) == 1:
                raise MidiWorkstationError("Cannot remove missing or final score track")
            del tracks[op.track_id]
            order.remove(op.track_id)
            continue
        if isinstance(op, SetTempo):
            tempo[op.tick] = TempoPoint(op.tick, op.microseconds_per_quarter)
            continue
        if isinstance(op, RemoveTempo):
            if op.tick not in tempo:
                raise MidiWorkstationError("Tempo point does not exist")
            del tempo[op.tick]
            continue
        if isinstance(op, SetMeter):
            meter[op.point.tick] = op.point
            continue
        if op.track_id not in tracks:
            raise MidiWorkstationError("Edit references nonexistent track")
        tracks[op.track_id] = _apply_track_operation(tracks[op.track_id], op)
    candidate = replace(
        score, tracks=tuple(tracks[id] for id in order),
        tempo=tuple(sorted(tempo.values(), key=lambda p: p.tick)),
        meter=tuple(sorted(meter.values(), key=lambda p: p.tick)),
        revision=score.revision + 1,
    )
    before = {(t.id, n.id): n for t in score.tracks for n in t.notes}
    after = {(t.id, n.id): n for t in candidate.tracks for n in t.notes}
    old, new = set(before), set(after)
    receipt = EditReceipt(
        original_sha, candidate.sha256, score.revision, candidate.revision,
        len(tx.operations), score.note_count, candidate.note_count,
        len(score.tracks), len(candidate.tracks),
        len(new - old), len(old - new),
        sum(before[k] != after[k] for k in old & new),
    )
    if score.sha256 != original_sha:
        raise MidiIntegrityError("Atomic edit unexpectedly changed base score")
    return candidate, receipt


def transaction_for(score: Score, *operations: Operation) -> EditTransaction:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected typed Score")
    return EditTransaction(score.sha256, tuple(operations), score.revision)
