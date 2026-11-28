"""Auditable score revision deltas, conflict detection and three-way merge.

All comparisons use stable track and note IDs; relative array order is not
an identity, and revision counts are metadata rather than authentication.
A merge starts from an explicit common base and rejects conflicting edits
instead of discarding either side silently. There are no remote operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .contracts import MidiIntegrityError, MidiLimitError, MidiWorkstationError, integer
from .score import Score, ScoreNote, ScoreTrack, note_sort_key

MAX_DIFF_ITEMS = 8192
MAX_MERGE_TRACKS = 48


@dataclass(frozen=True, slots=True)
class ValueChange:
    """Single structural object change at a stable, human-readable path."""

    location: str
    old: Any
    new: Any

    def __post_init__(self):
        if type(self.location) is not str or not self.location or len(self.location) > 200:
            raise MidiWorkstationError("Revision change requires bounded location")


@dataclass(frozen=True, slots=True)
class ScoreDifference:
    from_digest: str
    to_digest: str
    changes: tuple[ValueChange, ...]
    tracks_added: int
    tracks_removed: int
    notes_added: int
    notes_removed: int
    notes_updated: int
    protocol: str = "sonumio-workstation-score-diff-v1"

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "from_digest": self.from_digest,
            "to_digest": self.to_digest,
            "tracks_added": self.tracks_added,
            "tracks_removed": self.tracks_removed,
            "notes_added": self.notes_added,
            "notes_removed": self.notes_removed,
            "notes_updated": self.notes_updated,
            "changes": [asdict(item) for item in self.changes],
            "source_mutated": False,
            "remote_writes": False,
        }


@dataclass(frozen=True, slots=True)
class MergeConflict:
    path: str
    base: Any
    ours: Any
    theirs: Any


@dataclass(frozen=True, slots=True)
class MergeResult:
    score: Score | None
    conflicts: tuple[MergeConflict, ...]
    base_digest: str
    ours_digest: str
    theirs_digest: str
    result_digest: str | None
    shared_changes: int
    independent_changes: int
    protocol: str = "sonumio-workstation-three-way-merge-v1"

    @property
    def succeeded(self):
        return self.score is not None and not self.conflicts

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "base_sha256": self.base_digest,
            "ours_sha256": self.ours_digest,
            "theirs_sha256": self.theirs_digest,
            "result_sha256": self.result_digest,
            "merged": self.succeeded,
            "conflicts": [asdict(c) for c in self.conflicts],
            "shared_changes": self.shared_changes,
            "independent_changes": self.independent_changes,
            "source_mutated": False,
            "no_external_mutation": True,
        }


_ABSENT = object()


def _display(value):
    if value is _ABSENT:
        return None
    if isinstance(value, (ScoreTrack, ScoreNote)):
        return asdict(value)
    if isinstance(value, tuple):
        return [asdict(x) if hasattr(x, "__dataclass_fields__") else x for x in value]
    return value


def _map_notes(track: ScoreTrack | object):
    if track is _ABSENT:
        return {}
    return {n.id: n for n in track.notes}


def _map_tracks(score: Score):
    return {t.id: t for t in score.tracks}


def _meta(track):
    if track is _ABSENT:
        return _ABSENT
    return (track.name, track.channel, track.program,
            track.bank_msb, track.bank_lsb, track.muted, track.solo)


def _compare(base, updated, path, output):
    if base != updated:
        output.append(ValueChange(path, _display(base), _display(updated)))
        if len(output) > MAX_DIFF_ITEMS:
            raise MidiLimitError("Too many revision changes")


def diff_scores(before: Score, after: Score) -> ScoreDifference:
    if not isinstance(before, Score) or not isinstance(after, Score):
        raise MidiWorkstationError("Expected two typed Scores")
    changes: list[ValueChange] = []
    for field in ("title", "ppqn", "end", "tempo", "meter"):
        _compare(getattr(before, field), getattr(after, field), field, changes)
    old_tracks = _map_tracks(before)
    new_tracks = _map_tracks(after)
    added_tracks = removed_tracks = added = removed = changed = 0
    for track_id in sorted(set(old_tracks) | set(new_tracks)):
        previous = old_tracks.get(track_id, _ABSENT)
        current = new_tracks.get(track_id, _ABSENT)
        location = f"tracks/{track_id}"
        if previous is _ABSENT:
            added_tracks += 1
        if current is _ABSENT:
            removed_tracks += 1
        _compare(_meta(previous), _meta(current), location + "/properties", changes)
        old_notes = _map_notes(previous)
        new_notes = _map_notes(current)
        for note_id in sorted(set(old_notes) | set(new_notes)):
            old_note = old_notes.get(note_id, _ABSENT)
            new_note = new_notes.get(note_id, _ABSENT)
            _compare(old_note, new_note, location + f"/notes/{note_id}", changes)
            if old_note is _ABSENT:
                added += 1
            elif new_note is _ABSENT:
                removed += 1
            elif old_note != new_note:
                changed += 1
    return ScoreDifference(before.sha256, after.sha256, tuple(changes),
                           added_tracks, removed_tracks, added, removed, changed)


def _choose(base, ours, theirs, path, conflicts, stats):
    """Return nonconflicting value. Explicit sentinel avoids conflating removal with None."""
    if ours == theirs:
        if ours != base:
            stats[0] += 1
        return ours
    if ours == base:
        stats[1] += 1
        return theirs
    if theirs == base:
        stats[1] += 1
        return ours
    conflicts.append(MergeConflict(path, _display(base),
                                   _display(ours), _display(theirs)))
    return _ABSENT


def _three_way_track(base, ours, theirs, track_id, conflicts, stats):
    path = f"tracks/{track_id}"
    if base is _ABSENT:
        if ours is _ABSENT or theirs is _ABSENT:
            return _choose(base, ours, theirs, path, conflicts, stats)
        if _meta(ours) != _meta(theirs):
            conflicts.append(MergeConflict(path, None, _display(ours), _display(theirs)))
            return _ABSENT
        properties = _meta(ours)
    else:
        if ours is _ABSENT or theirs is _ABSENT:
            return _choose(base, ours, theirs, path, conflicts, stats)
        properties = _choose(_meta(base), _meta(ours), _meta(theirs),
                             path + "/properties", conflicts, stats)
        if properties is _ABSENT:
            return _ABSENT
    notes = []
    old_map = _map_notes(base)
    our_map = _map_notes(ours)
    their_map = _map_notes(theirs)
    for note_id in sorted(set(old_map) | set(our_map) | set(their_map)):
        previous = old_map.get(note_id, _ABSENT)
        mine = our_map.get(note_id, _ABSENT)
        other = their_map.get(note_id, _ABSENT)
        result = _choose(previous, mine, other,
                         path + f"/notes/{note_id}", conflicts, stats)
        if result is not _ABSENT:
            notes.append(result)
    if conflicts:
        return _ABSENT
    return ScoreTrack(track_id, properties[0], properties[1], properties[2],
                      properties[3], properties[4],
                      tuple(sorted(notes, key=note_sort_key)),
                      properties[5], properties[6])


def merge_scores(base: Score, ours: Score, theirs: Score,
                 *, strict_revision: bool = False) -> MergeResult:
    """Three-way ID-aware merge; return conflicts, never silently choose a side.

    The score clock and track property changes are atomic; note edits are
    resolved independently by stable ID. Track deletion vs. modification is
    a conflict. No output score is constructed if any conflict exists.
    """
    if any(not isinstance(x, Score) for x in (base, ours, theirs)):
        raise MidiWorkstationError("Three-way merge requires typed Scores")
    if type(strict_revision) is not bool:
        raise MidiWorkstationError("strict_revision must be bool")
    if strict_revision and (ours.revision < base.revision
                            or theirs.revision < base.revision):
        raise MidiIntegrityError("Merge branches predate base revision")
    conflicts: list[MergeConflict] = []
    stats = [0, 0]
    fields = {}
    for field in ("title", "ppqn", "end", "tempo", "meter"):
        fields[field] = _choose(getattr(base, field), getattr(ours, field),
                                getattr(theirs, field), field, conflicts, stats)
    base_tracks = _map_tracks(base)
    our_tracks = _map_tracks(ours)
    their_tracks = _map_tracks(theirs)
    result_tracks = []
    for track_id in sorted(set(base_tracks) | set(our_tracks) | set(their_tracks)):
        if len(result_tracks) > MAX_MERGE_TRACKS:
            raise MidiLimitError("Merged score exceeds bounded track count")
        track = _three_way_track(base_tracks.get(track_id, _ABSENT),
                                 our_tracks.get(track_id, _ABSENT),
                                 their_tracks.get(track_id, _ABSENT),
                                 track_id, conflicts, stats)
        if track is not _ABSENT:
            result_tracks.append(track)
    if not conflicts:
        # Existing score constraints are checked again before publication.
        # Conflicting patch routing, boundaries or note counts cause a
        # fail-closed exception rather than an invalid merged document.
        next_revision = max(base.revision, ours.revision, theirs.revision) + 1
        if next_revision > 2**31 - 1:
            raise MidiLimitError("Merged score revision overflow")
        result = Score(fields["title"], fields["ppqn"], tuple(result_tracks),
                       fields["tempo"], fields["meter"], fields["end"],
                       next_revision)
    else:
        result = None
    return MergeResult(result, tuple(conflicts), base.sha256, ours.sha256,
                       theirs.sha256, result.sha256 if result else None,
                       stats[0], stats[1])
